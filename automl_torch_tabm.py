"""CPU-only full efficient tabular ensemble, implemented with permitted PyTorch.

Uses train-fitted piecewise numeric codes with learned embeddings and per-layer
input/output adapters. No official tabm package, downloaded weights or task state.
"""
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from threadpoolctl import threadpool_limits


class PiecewiseEncoder:
    def __init__(self,bins=8): self.bins=bins
    def fit(self,X):
        values=np.asarray(X,float)
        if values.ndim!=2 or not np.isfinite(values).all(): raise ValueError('Finite numeric matrix required')
        self.left_=np.zeros((values.shape[1],self.bins))
        self.width_=np.ones_like(self.left_)
        self.mask_=np.zeros_like(self.left_,dtype=bool)
        for j in range(values.shape[1]):
            edges=np.unique(np.quantile(values[:,j],np.linspace(0,1,self.bins+1)))
            if len(edges)>1:
                n=len(edges)-1
                self.left_[j,:n]=edges[:-1]
                self.width_[j,:n]=np.diff(edges)
                self.mask_[j,:n]=True
        return self
    def transform(self,X):
        values=np.asarray(X,float)
        if values.ndim!=2 or values.shape[1]!=len(self.left_) or not np.isfinite(values).all():
            raise ValueError('Numeric schema mismatch')
        return (np.clip((values[:,:,None]-self.left_)/self.width_,0.,1.)*self.mask_).astype(np.float32)


class TorchFrameEncoder:
    def __init__(self,bins=8): self.bins=bins
    def _frame(self,X):
        if not isinstance(X,pd.DataFrame) or not X.columns.is_unique or set(X.columns)!=set(self.columns_):
            raise ValueError('Feature schema mismatch')
        frame=X.loc[:,self.columns_].copy()
        for col in self.numeric_:
            frame[col]=pd.to_numeric(frame[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
        for col in self.categorical_:
            frame[col]=frame[col].map(lambda v:'M:' if pd.isna(v) else 'V:'+str(v))
        return frame
    def fit(self,X):
        self.columns_=list(X.columns)
        self.numeric_=list(X.select_dtypes(include='number').columns)
        self.categorical_=[c for c in X if c not in self.numeric_]
        frame=self._frame(X)
        self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True) if self.numeric_ else None
        numeric=self.imputer_.fit_transform(frame[self.numeric_]) if self.numeric_ else np.empty((len(X),0))
        self.pwl_=PiecewiseEncoder(self.bins).fit(numeric)
        self.onehot_=OneHotEncoder(handle_unknown='ignore',max_categories=64,sparse_output=False,dtype=np.float32) if self.categorical_ else None
        if self.onehot_ is not None:self.onehot_.fit(frame[self.categorical_])
        return self
    def transform(self,X):
        frame=self._frame(X)
        num=self.imputer_.transform(frame[self.numeric_]) if self.imputer_ is not None else np.empty((len(X),0))
        cat=self.onehot_.transform(frame[self.categorical_]) if self.onehot_ is not None else np.empty((len(X),0),np.float32)
        return self.pwl_.transform(num),cat.astype(np.float32)


class EnsembleLinear(nn.Module):
    def __init__(self,inputs,outputs,members):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(outputs,inputs))
        self.r=nn.Parameter(torch.ones(members,inputs))
        self.s=nn.Parameter(torch.ones(members,outputs))
        self.bias=nn.Parameter(torch.zeros(members,outputs))
        nn.init.kaiming_uniform_(self.weight,a=np.sqrt(5))
    def forward(self,X):
        return F.linear(X*self.r,self.weight)*self.s+self.bias


class FullTabM(nn.Module):
    def __init__(self,numeric,categorical,outputs,members=8,width=96,bins=8,embedding=16):
        super().__init__()
        self.members=members
        self.numeric_embedding=nn.Parameter(torch.randn(numeric,bins,embedding)*.1)
        inputs=numeric*embedding+categorical
        self.layers=nn.ModuleList([EnsembleLinear(inputs,width,members),EnsembleLinear(width,width,members)])
        with torch.no_grad():
            # All embedded coordinates of one original numeric variable share
            # its initial adapter value; subsequent layers start at ones.
            sections=[embedding]*numeric+[1]*categorical
            values=torch.randn(members,len(sections))
            self.layers[0].r.copy_(torch.repeat_interleave(values,torch.tensor(sections),dim=1))
        self.head_weight=nn.Parameter(torch.empty(members,width,outputs))
        self.head_bias=nn.Parameter(torch.zeros(members,outputs))
        nn.init.uniform_(self.head_weight,-1/np.sqrt(width),1/np.sqrt(width))
        self.dropout=nn.Dropout(.1)
    def forward(self,numeric,categorical):
        embedded=torch.einsum('bnq,nqe->bne',numeric,self.numeric_embedding).flatten(1)
        x=torch.cat([embedded,categorical],dim=1)[:,None,:].expand(-1,self.members,-1)
        for layer in self.layers:x=self.dropout(F.relu(layer(x)))
        return torch.einsum('bkh,kho->bko',x,self.head_weight)+self.head_bias


class TorchTabM(BaseEstimator):
    def __init__(self,kind='regression',members=8,width=96,epochs=60,batch_size=256,
                 learning_rate=.001,weight_decay=.0001,seconds=120,seed=42,threads=4):
        self.kind=kind;self.members=members;self.width=width;self.epochs=epochs;self.batch_size=batch_size
        self.learning_rate=learning_rate;self.weight_decay=weight_decay;self.seconds=seconds
        self.seed=seed;self.threads=threads
    def configuration(self):
        return {'family':'full-tabm-pwl-v1','parameters':self.get_params(),'numeric_bins':8,'numeric_embedding':16,
                'blocks':2,'dropout':.1,'adapter_layers':'all_hidden','checkpoint':'best_completed_training_epoch',
                'class_loss':'inverse_sqrt_train_weighted_CE','reg_loss':'standardized_original_MSE',
                'target_or_query_leakage':False}
    def fit(self,X,y):
        started=time.monotonic()
        for value,maximum in [(self.members,32),(self.width,512),(self.epochs,200),(self.batch_size,4096),(self.threads,8)]:
            if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=maximum:raise ValueError('Resource setting invalid')
        if not np.isfinite(self.seconds) or not 0<self.seconds<=900:raise ValueError('Invalid time budget')
        if self.kind not in ['regression','classification']:raise ValueError('Unsupported task')
        target=np.asarray(y)
        if len(X)<2 or target.shape!=(len(X),) or pd.isna(target).any():raise ValueError('Invalid target alignment')
        self.encoder_=TorchFrameEncoder().fit(X)
        a,b=self.encoder_.transform(X)
        numeric=torch.from_numpy(a);categorical=torch.from_numpy(b)
        if self.kind=='classification':
            labels=LabelEncoder().fit(target);self.classes_=labels.classes_
            if len(self.classes_)<2:raise ValueError('At least two classes required')
            encoded=labels.transform(target)
            counts=np.bincount(encoded,minlength=len(self.classes_))
            weights=1/np.sqrt(counts.astype(float));weights/=np.mean(weights[encoded])
            self.class_weights_=weights
            class_weight=torch.tensor(weights,dtype=torch.float32)
            tensor_y=torch.tensor(encoded,dtype=torch.long);outputs=len(self.classes_)
        else:
            target=np.asarray(target,float)
            if not np.isfinite(target).all():raise ValueError('Nonfinite target')
            self.target_mean_=float(target.mean());self.target_scale_=float(target.std()) or 1.
            if not np.isfinite(self.target_scale_):raise ValueError('Target scale overflow')
            tensor_y=torch.tensor((target-self.target_mean_)/self.target_scale_,dtype=torch.float32)
            class_weight=None;outputs=1
        old_threads=torch.get_num_threads();torch.set_num_threads(self.threads)
        try:
            with torch.random.fork_rng(devices=[]),threadpool_limits(limits=self.threads):
                torch.manual_seed(self.seed)
                self.network_=FullTabM(a.shape[1],b.shape[1],outputs,self.members,self.width)
                optimizer=torch.optim.AdamW(self.network_.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
                rng=np.random.default_rng(self.seed);best=None;best_loss=float('inf');history=[];steps=0;timed=False
                for epoch in range(self.epochs):
                    self.network_.train();total=0.;seen=0
                    order=rng.permutation(len(X))
                    for start in range(0,len(X),self.batch_size):
                        if time.monotonic()-started>=self.seconds:timed=True;break
                        idx=torch.from_numpy(order[start:start+self.batch_size])
                        z=self.network_(numeric[idx],categorical[idx]);actual=tensor_y[idx]
                        if self.kind=='regression':loss=F.mse_loss(z[:,:,0],actual[:,None].expand(-1,self.members))
                        else:
                            flat=actual[:,None].expand(-1,self.members).reshape(-1)
                            loss=F.cross_entropy(z.reshape(-1,outputs),flat,weight=class_weight,reduction='none').mean()
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                        total+=float(loss.detach())*len(idx);seen+=len(idx);steps+=1
                    if timed:break
                    value=total/seen;history.append(value)
                    if value<best_loss:best_loss=value;best=copy.deepcopy(self.network_.state_dict());self.selected_epoch_=epoch+1
                    if (epoch+1)%10==0:print(f'fullTabM epoch={epoch+1} loss={value:.7f} seconds={time.monotonic()-started:.1f}',flush=True)
                if best is None:raise TimeoutError('No complete training epoch')
                self.network_.load_state_dict(best);self.network_.eval()
                self.audit_={'family':'full-tabm-pwl-v1','fit_rows':len(X),'epochs_completed':len(history),
                    'selected_epoch':self.selected_epoch_,'training_losses':history,'time_limited':timed,
                    'training_steps':steps,'seconds':time.monotonic()-started,'torch_version':torch.__version__,
                    'parameter_count':sum(p.numel() for p in self.network_.parameters()),'statistics_source':'fit_only',
                    'external_validation_used':False,'class_weights':self.class_weights_.tolist() if self.kind=='classification' else None}
        finally:torch.set_num_threads(old_threads)
        return self
    def _prediction(self,X):
        a,b=self.encoder_.transform(X)
        if not len(X):return np.empty((0,len(self.classes_))) if self.kind=='classification' else np.empty(0)
        old=torch.get_num_threads();torch.set_num_threads(self.threads)
        try:
            parts=[]
            with torch.no_grad(),threadpool_limits(limits=self.threads):
                self.network_.eval()
                for start in range(0,len(X),512):
                    z=self.network_(torch.from_numpy(a[start:start+512]),torch.from_numpy(b[start:start+512]))
                    value=torch.softmax(z,dim=-1).mean(1) if self.kind=='classification' else z[:,:,0].mean(1)
                    parts.append(value.numpy())
            p=np.concatenate(parts)
            if not np.isfinite(p).all():raise ValueError('Nonfinite model prediction')
            return p
        finally:torch.set_num_threads(old)
    def predict_proba(self,X):
        if self.kind!='classification':raise ValueError('Classification only')
        return self._prediction(X)
    def predict(self,X):
        p=self._prediction(X)
        result=self.classes_[p.argmax(1)] if self.kind=='classification' else p*self.target_scale_+self.target_mean_
        if self.kind=='regression' and not np.isfinite(result).all():raise ValueError('Prediction scale overflow')
        return result
