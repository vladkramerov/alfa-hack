"""Portable fixed classification mechanisms for SPEC-009; no experiment imports."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

STRATEGIES={'soft_distillation','balanced_softmax','nystrom_rbf','masked_augmentation'}


def _fit_once(model,X,y,configuration):
    from automl_combo_memory import fit_once
    return fit_once(model,X,y,configuration)


def softmax(logits):
    values=np.asarray(logits,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Finite matrix of margins required')
    values=values-values.max(axis=1,keepdims=True)
    exp=np.exp(values)
    return exp/exp.sum(axis=1,keepdims=True)


def _validated_targets(q,shape):
    q=np.asarray(q,dtype=float)
    if q.shape!=shape or not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(axis=1),1.,atol=1e-8):
        raise ValueError('Invalid soft-target distributions')
    return q


def soft_derivatives(margins,q,shift=None):
    """Gradient and a diagonal PSD upper bound to categorical cross-entropy Hessian.

    H=diag(p)-pp.T.  D=2diag(p(1-p)) dominates H: D-H has
    diagonal p_i sum(j!=i)p_j and off-diagonal p_i p_j, and
    v.T(D-H)v=sum(i<j)p_i*p_j*(v_i+v_j)^2 >= 0.
    Flooring D's entries at 1e-6 preserves this PSD upper bound.
    """
    margins=np.asarray(margins,dtype=float)
    q=_validated_targets(q,margins.shape)
    p=softmax(margins if shift is None else margins+np.asarray(shift,dtype=float))
    return p-q,np.maximum(2*p*(1-p),1e-6)


def soft_cross_entropy(margins,q,shift=None):
    values=np.asarray(margins,dtype=float)
    q=_validated_targets(q,values.shape)
    values=values if shift is None else values+np.asarray(shift,dtype=float)
    maximum=values.max(axis=1,keepdims=True)
    logsum=maximum+np.log(np.exp(values-maximum).sum(axis=1,keepdims=True))
    return float(-np.sum(q*(values-logsum)))


def mix_soft_targets(encoded_y,teacher_probabilities,n_classes):
    labels=np.asarray(encoded_y,dtype=int)
    if labels.ndim!=1 or np.any(labels<0) or np.any(labels>=n_classes):raise ValueError('Invalid integer labels')
    teacher=_validated_targets(teacher_probabilities,(len(labels),n_classes))
    return .7*np.eye(n_classes)[labels]+.3*teacher


class _CustomBooster:
    """fit/predict adapter so each xgboost.train call uses the persistent fit guard."""
    def __init__(self,parameters,rounds,n_classes,q,shift,threads):
        self.parameters,self.rounds,self.n_classes=parameters,rounds,n_classes
        self.q,self.shift,self.threads=q,shift,threads

    def fit(self,X,y):
        import xgboost as xgb
        values=xgb.DMatrix(X,label=np.asarray(y,dtype=int),nthread=self.threads)
        q,shift=self.q,self.shift
        def objective(margins,dmatrix):
            gradient,hessian=soft_derivatives(np.asarray(margins).reshape(len(q),self.n_classes),q,shift)
            return gradient,hessian
        settings={**self.parameters,'objective':'multi:softprob','num_class':self.n_classes,
                  'nthread':self.threads,'seed':42,'base_score':0.,'disable_default_eval_metric':1}
        self.booster_=xgb.train(settings,values,num_boost_round=self.rounds,obj=objective,verbose_eval=False)
        self.q=None
        return self

    def margins(self,X):
        import xgboost as xgb
        return np.asarray(self.booster_.predict(xgb.DMatrix(X,nthread=self.threads),output_margin=True)).reshape(len(X),self.n_classes)


class RareCategoryCap(TransformerMixin,BaseEstimator):
    """Keep <=127 frequent fit keys plus one unambiguous rare/unseen bucket."""
    def __init__(self,max_categories=128):self.max_categories=max_categories
    @staticmethod
    def _keys(series):return ('V:'+series.astype('string')).fillna('M:')
    def fit(self,X,y=None):
        if not isinstance(X,pd.DataFrame) or not X.columns.is_unique:raise ValueError('Unique-column DataFrame required')
        if not isinstance(self.max_categories,int) or self.max_categories<2:raise ValueError('Category cap must be at least2')
        self.columns_=list(X.columns)
        self.categorical_=[c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
        self.levels_={};records={}
        for c in self.categorical_:
            counts=self._keys(X[c]).value_counts(dropna=False)
            ordered=sorted(counts.index,key=lambda key:(-int(counts[key]),str(key)))
            keep=ordered if len(ordered)<=self.max_categories else ordered[:self.max_categories-1]
            self.levels_[c]=set(keep)
            records[c]={'fit_distinct':len(ordered),'kept':len(keep),'rare_bucket':len(ordered)>len(keep)}
        self.audit_={'max_fit_categories':self.max_categories,'columns':records,'fit_only':True}
        return self
    def transform(self,X):
        if any(c not in X for c in self.columns_):raise ValueError('Missing fitted input column')
        out=X[self.columns_].copy()
        for c in self.categorical_:
            keys=self._keys(out[c]);out[c]=keys.where(keys.isin(self.levels_[c]),'R:').astype(object)
        return out


def median_rbf_gamma(values):
    from scipy.spatial.distance import pdist
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all():raise ValueError('Finite kernel fit matrix required')
    rows=np.arange(len(values))
    if len(rows)>512:rows=np.random.RandomState(42).choice(rows,512,replace=False)
    distances=pdist(values[rows],metric='sqeuclidean')
    positive=distances[distances>0]
    return float(1./np.median(positive)) if len(positive) else 1.


def masked_copy(X,y):
    if len(X)!=len(y):raise ValueError('Target row mismatch')
    fields=[c for c in X.select_dtypes(include='number') if c not in {'proto','service','state','Year_Factor'}]
    generated=X.copy()
    mask=np.random.RandomState(42).random_sample((len(X),len(fields)))<.05
    if fields:generated[fields]=generated[fields].mask(mask)
    combined=pd.concat([X,generated],ignore_index=True)
    audit={'original_rows':len(X),'augmented_rows':len(generated),'numeric_fields':fields,
           'mask_probability':.05,'seed':42,'masked_cells':int(mask.sum()),
           'categorical_keys_preserved':True,'original_rows_preserved':True}
    return combined,np.concatenate([np.asarray(y),np.asarray(y)]),audit


def _row_audit(model,X,strategy):
    model.keep_mask_=np.ones(len(X),dtype=bool);model.removed_positions_=np.empty(0,dtype=int)
    model.audit_={'strategy':strategy,'fit_original_rows':len(X),'fit_retained_rows':len(X),
                  'validation_target_used':False,'outer_model_fits':1,'inner_model_fits':0}
    model.learned_state_={}


class NextBoosting:
    def __init__(self,candidate,threads=4):self.candidate,self.threads=dict(candidate),threads
    def fit(self,X,y,teacher_probabilities=None):
        from automl_models import Encoder
        from xgboost import XGBClassifier
        y=np.asarray(y);self.labels_=LabelEncoder().fit(y);self.classes_=self.labels_.classes_
        self.strategy_=self.candidate['strategy'];_row_audit(self,X,self.strategy_)
        target=self.labels_.transform(y);frame=X.copy()
        q,shift=None,None
        if self.strategy_=='soft_distillation':
            if teacher_probabilities is None:raise ValueError('Distillation requires aligned train-only teacher probabilities; teacher refit is not implicit')
            q=mix_soft_targets(target,teacher_probabilities,len(self.classes_))
            self.audit_.update(teacher_source='in_sample_meta_on_base_grouped_oof',teacher_is_independent_oof=False,
                               hard_target_weight=.7,teacher_target_weight=.3,teacher_refits=0)
        elif self.strategy_=='balanced_softmax':
            counts=np.bincount(target,minlength=len(self.classes_));shift=np.log(counts.astype(float))
            q=np.eye(len(self.classes_))[target]
            self.audit_.update(class_counts=dict(zip(map(str,self.classes_),map(int,counts))),
                               training_shift='log_fit_class_count',inference_shift='none')
        elif self.strategy_=='masked_augmentation':
            frame,masked_target,augmentation=masked_copy(X,target);target=masked_target
            self.audit_['augmentation']=augmentation
        else:raise ValueError('Unsupported boosting mechanism')
        self.encoder_=Encoder(native=False,scale=False);values=self.encoder_.fit_transform(frame,target)
        if len(self.classes_)==1:self.model_=None;self.audit_['outer_model_fits']=0;return self
        settings=dict(max_depth=6,learning_rate=.05,reg_lambda=1.,tree_method='hist',device='cpu')
        parameters=dict(self.candidate.get('parameters',{}));rounds=parameters.pop('n_estimators',600);settings.update(parameters)
        configuration={'task':'multiclass','family':'xgboost','parameters':{**settings,'n_estimators':rounds},
                       'encoder':{'native':False,'scale':False},'objective':self.strategy_}
        if q is not None:
            configuration.update(custom_objective='shifted_soft_cross_entropy',base_score=0.,
                                 soft_targets_sha256=hashlib.sha256(np.ascontiguousarray(q,dtype='<f8').tobytes()).hexdigest(),
                                 log_count_shift=None if shift is None else shift.tolist(),
                                 hessian='diagonal_PSD_upper_bound_2p1mp_floor1e-6')
            estimator=_CustomBooster(settings,rounds,len(self.classes_),q,shift,self.threads)
        else:
            configuration['augmentation']={'numeric_mask_probability':.05,'copies':1,'keys_preserved':['proto','service','state']}
            estimator=XGBClassifier(**settings,n_estimators=rounds,random_state=42,n_jobs=self.threads)
        self.model_=_fit_once(estimator,values,target,configuration)
        self.audit_.update(configuration=configuration,fit_model_rows=len(frame))
        return self
    def predict_proba(self,X):
        if self.model_ is None:return np.ones((len(X),1))
        values=self.encoder_.transform(X)
        if self.strategy_=='masked_augmentation':
            p=np.asarray(self.model_.predict_proba(values),dtype=float)
            return p/p.sum(axis=1,keepdims=True)
        # Balanced Softmax uses raw margins at inference, without log-count shift.
        return softmax(self.model_.margins(values))
    def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


class NystromClassifier:
    def __init__(self,candidate,threads=4):self.candidate,self.threads=dict(candidate),threads
    def fit(self,X,y):
        from automl_models import Encoder
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import LogisticRegression
        y=np.asarray(y);self.classes_=np.unique(y);_row_audit(self,X,'nystrom_rbf')
        self.cap_=RareCategoryCap(128).fit(X)
        self.encoder_=Encoder(native=False,scale=True)
        values=self.encoder_.fit_transform(self.cap_.transform(X))
        self.gamma_=median_rbf_gamma(values)
        self.basis_=Nystroem(kernel='rbf',gamma=self.gamma_,n_components=min(256,len(X)),random_state=42,n_jobs=self.threads)
        features=self.basis_.fit_transform(values)
        configuration={'task':'multiclass','family':'logistic_regression','parameters':{'C':1.,'solver':'lbfgs','max_iter':1000,'class_weight':None},
                       'features':{'name':'standardized_onehot_rare128_nystrom_rbf','landmarks':min(256,len(X)),
                                   'gamma':self.gamma_,'gamma_rule':'reciprocal_median_positive_squared_distance_max512_fitrows'}}
        if len(self.classes_)==1:self.model_=None;self.audit_['outer_model_fits']=0
        else:self.model_=_fit_once(LogisticRegression(C=1.,solver='lbfgs',max_iter=1000),features,y,configuration)
        self.audit_.update(configuration=configuration,rare_categories=self.cap_.audit_,gamma=self.gamma_,
                           unsupervised_basis_fit_rows=len(X),unsupervised_basis_uses_target=False,
                           landmark_fit_indices=self.basis_.component_indices_.tolist())
        return self
    def predict_proba(self,X):
        if self.model_ is None:return np.ones((len(X),1))
        features=self.basis_.transform(self.encoder_.transform(self.cap_.transform(X)))
        p=np.asarray(self.model_.predict_proba(features),dtype=float)
        return p/p.sum(axis=1,keepdims=True)
    def predict(self,X):return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


def build_next_class(candidate,threads=4):
    candidate=dict(candidate);strategy=candidate.get('strategy')
    if strategy not in STRATEGIES:raise ValueError('Unknown next-wave classification strategy')
    if not isinstance(threads,int) or isinstance(threads,bool) or not 1<=threads<=4:raise ValueError('Thread budget must be1..4')
    parameters=dict(candidate.get('parameters',{}))
    for k in ('random_state','random_seed','seed','n_jobs','thread_count','verbose','verbosity'):parameters.pop(k,None)
    allowed={'n_estimators','max_depth','learning_rate','reg_lambda','reg_alpha','subsample','colsample_bytree','tree_method','device'}
    if set(parameters)-allowed or (strategy=='nystrom_rbf' and parameters):raise ValueError('Unsupported fixed-mechanism parameters')
    for key,upper in [('n_estimators',1200),('max_depth',32)]:
        if key in parameters and (isinstance(parameters[key],bool) or not isinstance(parameters[key],int) or not 1<=parameters[key]<=upper):raise ValueError('Parameter exceeds worker resource bounds')
    if parameters.get('device','cpu')!='cpu':raise ValueError('Only CPU training is permitted')
    if candidate.get('feature_code','').strip():raise ValueError('Versioned methods do not consume generated feature code')
    candidate['parameters']=parameters
    return NystromClassifier(candidate,threads) if strategy=='nystrom_rbf' else NextBoosting(candidate,threads)
