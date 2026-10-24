"""One frozen, group-cross-fitted categorical interaction feature experiment."""
import sys,json,time,resource,itertools
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task,validation_split,score,split_key
from automl_models import build_pipeline
from automl_engine import _search_windows
from automl_worker import aligned_probabilities,class_decisions,train_priors
from sklearn.model_selection import StratifiedGroupKFold

class GroupOOFInteractions:
 def keys(self,X,columns):return pd.util.hash_pandas_object(X[list(columns)].astype('string').fillna('__MISSING__'),index=False).to_numpy()
 def fit_maps(self,X,target):
  prior=np.bincount(target,minlength=self.nclass)/len(target);maps=[]
  for cols in self.combinations:
   frame=pd.DataFrame({'key':self.keys(X,cols),'y':target});counts=pd.crosstab(frame.key,frame.y).reindex(columns=range(self.nclass),fill_value=0)
   values=(counts.to_numpy()+20*prior)/(counts.sum(axis=1).to_numpy()[:,None]+20)
   maps.append(pd.DataFrame(values,index=counts.index))
  return maps,prior
 def apply(self,X,state):
  maps,prior=state;parts=[]
  for cols,table in zip(self.combinations,maps):
   values=table.reindex(self.keys(X,cols)).to_numpy();unknown=~np.isfinite(values).all(axis=1);values[unknown]=prior;parts.append(values)
  return np.column_stack(parts).astype(np.float32)
 def fit_transform(self,X,y):
  self.classes_,target=np.unique(np.asarray(y),return_inverse=True);self.nclass=len(self.classes_)
  columns=[c for c in X.select_dtypes(exclude='number') if 2<=X[c].nunique()<=10000][:3]
  self.combinations=[(c,)for c in columns]+list(itertools.combinations(columns,2))
  if not columns:raise ValueError('No eligible categorical features')
  groups=pd.util.hash_pandas_object(X,index=False).to_numpy();encoded=np.zeros((len(X),len(self.combinations)*self.nclass),np.float32);self.audit=[]
  for f,v in StratifiedGroupKFold(3,shuffle=True,random_state=91).split(X,target,groups):
   assert not np.intersect1d(groups[f],groups[v]).size
   encoded[v]=self.apply(X.iloc[v],self.fit_maps(X.iloc[f],target[f]));self.audit.append({'fit':len(f),'valid':len(v),'group_overlap':0})
  self.state=self.fit_maps(X,target);return self.augment(X,encoded)
 def augment(self,X,values):
  names=[f'__oof_cat_{i}'for i in range(values.shape[1])]
  if set(names)&set(X):raise ValueError('Feature collision')
  return pd.concat([X,pd.DataFrame(values,index=X.index,columns=names)],axis=1)
 def transform(self,X):return self.augment(X,self.apply(X,self.state))

if __name__=='__main__':
 OUT=ROOT/'results/strongest-team/catprep.json';LOCAL=Path('/private/tmp/alfa-strongest-team')
 if OUT.exists():raise RuntimeError('Reuse result')
 b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Многоклассовая классификация/task.md');of,ov,_=validation_split(b);w=_search_windows(b,of,1800)[0][0]
 c=json.loads((ROOT/'results/public-v5/multiclass-adaptive/frozen_selection.json').read_text())['candidates'][-1]
 r={'status':'running','hypothesis':'ST9','candidate':c,'hashes':b.hashes,'folds':[],'outer_adaptive':True}
 def save():OUT.write_text(json.dumps(r,indent=2))
 for stage,f,v in [('inner',w['fit'],w['valid']),('outer',of,ov)]:
  if stage=='outer' and r['folds'][0]['score']<.6417083180924042:break
  save();start=time.monotonic();encoder=GroupOOFInteractions();train=encoder.fit_transform(b.X.iloc[f],b.y.iloc[f]);valid=encoder.transform(b.X.iloc[v]);preprocessing=time.monotonic()-start
  start=time.monotonic();m=build_pipeline('classification',c,8).fit(train,b.y.iloc[f]);seconds=time.monotonic()-start;p=aligned_probabilities(m,valid,b.classes);pred=class_decisions(p,b.classes,train_priors(b.y.iloc[f],b.classes),.3)
  value=score('macro_f1',b.y.iloc[v],pred);r['folds'].append({'stage':stage,'score':value,'key':split_key(f,v),'seconds':seconds,'preprocessing_seconds':preprocessing,'group_crossfit':encoder.audit,'combinations':[list(t)for t in encoder.combinations]});save()
  np.savez_compressed(LOCAL/f'catprep-{stage}.npz',positions=v,probabilities=p,prediction=pred,classes=np.asarray(b.classes));print(stage,value,flush=True)
 r['status']='completed';r['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;save()
