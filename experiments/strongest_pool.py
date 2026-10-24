import sys,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task,validation_split,score,split_key
from automl_transfer import TemporalDonor
OUT=ROOT/'results/strongest-team/pool.json';LOCAL=Path('/private/tmp/alfa-strongest-team')
if OUT.exists():raise RuntimeError('Reuse')
b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Регрессия/task.md');of,ov,_=validation_split(b);t=b.X['Year_Factor'].to_numpy()
def stable(model,X,base):
 out=base.copy();periods=X.Year_Factor.to_numpy()
 for i,(key,period) in enumerate(zip(model.key_._keys(X,model.columns_),periods)):
  rec=model.records_.get(key)
  if rec is None:continue
  times,target,pos=rec;stop=np.searchsorted(times,period,side='left')
  if not stop:continue
  past=target[:stop];weights=.5**((times[stop-1]-times[:stop])/model.step_)
  alpha=min(.9,stop/(stop+1))*np.exp(-.25*max(0,(period-times[stop-1])/model.step_-1))
  out[i]=alpha*np.average(past,weights=weights)+(1-alpha)*base[i]
 return out
r={'status':'running','components':['base','stable','dynamics'],'folds':[],'outer_adaptive':True}
def save():OUT.write_text(json.dumps(r,indent=2))
for stage in ['inner','outer']:
 if stage=='inner': f=np.flatnonzero(t<4);v=np.flatnonzero(t==4)
 else:f,v=of,ov
 with np.load('/private/tmp/alfa-score-recovery-panel/'+('candidate-3-fold-0.npz' if stage=='inner' else 'selected-outer.npz')) as z:assert np.array_equal(z['positions'],v);base=z['prediction']
 with np.load(LOCAL/('residual-strict-0.npz' if stage=='inner' else 'residual-strict-selected-outer.npz')) as z:
  assert np.array_equal(z['positions'],v);dyn=z['prediction'];dyn=dyn if stage=='inner' else (dyn-.25*base)/.75
 m=TemporalDonor('Year_Factor','log').fit(b.X.iloc[f],b.y.iloc[f]);mem=stable(m,b.X.iloc[v],base);matrix=np.column_stack([base,mem,dyn]);np.savez_compressed(LOCAL/f'pool-{stage}.npz',positions=v,components=matrix)
 rec={'stage':stage,'key':split_key(f,v),'components':[score('rmse',b.y.iloc[v],matrix[:,i]) for i in range(3)],'fixed_hybrid':score('rmse',b.y.iloc[v],(mem+dyn)/2)}
 if stage=='inner':
  target=b.y.iloc[v].to_numpy();fn=lambda w:np.mean((matrix@w-target)**2)
  opt=minimize(fn,np.ones(3)/3,bounds=[(0,1)]*3,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],method='SLSQP',options={'maxiter':200,'ftol':1e-12});r['frozen_weights']=opt.x.tolist();r['optimization_success']=bool(opt.success);save()
 rec['pool']=score('rmse',b.y.iloc[v],matrix@np.asarray(r['frozen_weights']));r['folds'].append(rec);print(json.dumps(rec),flush=True);save()
r['status']='completed';save()
