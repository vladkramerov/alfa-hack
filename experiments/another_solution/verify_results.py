"""Read-only independent metrics and final-source parity, no model fits."""
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
import numpy as np,joblib
from sklearn.metrics import f1_score
import ag_finalize as F,ag_ml_metrics as M
S=R/'results/another_solution/2026-10-24'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))
checks=[];records=[]
def check(name,value):
 checks.append({'name':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
def independent(name,y,p,idx,post):
 if name=='rmse':
  values=np.asarray(p[idx],float)
  if post.get('clip'):
   lo,hi=post['clip'];values=np.clip(values,-np.inf if lo is None else lo,np.inf if hi is None else hi)
  if post.get('scale'):values*=post['scale']
  return float(np.sqrt(np.mean((y[idx]-values)**2)))
 return float(f1_score(y[idx],np.argmax(p[idx]*np.asarray(post.get('weights',np.ones(p.shape[1]))),axis=1),average='macro'))
for kind in ['regression','multiclass']:
 original=S/('original-'+kind);d=joblib.load(original/'replay.joblib');M.configure(d['task']);metric=M.METRICS[d['task'].metric];y,A,B=d['y'],d['A'],d['B'];cfg=d['config'];covered=np.concatenate([A,B])
 check(kind+'-A-B-disjoint',not len(np.intersect1d(A,B)))
 for name in ['original','AS001','AS002','AS003']:
  folder=original if name=='original' else S/(name+'-'+kind);m=json.loads((folder/'metrics.json').read_text());post=m['post'];path=folder/('final-predictions.npz' if name=='original' else 'predictions.npz')
  z=np.load(path);oof=z['oof'];check(kind+name+'-y',np.array_equal(z['y'],y));check(kind+name+'-finite',np.isfinite(oof).all() and np.isfinite(z['test_pred']).all())
  score=independent(metric.name,y,oof,covered,post);sb=independent(metric.name,y,oof,B,post)
  check(kind+name+'-OOF',np.isclose(score,m['oof_score'],rtol=1e-12));check(kind+name+'-B',np.isclose(sb,m['holdout_B_score'],rtol=1e-12))
  records.append({'task':kind,'case':name,'metric':metric.name,'oof':score,'B':sb,'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 single=d['single'];ens=F.greedy_ensemble(metric,y,d['runs'],A,B,cfg.ensemble_max_models,step_gain=cfg.ensemble_step_gain);choice=F.choose(metric,y,single,ens,B,cfg.ensemble_min_gain,cfg.ensemble_confidence);sol=choice.solution
 expected=S/('AS002-'+kind);z=np.load(expected/'predictions.npz');check(kind+'-integrated-OOF-parity',np.allclose(sol.oof,z['oof'],rtol=1e-12,atol=1e-12));check(kind+'-integrated-test-parity',np.allclose(sol.test_pred,z['test_pred'],rtol=1e-12,atol=1e-12))
 save(S/('final-'+kind+'-metrics.json'),{'metric':metric.name,'oof_score':sol.cv,'holdout_B_score':sol.holdout_b,'members':sol.members,'post':sol.post,'selection_reason':choice.reason,'model_refits':0,'source_sha256':hashlib.sha256((R/'ag_finalize.py').read_bytes()).hexdigest()})
 np.savez_compressed(S/('final-'+kind+'-predictions.npz'),y=y,oof=sol.oof,test_pred=sol.test_pred,A=A,B=B)
 if kind=='regression':
  rng=np.random.default_rng(0);old=d['choice'].solution;wins=0;deltas=[]
  for _ in range(200):
   idx=rng.choice(B,size=len(B),replace=True);a=independent('rmse',y,old.oof,idx,old.post);b=independent('rmse',y,sol.oof,idx,sol.post);wins+=b<a;deltas.append(1-b/a)
  save(S/'paired-bootstrap.json',{'seed':0,'draws':200,'win_rate':wins/200,'relative_gain_95_percentile_interval':np.quantile(deltas,[.025,.975]).tolist(),'unit':'paired OOF rows','caveat':'Same public OOF/B already used for validation selection; not an independent hidden score'})
result={'status':'passed','checks':len(checks),'failed':0,'details':checks,'records':records,'new_model_fits':0}
save(S/'independent-verification.json',result);print(json.dumps({k:result[k] for k in ['status','checks','failed','new_model_fits']}))
