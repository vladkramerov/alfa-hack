"""Native boosting with train-only stopping and fixed probability-pooling ablation."""
import json,sys,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task, validation_split,split_key,score
from automl_engine import _search_windows
from automl_models import build_pipeline,canonical_candidate
from automl_worker import aligned_probabilities,train_priors,class_decisions,_cache_classes,_training_key
out=ROOT/'results/score-recovery/classification.json'
if out.exists(): raise RuntimeError('Reuse result')
b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Многоклассовая классификация/task.md');of,ov,_=validation_split(b);w=_search_windows(b,of,1800)[0][0];f,v=w['fit'],w['valid']
old=ROOT/'results/public-v5/multiclass-adaptive';meta=json.loads((old/'validation.json').read_text());assert split_key(f,v)==meta['search_key']
control=json.loads((old/'frozen_selection.json').read_text())['candidates'][-1]
cache=next(Path('/private/tmp/alfa-work-v5/multiclass-adaptive/agent_runs').glob('*/experiment-09-fold01/probability_cache.npz'))
with np.load(cache,allow_pickle=False) as z:
 print('cache_fields',list(z.files),flush=True)
 base=z['valid_probabilities'];cached_classes=z['class_order'];assert np.array_equal(z['fit_positions'],f) and np.array_equal(z['valid_positions'],v);assert str(z['training_key'])==_training_key(control)
classes=_cache_classes(b.classes)
assert np.array_equal(cached_classes,classes)
priors=train_priors(b.y.iloc[f],b.classes)
assert abs(score('macro_f1',b.y.iloc[v],class_decisions(base,b.classes,priors,.3))-.6367083180924042)<1e-10
recipes=[{'family':'catboost','catboost_encoding':'native','early_stopping':True,'parameters':{'iterations':1200,'depth':8},'class_weight_power':.35},
         {'family':'lightgbm','early_stopping':True,'parameters':{'n_estimators':1200,'num_leaves':63,'min_child_samples':10,'reg_lambda':3,'subsample':.8,'colsample_bytree':.85},'class_weight_power':.2}]
r={'control_inner':.6367083180924042,'control_outer':.6373379463394646,'data_hashes':b.hashes,'search_key':split_key(f,v),'recipes':[],'fit_seconds':0,'ensembles':[]}
local=Path('/private/tmp/alfa-score-recovery-class');local.mkdir(exist_ok=False)
def save():out.write_text(json.dumps(r,indent=2))
save()
for i,c in enumerate(recipes):
 start=time.monotonic();m=build_pipeline('classification',c,8).fit(b.X.iloc[f],b.y.iloc[f]);seconds=time.monotonic()-start;r['fit_seconds']+=seconds
 p=aligned_probabilities(m,b.X.iloc[v],b.classes);pred=class_decisions(p,b.classes,priors,c.get('decision_power',0));value=score('macro_f1',b.y.iloc[v],pred)
 rec={'candidate':canonical_candidate(c),'score':value,'fit_seconds':seconds,'stopping':m.audit_.get('early_stopping')};r['recipes'].append(rec)
 np.savez_compressed(local/f'candidate-{i}.npz',positions=v,classes=np.asarray(b.classes),probabilities=p)
 for mode in ['arithmetic','geometric']:
  pooled=(base+p)/2 if mode=='arithmetic' else np.sqrt(np.maximum(base,1e-9)*np.maximum(p,1e-9));pooled/=pooled.sum(axis=1,keepdims=True)
  value=score('macro_f1',b.y.iloc[v],class_decisions(pooled,b.classes,priors,.3))
  r['ensembles'].append({'score':value,'candidate':canonical_candidate({'family':'blend','blend_mode':mode,'members':[control,c],'weights':[.5,.5],'decision_power':.3})})
 save();print(json.dumps(rec),flush=True)
all_rec=r['recipes']+r['ensembles'];winner=max(all_rec,key=lambda x:x['score']);r['frozen_selection']=winner['candidate'] if winner['score']>r['control_inner'] else None;save()
if r['frozen_selection']:
 start=time.monotonic();m=build_pipeline('classification',winner['candidate'],8).fit(b.X.iloc[of],b.y.iloc[of]);seconds=time.monotonic()-start;r['fit_seconds']+=seconds
 p=aligned_probabilities(m,b.X.iloc[ov],b.classes);pred=class_decisions(p,b.classes,train_priors(b.y.iloc[of],b.classes),winner['candidate'].get('decision_power',0));r['selected_outer']=score('macro_f1',b.y.iloc[ov],pred);r['outer_key']=split_key(of,ov)
 from sklearn.metrics import classification_report
 r['per_class']=classification_report(b.y.iloc[ov],pred,output_dict=True,zero_division=0)
 np.savez_compressed(local/'selected-outer.npz',positions=ov,prediction=pred,probabilities=p,classes=np.asarray(b.classes))
else:r['selected_outer']=r['control_outer']
r['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;r['predictions_local']=str(local);r['status']='completed';save();print(json.dumps({'selected_outer':r['selected_outer'],'fit_seconds':r['fit_seconds'],'ensembles':[(x['candidate'].get('blend_mode','arithmetic'),x['score']) for x in r['ensembles']]}),flush=True)
