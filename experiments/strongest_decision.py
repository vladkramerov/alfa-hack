import sys,json,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task,validation_split,score,split_key
from automl_models import build_pipeline
from automl_decision import PosteriorDecision
from automl_engine import _search_windows
from automl_worker import class_decisions,train_priors,aligned_probabilities
OUT=ROOT/'results/strongest-team/posterior.json';LOCAL=Path('/private/tmp/alfa-strongest-team')
if OUT.exists():raise RuntimeError('Reuse result')
b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Многоклассовая классификация/task.md');of,ov,_=validation_split(b);w=_search_windows(b,of,1800)[0][0];f,v=w['fit'],w['valid']
c=json.loads((ROOT/'results/public-v5/multiclass-adaptive/frozen_selection.json').read_text())['candidates'][-1]
r={'status':'running','hypothesis':'ST6','candidate':c,'hashes':b.hashes,'teacher_in_sample':True,'folds':[]}
def save():OUT.write_text(json.dumps(r,indent=2))
for stage,f,v in [('inner',f,v),('outer',of,ov)]:
 if stage=='outer' and r['frozen_mode']=='control':break
 save();start=time.monotonic();m=build_pipeline('classification',c,8).fit(b.X.iloc[f],b.y.iloc[f]);seconds=time.monotonic()-start
 p=aligned_probabilities(m,b.X.iloc[v],b.classes);train=aligned_probabilities(m,b.X.iloc[f],b.classes)
 rec={'stage':stage,'key':split_key(f,v),'teacher_seconds':seconds,'base':score('macro_f1',b.y.iloc[v],class_decisions(p,b.classes,train_priors(b.y.iloc[f],b.classes),.3))}
 if stage=='inner':
  cache=next(Path('/private/tmp/alfa-work-v5/multiclass-adaptive/agent_runs').glob('*/experiment-09-fold01/probability_cache.npz'))
  with np.load(cache,allow_pickle=False) as z:rec['cache_probability_max_difference']=float(np.max(np.abs(p-z['valid_probabilities'])))
 start=time.monotonic();decision=PosteriorDecision().fit(train,np.searchsorted(np.asarray(b.classes),b.y.iloc[f].to_numpy()));rec['calibration_seconds']=time.monotonic()-start
 rec['audit']=decision.audit_;rec['ce_only']=score('macro_f1',b.y.iloc[v],np.asarray(b.classes)[decision.calibrate(p).argmax(axis=1)])
 rec['expected_f1']=score('macro_f1',b.y.iloc[v],np.asarray(b.classes)[decision.predict_proba(p).argmax(axis=1)])
 np.savez_compressed(LOCAL/f'posterior-{stage}.npz',positions=v,probabilities=p,expected=decision.predict_proba(p),ce=decision.calibrate(p),classes=np.asarray(b.classes))
 r['folds'].append(rec)
 if stage=='inner':r['frozen_mode']=max(['control','ce_only','expected_f1'],key=lambda k:rec['base' if k=='control' else k])
 save();print(json.dumps(rec),flush=True)
r['status']='completed';r['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;save()
