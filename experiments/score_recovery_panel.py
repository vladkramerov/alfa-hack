"""Frozen generic entity-history ablation; reuse public-v5 controls without fits."""
import json, sys, time, resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from automl_engine import _search_windows
from automl_models import build_pipeline, canonical_candidate

out=ROOT/'results/score-recovery/panel.json'
if out.exists(): raise RuntimeError('Reuse existing result')
b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Регрессия/task.md')
f,v,desc=validation_split(b); windows,_=_search_windows(b,f,1800)
old=json.loads((ROOT/'results/public-v5/regression-adaptive/validation.json').read_text())
assert all(not np.intersect1d(np.r_[w['fit'],w['valid']],v).size for w in windows)
control=json.loads((ROOT/'results/public-v5/regression-adaptive/frozen_selection.json').read_text())['candidates'][-1]
recipes=[control, {'family':'extra_trees','target_transform':'log1p'}, {'family':'lightgbm','parameters':{'n_estimators':500,'num_leaves':31,'reg_lambda':2},'panel_time_column':'Year_Factor'},
         {'family':'extra_trees','target_transform':'log1p','panel_time_column':'Year_Factor'}]
r={'metric':'rmse','data_hashes':b.hashes,'control_inner':None,'control_outer':43.70058963959562,'team_best_outer':34.520268654585415,'recipes':[],'fit_seconds':0}
local=Path('/private/tmp/alfa-score-recovery-panel');local.mkdir(exist_ok=False)
def save(): out.write_text(json.dumps(r,indent=2))
save()
for i,c in enumerate(recipes):
 rec={'candidate':canonical_candidate(c),'folds':[]};r['recipes'].append(rec)
 for j,w in enumerate(windows):
  start=time.monotonic();m=build_pipeline('regression',c,8).fit(b.X.iloc[w['fit']],b.y.iloc[w['fit']]);fit_seconds=time.monotonic()-start;r['fit_seconds']+=fit_seconds
  pred=m.predict(b.X.iloc[w['valid']]);value=score('rmse',b.y.iloc[w['valid']],pred)
  np.savez_compressed(local/f'candidate-{i}-fold-{j}.npz',positions=w['valid'],prediction=pred)
  rec['folds'].append({'score':value,'weight':w['weight'],'key':split_key(w['fit'],w['valid']),'fit_seconds':fit_seconds,'panel':m.audit_.get('causal_panel')});save()
 rec['score']=sum(x['score']*x['weight'] for x in rec['folds']);
 if i==0: r['control_inner']=rec['score']
 save();print(json.dumps(rec),flush=True)
winner=min(r['recipes'],key=lambda x:x['score']);r['frozen_selection']=winner['candidate'] if winner['score']<r['control_inner'] else None;save()
if r['frozen_selection']:
 start=time.monotonic();m=build_pipeline('regression',winner['candidate'],8).fit(b.X.iloc[f],b.y.iloc[f]);seconds=time.monotonic()-start;r['fit_seconds']+=seconds
 pred=m.predict(b.X.iloc[v]);r['selected_outer']=score('rmse',b.y.iloc[v],pred);r['outer_key']=split_key(f,v);r['outer_panel']=m.audit_.get('causal_panel');np.savez_compressed(local/'selected-outer.npz',positions=v,prediction=pred)
r['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;r['predictions_local']=str(local);r['status']='completed';save();print(json.dumps({k:r[k] for k in ['fit_seconds','peak_rss_mb','selected_outer']}),flush=True)
