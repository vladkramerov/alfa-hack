"""Paired one-SE policy replay on frozen source trial sequence, no model fitting."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task
from automl_panel import CausalPanelFeatures
OUT=ROOT/'results/strongest-team/selection-replay.json'
if OUT.exists():raise RuntimeError('Reuse result')
run=next(Path('/private/tmp/alfa-strongest-e2e-v2/regression-shadow/agent_runs').glob('*'))
b=load_task(run.parent.parent/'task.md');v=json.load(open(run/'validation.json'));windows=[]
for p in sorted(run.glob('search_split-*.npz')):
 with np.load(p) as z:windows.append({'fit':z['fit'],'valid':z['valid']})
weights=[.65,.35] if len(windows)==2 else [1]
key=CausalPanelFeatures('Year_Factor').fit(b.X.iloc[windows[0]['fit']],np.zeros(len(windows[0]['fit'])))
records=json.load(open(run/'experiments.json'));accepted=None;history=[]
for record in records:
 if record['status']!='completed':continue
 accept=accepted is None;entry={'id':record['id'],'score':record['score']}
 if accepted is not None and record['score']<accepted['score']:
  groups=[];influences=[]
  for i,w in enumerate(windows):
   def read(r):
    d=r['fold_results'][i]['directory'];p=run/d/'validation_predictions.npz'
    with np.load(p) as z:assert np.array_equal(z['positions'],w['valid']);return z['prediction']
   a,c=read(record),read(accepted);y=b.y.iloc[w['valid']].to_numpy();ra=np.sqrt(np.mean((a-y)**2));rc=np.sqrt(np.mean((c-y)**2))
   influence=weights[i]*((a-y)**2-(c-y)**2)/max(1e-12,ra+rc)/len(y)
   influences.extend(influence);groups.extend(key._keys(b.X.iloc[w['valid']],key.columns_) if key.columns_ else pd.util.hash_pandas_object(b.X.iloc[w['valid']],index=False).to_numpy())
  part=pd.DataFrame({'g':groups,'d':influences}).groupby('g').d.sum().to_numpy();delta=float(np.sum(influences));se=float(np.sqrt(np.sum((part-part.mean())**2)*len(part)/max(1,len(part)-1)))
  entry.update(delta=delta,cluster_se=se,incumbent=accepted['id']);accept=delta < -se
 if accept:accepted=record
 entry['accepted']=accept;history.append(entry)
r={'status':'completed','rule':'paired one-SE, threshold1 fixed before replay','outer_read':False,'trials':history,'selected_id':accepted['id'],'selected_candidate':accepted['candidate'],'source_run':str(run),'group_columns':key.columns_}
OUT.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
