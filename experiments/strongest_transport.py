"""ST4 role inference, frozen variants before any outer score."""
import sys,json,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task,validation_split,score,split_key
from automl_transfer import TemporalDonor
OUT=ROOT/'results/strongest-team/transport-roles.json';LOCAL=Path('/private/tmp/alfa-strongest-team')
if OUT.exists():raise RuntimeError('Reuse completed results')
b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Регрессия/task.md');of,ov,_=validation_split(b);t=b.X['Year_Factor'].to_numpy()
r={'status':'running','variants':[],'hypothesis':'ST4','data_hashes':b.hashes,'outer_adaptive':True}
def save():OUT.write_text(json.dumps(r,indent=2))
for mode in ['log','isotonic']:
 rec={'mode':mode,'folds':[]};r['variants'].append(rec)
 for j,period in enumerate([4,3]):
  f=np.flatnonzero(t<period);v=np.flatnonzero(t==period)
  with np.load(f'/private/tmp/alfa-score-recovery-panel/candidate-3-fold-{j}.npz') as z:assert np.array_equal(z['positions'],v);base=z['prediction']
  start=time.monotonic();m=TemporalDonor('Year_Factor',mode).fit(b.X.iloc[f],b.y.iloc[f]);pred=m.predict(b.X.iloc[v],base)
  rec['folds'].append({'key':split_key(f,v),'score':score('rmse',b.y.iloc[v],pred),'audit':m.audit_,'seconds':time.monotonic()-start})
  np.savez_compressed(LOCAL/f'roles-{mode}-{j}.npz',positions=v,prediction=pred);save()
 rec['score']=.65*rec['folds'][0]['score']+.35*rec['folds'][1]['score'];print(mode,rec['score'],flush=True);save()
w=min(r['variants'],key=lambda x:x['score']);r['frozen_mode']=w['mode'];save()
if w['score']<38.8220090577851:
 with np.load('/private/tmp/alfa-score-recovery-panel/selected-outer.npz') as z:assert np.array_equal(z['positions'],ov);base=z['prediction']
 m=TemporalDonor('Year_Factor',w['mode']).fit(b.X.iloc[of],b.y.iloc[of]);pred=m.predict(b.X.iloc[ov],base)
 r['outer']={'score':score('rmse',b.y.iloc[ov],pred),'key':split_key(of,ov),'audit':m.audit_}
 np.savez_compressed(LOCAL/'roles-selected-outer.npz',positions=ov,prediction=pred)
r['status']='completed';r['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;save();print(json.dumps(r.get('outer')),flush=True)
