import sys,json,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task,validation_split,score,split_key
from automl_residual import CausalResidual
OUT=ROOT/'results/strongest-team/residual.json';LOCAL=Path('/private/tmp/alfa-strongest-team')
if OUT.exists():raise RuntimeError('Reuse results')
b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Регрессия/task.md');of,ov,_=validation_split(b);t=b.X['Year_Factor'].to_numpy();r={'status':'running','hypothesis':'ST5','hashes':b.hashes,'folds':[]}
def save():OUT.write_text(json.dumps(r,indent=2))
for j,period in enumerate([4,3]):
 f=np.flatnonzero(t<period);v=np.flatnonzero(t==period)
 with np.load(f'/private/tmp/alfa-score-recovery-panel/candidate-3-fold-{j}.npz') as z:assert np.array_equal(z['positions'],v);base=z['prediction']
 save();start=time.monotonic();m=CausalResidual('Year_Factor',8).fit(b.X.iloc[f],b.y.iloc[f]);p=m.predict(b.X.iloc[v],base)
 rec={'key':split_key(f,v),'seconds':time.monotonic()-start,'audit':m.audit_,'grid':[]}
 for w in [0,.25,.5,.75,1]:rec['grid'].append({'weight':w,'score':score('rmse',b.y.iloc[v],(1-w)*base+w*p)})
 np.savez_compressed(LOCAL/f'residual-{j}.npz',positions=v,prediction=p,base=base);r['folds'].append(rec);save()
grid=[{'weight':item['weight'],'score':.65*item['score']+.35*r['folds'][1]['grid'][i]['score']} for i,item in enumerate(r['folds'][0]['grid'])];r['grid']=grid
w=min(grid,key=lambda x:x['score']);r['frozen_weight']=w['weight'];save();print('INNER',grid,flush=True)
if w['weight']>0:
 with np.load('/private/tmp/alfa-score-recovery-panel/selected-outer.npz') as z:assert np.array_equal(z['positions'],ov);base=z['prediction']
 start=time.monotonic();m=CausalResidual('Year_Factor',8).fit(b.X.iloc[of],b.y.iloc[of]);p=m.predict(b.X.iloc[ov],base);pred=(1-w['weight'])*base+w['weight']*p
 r['outer']={'score':score('rmse',b.y.iloc[ov],pred),'base_score':score('rmse',b.y.iloc[ov],base),'key':split_key(of,ov),'seconds':time.monotonic()-start,'audit':m.audit_,'adaptively_viewed':True}
 np.savez_compressed(LOCAL/'residual-selected-outer.npz',positions=ov,prediction=pred);print('OUTER',r['outer'],flush=True)
r['status']='completed';r['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;save()
