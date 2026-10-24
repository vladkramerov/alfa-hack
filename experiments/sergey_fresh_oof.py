"""SPEC011: cache the exact measured leader once per frozen protocol."""
import argparse,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np
from automl_data import load_task,validation_split,split_key,score
ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'results/2026-10-24/sergey-improvement-v1'
def save(path,value):
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def main(task,stage):
 assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='sergey_branch'
 out=ROOT/'results/2026-10-24/HYP-075-01-multiclass-fresh-oof';out.mkdir(parents=True,exist_ok=False)
 ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
 started=time.monotonic();result={}
 try:
  b=load_task(ROOT/'data'/task/'task.md')
  if stage=='primary':fit,valid,description=validation_split(b)
  elif task=='regression':
   fit=np.flatnonzero(b.X.Year_Factor.to_numpy()<4);valid=np.flatnonzero(b.X.Year_Factor.to_numpy()==4);description='temporal:Year_Factor<4->4'
  else:
   from sklearn.model_selection import StratifiedGroupKFold
   import pandas as pd
   groups=pd.util.hash_pandas_object(b.X,index=False).to_numpy()
   fit,valid=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(b.X,b.y,groups))[1];description='stratified-group5:fold1:seed42'
  np.savez_compressed(out/'splits.npz',fit=fit,valid=valid)
  c=json.loads((ROOT/'baseline.json').read_text())[task]['candidate']
  c.pop('meta_state',None)
  evidence={'hypothesis':'HYP-075','user_authorized_reproduction':True,'task':task,'stage':stage,'candidate':c,
   'hashes':b.hashes,'split_key':split_key(fit,valid),'description':description,'fit_rows':len(fit),'valid_rows':len(valid),
   'ledger_sha256_before_fit':hashlib.sha256(ledger.encode()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
   'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('automl_*.py')},'python':sys.version,'platform':platform.platform()}
  save(out/'evidence.json',evidence)
  (out/'report.md').write_text('# HYP-075 fresh OOF '+task+'/'+stage+'\n\nStarted before fit. Reference: solution_alternative_v2.zip. Saved outputs absent from Git; direct request authorizes reproduction.\n')
  os.environ.update(COMBO_RUN_DIR=str(SUITE/'control-fit-log'),COMBO_LEDGER_PATH=str(ROOT/'HYPOTHESES.md'),COMBO_HYPOTHESIS='HYP-075-'+task+'-'+stage)
  from automl_models import build_pipeline
  model=build_pipeline(b.task_type,c,threads=4).fit(b.X.iloc[fit],b.y.iloc[fit])
  pred=model.predict(b.X.iloc[valid]);data={'positions':valid,'y':np.asarray(b.y.iloc[valid]),'prediction':pred}
  if task=='multiclass':
   data={k:v.astype(str) if k in ['y','prediction'] else v for k,v in data.items()}
   data.update(classes=np.asarray(model.classes_,dtype=str),probabilities=model.predict_proba(b.X.iloc[valid]))
  else:
   data['base_prediction']=model.base_.predict(b.X.iloc[valid])
  np.savez_compressed(out/'validation_predictions.npz',**data)
  joblib.dump(model,out/'model.joblib',compress=3)
  save(out/'audit.json',model.audit_)
  result={'status':'completed','metric':b.metric,'score':score(b.metric,b.y.iloc[valid],pred),'seconds':time.monotonic()-started,
   'fit_rows':len(fit),'valid_rows':len(valid),'description':description,'frozen_public_meta_reused':bool(c.get('meta_state'))}
 except Exception as e:
  result={'status':'failed','error':type(e).__name__+': '+str(e),'seconds':time.monotonic()-started}
  (out/'traceback.txt').write_text(traceback.format_exc())
 save(out/'metrics.json',result)
 with (out/'report.md').open('a') as f:f.write('\nResult:\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
 with (ROOT/'HYPOTHESES.md').open('a') as f:f.write('\n- HYP-075 primary: '+str(result)+'; [report]('+out.relative_to(ROOT).as_posix()+'/report.md). Не повторять.\n')
 with (ROOT/'docs/hypotheses/HYP-075.md').open('a') as f:f.write('\nРезультат: '+str(result)+'\n')
 with (ROOT/'results/README.md').open('a') as f:f.write('\n| SPEC011 control | '+task+'/'+stage+' | '+str(result)+' | [report](../'+out.relative_to(ROOT).as_posix()+'/report.md) |\n')
 print(json.dumps({'task':task,'stage':stage,**result}),flush=True)
 if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['regression','multiclass'],required=True);p.add_argument('--stage',choices=['primary','secondary'],required=True);a=p.parse_args();main(a.task,a.stage)
