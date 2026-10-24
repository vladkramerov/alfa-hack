import sys,os,time,json,hashlib,subprocess,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(R)]
import numpy as np,joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error
from automl_data import load_task
from automl_panel_change import PanelChangeForecaster
S=R/'results/2026-10-24/sergey-scientific-v2'
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str))
assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='sergey_branch'
b=load_task(R/'data/regression/task.md')
for hid in [83,84]:
 ledger=(R/'HYPOTHESES.md').read_text()
 for num,stage in enumerate(['primary','secondary'],1):
  out=R/f'results/2026-10-24/HYP-{hid:03d}-{num:02d}-panel-change-{stage}';out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
  old=R/'results/2026-10-24/sergey-improvement-v1'/f'control-regression-{stage}'
  with np.load(old/'splits.npz') as z:fi=z['fit'];va=z['valid']
  h73=np.load(R/f'results/2026-10-24/HYP-073-{num:02d}-regression-panel-{stage}/validation_predictions.npz');control=h73['prediction'];actual=b.y.iloc[va].to_numpy(float)
  save(out/'evidence.json',{'hypothesis':hid,'data_hashes':b.hashes,'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),W/'automl_panel_change.py']},'control_predictions_sha256':hashlib.sha256((R/f'results/2026-10-24/HYP-073-{num:02d}-regression-panel-{stage}/validation_predictions.npz').read_bytes()).hexdigest(),'outer_targets_used_for_fit':False,'fit_rows':len(fi),'valid_rows':len(va)})
  np.savez_compressed(out/'splits.npz',fit=fi,valid=va)
  (out/'report.md').write_text(f'# HYP-{hid:03d} {stage}\n\nSPEC012, preregistered causal first-difference features; H43 fallback cached, H73 control predictions reused.\n')
  try:
   if hid==83:
    base=joblib.load(old/'model.joblib').base_;fallback=base.predict(b.X.iloc[va])
    os.environ.update(COMBO_RUN_DIR=str(S/'panel-change-fit-journal'),COMBO_LEDGER_PATH=str(R/'HYPOTHESES.md'),COMBO_HYPOTHESIS=f'HYP083-{stage}')
    model=PanelChangeForecaster(4).fit(b.X.iloc[fi],b.y.iloc[fi]);pred=model.predict_with_fallback(b.X.iloc[va],fallback);save(out/'audit.json',model.audit_);joblib.dump(model,out/'model.joblib',compress=3)
   else:
    new=np.load(R/f'results/2026-10-24/HYP-083-{num:02d}-panel-change-{stage}/validation_predictions.npz')['prediction'];pred=.5*control+.5*new
   c=float(np.sqrt(mean_squared_error(actual,control)));score=float(np.sqrt(mean_squared_error(actual,pred)))
   np.savez_compressed(out/'validation_predictions.npz',y=actual,prediction=pred,control=control,positions=va)
   result={'status':'completed','metric':'rmse','score':score,'control':c,'relative_gain':1-score/c,'seconds':time.monotonic()-started,'new_tree_fits':1 if hid==83 else 0,'stage':stage,'mae':float(mean_absolute_error(actual,pred))}
  except Exception as e:
   result={'status':'failed','error':repr(e),'stage':stage,'seconds':time.monotonic()-started};(out/'traceback.txt').write_text(traceback.format_exc())
  save(out/'metrics.json',result)
  with (out/'report.md').open('a') as f:f.write('\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
  with (R/'HYPOTHESES.md').open('a') as f:f.write(f'\n- SPEC012 **HYP-{hid:03d}/{stage}**: '+json.dumps(result)+'; [report]('+str(out.relative_to(R))+'/report.md). Не повторять.\n')
  with (R/'results/README.md').open('a') as f:f.write(f'\n| H{hid} | {stage} | '+str(result.get('score',result['status']))+' | [report](../'+str(out.relative_to(R))+'/report.md) |\n')
  print(hid,stage,json.dumps(result),flush=True)
  if result['status']!='completed':sys.exit(1)
