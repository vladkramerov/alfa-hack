import sys,os,time,json,hashlib,dataclasses,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
import numpy as np,joblib
import ag_finalize as F,ag_ml_metrics as M
from refinements import bounded_postprocess,refined_regression_ensemble,macro_aware_ensemble,BASE_GREEDY,BASE_POST
S=R/'results/another_solution/2026-10-24'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))
for hid in ['AS001','AS002','AS003']:
 ledger=(R/'HYPOTHESES.md').read_text()
 for kind in ['regression','multiclass']:
  source=S/('original-'+kind)
  while not (source/'metrics.json').exists():
   if (source/'failure.json').exists():raise RuntimeError((source/'failure.json').read_text())
   time.sleep(1)
  out=S/(hid+'-'+kind);out.mkdir(exist_ok=False);started=time.monotonic();d=joblib.load(source/'replay.joblib');task=d['task'];M.configure(task);metric=M.METRICS[task.metric];y,A,B=d['y'],d['A'],d['B'];cfg=d['config']
  applicable=(hid=='AS002' and kind=='regression') or (hid in ['AS001','AS003'] and kind=='multiclass')
  save(out/'evidence.json',{'hypothesis':hid,'control_archive_sha256':'cbf5de846d8a4a4f760e9574a60d47c42c1adeb7b340fd655f5d296b41589547','data_and_source':json.loads((source/'evidence.json').read_text()),'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'candidate_source_sha256':hashlib.sha256(Path(__file__).with_name('refinements.py').read_bytes()).hexdigest(),'tree_fits':0,'applicable':applicable,'fit_weights_on_A_only':True,'selection_on_B_same_as_original':True})
  F.fit_postprocess=bounded_postprocess if hid=='AS001' else BASE_POST
  F.greedy_ensemble=refined_regression_ensemble if hid=='AS002' else macro_aware_ensemble if hid=='AS003' else BASE_GREEDY
  old=d['choice'].solution;single=d['single']
  if hid=='AS001':
   post=F.fit_postprocess(metric,y,single.oof,A,B);single=dataclasses.replace(single,post=post,cv=F._cv(metric,y,single.oof,post,A,B),holdout_b=M.score(metric,y[B],single.oof[B],post))
  ens=F.greedy_ensemble(metric,y,d['runs'],A,B,cfg.ensemble_max_models,step_gain=cfg.ensemble_step_gain)
  choice=F.choose(metric,y,single,ens,B,cfg.ensemble_min_gain,cfg.ensemble_confidence);sol=choice.solution
  sign=1 if metric.greater_is_better else -1
  result={'status':'completed' if applicable else 'not_applicable_unchanged','metric':metric.name,'oof_score':sol.cv,'holdout_B_score':sol.holdout_b,'control_oof_score':old.cv,'control_holdout_B_score':old.holdout_b,'relative_gain_B':sign*(sol.holdout_b-old.holdout_b)/abs(old.holdout_b),'relative_gain_OOF':sign*(sol.cv-old.cv)/abs(old.cv),'selected':sol.name,'members':sol.members,'post':sol.post,'choice_reason':choice.reason,'seconds':time.monotonic()-started,'new_tree_fits':0}
  if applicable:result['win_rate_vs_original_on_B']=F.paired_win_rate(metric,y,old,sol,B,n_boot=200)
  np.savez_compressed(out/'predictions.npz',y=y,oof=sol.oof,test_pred=sol.test_pred,A=A,B=B)
  save(out/'metrics.json',result);save(out/'post.json',sol.post)
  (out/'report.md').write_text('# '+hid+'/'+kind+'\n\nOnly cached original OOF/test predictions, no model refits. A for fitting, B for selection; exploratory local validation.\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
  with (R/'HYPOTHESES.md').open('a') as f:f.write('\n- '+hid+'/'+kind+': '+json.dumps(result,ensure_ascii=False)+'; [отчёт]('+str(out.relative_to(R))+'/report.md).\n')
  print(hid,kind,json.dumps(result,ensure_ascii=False),flush=True)
