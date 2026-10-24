"""One disclosed technical reconstruction of missing calibration arrays; no outer fit."""
import sys,os,json,time,hashlib,subprocess,traceback
from pathlib import Path
R=Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack');sys.path.insert(0,str(R))
import numpy as np,joblib
from automl_data import load_task,split_key
from automl_advanced import calibration_folds
from automl_class_combo import ClassBranch,COMPONENT_ORDER,aligned_probabilities
S=R/'results/2026-10-24/sergey-scientific-v2'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))
assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='sergey_branch'
b=load_task(R/'data/multiclass/task.md')
for stage in ['primary','secondary']:
 ledger=(R/'HYPOTHESES.md').read_text();out=S/('oof-cache-'+stage);out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();control=R/'results/2026-10-24/sergey-improvement-v1'/('control-multiclass-'+stage)
 with np.load(control/'splits.npz') as f: fit=f['fit'];valid=f['valid']
 X=b.X.iloc[fit].reset_index(drop=True);y=np.asarray(b.y.iloc[fit],dtype=str);classes=np.unique(y)
 folds,reason=calibration_folds(X,y);assert len(folds)==3,reason
 save(out/'evidence.json',{'purpose':'Required calibration inputs for NEW H76-H78, NOT a baseline metric experiment','data_hashes':b.hashes,'split_key':split_key(fit,valid),'stage':stage,'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'reconstruction_disclosure':'9 component fits duplicate historical OOF configurations because raw arrays were never saved. Full outer-fit branches ARE reused, zero full branch/baseline fits. Outputs cached for all new meta hypotheses.','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()})
 (out/'report.md').write_text('# Техническое восстановление отсутствующего OOF cache\n\nЭто не новая гипотеза baseline. 9 необходимых component fits, повтор исторических OOF конфигураций раскрыт заранее в SPEC012; все full branches используются из сохранённого model.joblib. Новые calibration arrays сохраняются для всех дальнейших гипотез.\n')
 os.environ.update(COMBO_RUN_DIR=str(out/'fit-journal'),COMBO_LEDGER_PATH=str(R/'HYPOTHESES.md'),COMBO_HYPOTHESIS='SPEC012-required-OOF-inputs-'+stage)
 oof={c:np.zeros((len(X),len(classes))) for c in COMPONENT_ORDER}
 try:
  for number,(fi,va) in enumerate(folds):
   np.savez_compressed(out/f'fold-{number}-split.npz',fit=fi,valid=va)
   for c in COMPONENT_ORDER:
    branch=ClassBranch(c,threads=4).fit(X.iloc[fi],y[fi]);p=aligned_probabilities(branch.predict_proba(X.iloc[va]),branch.classes_,classes)
    oof[c][va]=p;np.savez_compressed(out/f'fold-{number}-{c}.npz',positions=va,probabilities=p)
    joblib.dump(branch,out/f'fold-{number}-{c}.joblib',compress=3)
    save(out/f'fold-{number}-{c}-audit.json',branch.audit_)
    print(stage,'fold',number,c,round(time.monotonic()-started,2),flush=True)
  full=joblib.load(control/'model.joblib')
  outer={c:aligned_probabilities(full.branches_[c].predict_proba(b.X.iloc[valid]),full.branches_[c].classes_,classes) for c in COMPONENT_ORDER}
  np.savez_compressed(out/'calibration.npz',y=y,outer_y=np.asarray(b.y.iloc[valid],dtype=str),classes=classes,fit_positions=fit,valid_positions=valid,**{'oof_'+c:oof[c] for c in COMPONENT_ORDER},**{'outer_'+c:outer[c] for c in COMPONENT_ORDER})
  save(out/'folds.json',[{'fit':fi.tolist(),'valid':va.tolist()} for fi,va in folds])
  result={'status':'completed','new_required_component_fits':9,'outer_full_branch_fits':0,'seconds':time.monotonic()-started,'cache_sha256':hashlib.sha256((out/'calibration.npz').read_bytes()).hexdigest()}
 except Exception as e:
  result={'status':'failed','error':repr(e),'seconds':time.monotonic()-started};(out/'traceback.txt').write_text(traceback.format_exc())
 save(out/'metrics.json',result)
 with (out/'report.md').open('a') as f:f.write('\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
 print(stage,result,flush=True)
 if result['status']!='completed':sys.exit(1)
