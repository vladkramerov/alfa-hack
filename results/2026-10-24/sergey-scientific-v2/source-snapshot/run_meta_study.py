import sys,os,json,time,hashlib,subprocess,traceback
from pathlib import Path
W=Path(__file__).resolve().parent;R=Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack');sys.path[:0]=[str(W/'meta'),str(R)]
import numpy as np,joblib
import torch
torch.set_num_threads(4)
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
from automl_meta_research import fit_normalized_state,predict_state,crossfit_meta,fit_affine_rule,predict_affine,fit_mcc_state
S=R/'results/2026-10-24/sergey-scientific-v2';ORDER=['raw','smote','frequency'];CONTROL={'primary':.64472256711304,'secondary':.6075743572072007}
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))
def metric(y,p,classes):return float(f1_score(y,classes[p.argmax(1)],labels=classes,average='macro',zero_division=0))
assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='sergey_branch'
for hid in [76,77,78]:
 ledger=(R/'HYPOTHESES.md').read_text();assert f'HYP-{hid:03d}' in ledger
 hypothesis_start=time.monotonic()
 for number,stage in enumerate(['primary','secondary'],1):
  while not (S/f'oof-cache-{stage}/metrics.json').exists():time.sleep(1)
  cache=S/f'oof-cache-{stage}';assert json.loads((cache/'metrics.json').read_text())['status']=='completed'
  d=np.load(cache/'calibration.npz');y=d['y'];oy=d['outer_y'];classes=d['classes'];parts=[d['oof_'+c] for c in ORDER];outer=[d['outer_'+c] for c in ORDER]
  folds=[(np.asarray(f['fit']),np.asarray(f['valid'])) for f in json.loads((cache/'folds.json').read_text())]
  out=R/f'results/2026-10-24/HYP-{hid:03d}-{number:02d}-meta-{stage}';out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
  save(out/'evidence.json',{'hypothesis':hid,'stage':stage,'control':CONTROL[stage],'cache_sha256':hashlib.sha256((cache/'calibration.npz').read_bytes()).hexdigest(),'cache_evidence':json.loads((cache/'evidence.json').read_text()),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),W/'meta/automl_meta_research.py']},'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'fit_rows':len(y),'valid_rows':len(oy),'outer_labels_used_for_fit':False,'max_hypothesis_seconds':900})
  (out/'report.md').write_text(f'# HYP-{hid:03d} / {stage}\n\n[SPEC012](../../../docs/specs/SPEC-012-scientific-research.md). Calibration only current outer-fit, grouped meta-crossfit conditional on base-OOF; not fully nested base-model evaluation. Full branches reused, no new tree fits. All parameters preregistered.\n')
  try:
   if hid==76:
    state=fit_normalized_state(parts,y,classes);save(out/'state.json',state)
    p=predict_state(outer,state,classes)
    # These are calibration inputs for H77/H78, not extra outer experiments.
    cf,audit=crossfit_meta(parts,y,classes,folds,'normalized');save(out/'crossfit-audit.json',audit)
    np.savez_compressed(out/'meta-crossfit.npz',probabilities=cf)
    detail={'normalized_meta_crossfit_score':metric(y,cf,classes),'meta_fit_count':4,'converged':state['audit']['converged']}
   elif hid==77:
    norm=R/f'results/2026-10-24/HYP-076-{number:02d}-meta-{stage}';state=json.loads((norm/'state.json').read_text());normcf=np.load(norm/'meta-crossfit.npz')['probabilities']
    legacycf,legacy_audit=crossfit_meta(parts,y,classes,folds,'legacy');save(out/'legacy-crossfit-audit.json',legacy_audit)
    np.savez_compressed(out/'legacy-crossfit.npz',probabilities=legacycf)
    legacy_score=metric(y,legacycf,classes);norm_score=metric(y,normcf,classes)
    anchor='normalized' if norm_score>legacy_score else 'legacy'
    if anchor=='legacy':
     if stage=='primary':state=json.loads((R/'results/2026-10-24/HYP-075-02-multiclass-clean-runtime-continuation/metrics.json').read_text())['learned_state']['meta_state']
     else:state=joblib.load(R/'results/2026-10-24/sergey-improvement-v1/control-multiclass-secondary/model.joblib').learned_state_['meta_state']
    cf=normcf if anchor=='normalized' else legacycf;rule=fit_affine_rule(y,cf,classes)
    save(out/'state.json',state);save(out/'rule.json',rule);p=predict_affine(predict_state(outer,state,classes),rule,classes)
    detail={'meta_fit_count':3,'anchor_selected_inside_train':anchor,'legacy_meta_crossfit_score':legacy_score,'normalized_meta_crossfit_score':norm_score,'affine_training_score':rule['audit']['macro_f1'],'chosen_iteration':rule['chosen_iteration']}
   else:
    norm=R/f'results/2026-10-24/HYP-076-{number:02d}-meta-{stage}';initial=json.loads((norm/'state.json').read_text());naudit=json.loads((norm/'crossfit-audit.json').read_text());normcf=np.load(norm/'meta-crossfit.npz')['probabilities']
    cf,audit=crossfit_meta(parts,y,classes,folds,'mcc',initial_states=naudit['states']);save(out/'crossfit-audit.json',audit);np.savez_compressed(out/'meta-crossfit.npz',probabilities=cf)
    state=fit_mcc_state(parts,y,classes,initial);save(out/'state.json',state);raw=predict_state(outer,state,classes)
    mcc_score=metric(y,cf,classes);norm_score=metric(y,normcf,classes);selected=mcc_score>norm_score
    p=raw if selected else predict_state(outer,initial,classes)
    detail={'meta_fit_count':4,'mcc_crossfit_score':mcc_score,'normalized_meta_crossfit_score':norm_score,'mcc_selected_inside_train':selected,'standalone_mcc_outer_score':metric(oy,raw,classes)}
    np.savez_compressed(out/'standalone-mcc-predictions.npz',y=oy,classes=classes,probabilities=raw,prediction=classes[raw.argmax(1)],positions=d['valid_positions'])
   pred=classes[p.argmax(1)];score=metric(oy,p,classes)
   np.savez_compressed(out/'validation_predictions.npz',y=oy,classes=classes,probabilities=p,prediction=pred,positions=d['valid_positions'])
   save(out/'per-class.json',classification_report(oy,pred,labels=classes,output_dict=True,zero_division=0));save(out/'confusion.json',{'classes':classes.tolist(),'matrix':confusion_matrix(oy,pred,labels=classes).tolist()})
   result={'status':'completed','metric':'macro_f1','score':score,'control':CONTROL[stage],'relative_gain':score/CONTROL[stage]-1,'accuracy':float(accuracy_score(oy,pred)),'seconds':time.monotonic()-started,'stage':stage,'base_tree_fits':0,**detail}
  except Exception as e:
   result={'status':'failed','stage':stage,'error':repr(e),'seconds':time.monotonic()-started};(out/'traceback.txt').write_text(traceback.format_exc())
  save(out/'metrics.json',result)
  with (out/'report.md').open('a') as f:f.write('\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
  with (R/'HYPOTHESES.md').open('a') as f:f.write(f'\n- SPEC012 **HYP-{hid:03d}/{stage}**: '+json.dumps(result,ensure_ascii=False)+'; [отчёт]('+str(out.relative_to(R))+'/report.md). Не повторять.\n')
  with (R/'results/README.md').open('a') as f:f.write(f'\n| H{hid} | {stage} | '+str(result.get('score',result['status']))+' | [report](../'+str(out.relative_to(R))+'/report.md) |\n')
  print(hid,stage,json.dumps(result),flush=True)
  if result['status']!='completed':sys.exit(1)
  if time.monotonic()-hypothesis_start>900:raise TimeoutError(f'H{hid}900sec exceeded')
