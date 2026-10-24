"""Single first H50 evaluation on each fixed H75 protocol; 900s combined."""
import os,sys,json,time,hashlib,subprocess,traceback,shutil,resource,platform
from pathlib import Path
W=Path(__file__).resolve().parent;R=Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack')
sys.path[:0]=[str(W),str(R)]
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import numpy as np,pandas as pd,sklearn,xgboost,joblib
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
from automl_data import load_task,split_key
from focal_boosting import FocalBoostingClassifier

CONTROL={'primary':.64472256711304,'secondary':.6075743572072007}
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def guard():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='sergey_branch'

def main():
    guard();b=load_task(R/'data/multiclass/task.md');start=time.monotonic();deadline=start+900
    for number,stage in enumerate(['primary','secondary'],1):
        guard();ledger=(R/'HYPOTHESES.md').read_text();card=(R/'docs/hypotheses/HYP-050.md').read_text()
        assert 'HYP-050 first actual test' in ledger
        out=R/f'results/2026-10-24/HYP-050-{number:02d}-focal-{stage}'
        out.mkdir(parents=True,exist_ok=False);stage_start=time.monotonic()
        with np.load(R/f'results/2026-10-24/sergey-improvement-v1/control-multiclass-{stage}/splits.npz') as f:
            fit,valid=f['fit'],f['valid']
        np.savez_compressed(out/'splits.npz',fit=fit,valid=valid)
        sources=['focal_boosting.py','test_focal_boosting.py','run_focal.py']
        for name in sources:shutil.copyfile(W/name,out/name)
        save(out/'evidence.json',{'hypothesis':50,'stage':stage,'first_actual_H50_test':True,
            'data_hashes':b.hashes,'split_key':split_key(fit,valid),'control':CONTROL[stage],
            'source_sha256':{name:hashlib.sha256((W/name).read_bytes()).hexdigest() for name in sources},
            'Encoder_source_sha256':hashlib.sha256((R/'automl_models.py').read_bytes()).hexdigest(),
            'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'card_sha256':hashlib.sha256(card.encode()).hexdigest(),
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
            'fit_rows':len(fit),'valid_rows':len(valid),'outer_labels_used_for_fit':False,
            'configuration':{'estimators':600,'depth':6,'learning_rate':.05,'lambda':1,'tree_method':'hist',
                'device':'cpu','threads':4,'gamma':2,'alpha':'inverse sqrt fit counts, sample mean1',
                'gradient':'exact alpha*A*(p-y)','curvature':'positive diagonal surrogate, NOT exact Hessian',
                'preprocessing':'legacy raw median/OHE, no scaling','early_stopping':False,'calibration':False,'SMOTE':False},
            'total_budget_seconds':900,'command':'python work/spec012/focal/run_focal.py',
            'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                        'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'platform':platform.platform()}})
        (out/'report.md').write_text(f'# H50 / {stage}\n\nFirst actual run of previously proposed H50; '
            'registered before fit in SPEC012 and HYP050. Class-balanced gamma2 focal boosting on raw median/OHE. '
            'Exact gradient checked by finite differences; positive surrogate curvature is not an exact Hessian. '
            'Fixed600 trees, no early stopping, no outer tuning, calibration, or SMOTE. H78 was MCC meta learning and is different.\n')
        save(out/'metrics.json',{'status':'started','hypothesis':50,'stage':stage})
        save(out/'fit-event.json',{'phase':'single_fixed_focal_booster','status':'started','fit_rows':len(fit)})
        try:
            model=FocalBoostingClassifier().fit(b.X.iloc[fit],np.asarray(b.y.iloc[fit],dtype=str),deadline=deadline)
            save(out/'audit.json',model.audit_);joblib.dump(model,out/'model.joblib',compress=3)
            save(out/'fit-event.json',{'phase':'single_fixed_focal_booster','status':'completed','objective_calls':model.objective_.calls})
            p=model.predict_proba(b.X.iloc[valid]);classes=model.classes_;pred=classes[p.argmax(1)]
            y=np.asarray(b.y.iloc[valid],dtype=str)
            score=float(f1_score(y,pred,labels=classes,average='macro',zero_division=0))
            np.savez_compressed(out/'validation_predictions.npz',y=y,prediction=pred,probabilities=p,classes=classes,positions=valid)
            save(out/'per-class.json',classification_report(y,pred,labels=classes,output_dict=True,zero_division=0))
            save(out/'confusion.json',{'classes':classes.tolist(),'matrix':confusion_matrix(y,pred,labels=classes).tolist()})
            result={'status':'completed','metric':'macro_f1','score':score,'control':CONTROL[stage],
                'relative_gain':score/CONTROL[stage]-1,'stage':stage,'accuracy':float(accuracy_score(y,pred)),
                'model_fit_count':1,'objective_calls':model.objective_.calls,'seconds':time.monotonic()-stage_start,
                'total_seconds':time.monotonic()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'conclusion':'Подтверждена в проверенных условиях' if score>CONTROL[stage] else 'Не подтверждена по заданному критерию'}
        except Exception as error:
            result={'status':'interrupted' if isinstance(error,TimeoutError) else 'failed','stage':stage,
                'score':None,'error':repr(error),'seconds':time.monotonic()-stage_start,'total_seconds':time.monotonic()-start,
                'conclusion':'Недостаточно данных; технический исход'}
            (out/'traceback.txt').write_text(traceback.format_exc())
            save(out/'fit-event.json',{'phase':'single_fixed_focal_booster','status':result['status'],'error':repr(error)})
        save(out/'metrics.json',result)
        with (out/'report.md').open('a') as f:f.write('\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
        print('H50_RESULT',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
