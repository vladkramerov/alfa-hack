"""Standalone child process: all imports work from a flat platform archive."""
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
from automl_data import load_task,score,make_submission,atomic_submission
from automl_models import build_pipeline,predict_for_metric


def run(manifest):
    request=json.loads(Path(manifest).read_text(encoding='utf-8'))
    output=Path(request['output']); output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    try:
        bundle=load_task(request['task_path'])
        model=build_pipeline(bundle.task_type,request['candidate'],request.get('threads',4))
        if request['mode']=='evaluate':
            with np.load(request['split_path']) as split: fit,valid=split['fit'],split['valid']
            model.fit(bundle.X.iloc[fit],bundle.y.iloc[fit])
            pred=predict_for_metric(model,bundle.X.iloc[valid],bundle.metric)
            result={'status':'completed','metric':bundle.metric,'score':score(bundle.metric,bundle.y.iloc[valid],pred),'fit_rows':len(fit),'valid_rows':len(valid)}
            artifact={'positions':valid,'y':bundle.y.iloc[valid].to_numpy(),'prediction':pred}
            if bundle.task_type=='classification':
                artifact['probabilities']=model.predict_proba(bundle.X.iloc[valid]); artifact['classes']=model.classes_
            np.savez_compressed(output/'validation_predictions.npz',**artifact)
        else:
            model.fit(bundle.X,bundle.y)
            pred=predict_for_metric(model,bundle.X_test,bundle.metric)
            if bundle.metric=='rmsle': pred=np.maximum(pred,0)
            outputs=[c for c in bundle.sample if c!=bundle.id_column]
            if bundle.metric in ['roc_auc','log_loss'] and len(outputs)>1:
                mapping={str(c):i for i,c in enumerate(model.classes_)}
                if set(outputs)!=set(mapping): raise ValueError('Submission probability columns must name classes')
                pred=pred[:,[mapping[c] for c in outputs]]
            atomic_submission(bundle,make_submission(bundle,pred),output/'candidate_submission.csv')
            result={'status':'completed','mode':'refit','train_rows':len(bundle.train),'test_rows':len(bundle.test)}
        if hasattr(model,'audit_'):
            result['fit_retained_rows']=model.audit_['fit_retained_rows']
            if hasattr(model,'learned_state_'):result['learned_state']=model.learned_state_
            if request['candidate'].get('strategy')=='reg_convex_blend' and hasattr(model,'weights_'):
                result['learned_state']={'blend_weights':np.asarray(model.weights_).tolist()}
            if hasattr(model,'calibration_multipliers_'):
                result['calibration_multipliers']={str(k):float(v) for k,v in model.calibration_multipliers_.items()}
            (output/'preprocessing-audit.json').write_text(json.dumps(model.audit_,indent=2,default=str),encoding='utf-8')
            np.savez_compressed(output/'training-filter.npz',keep_mask=model.keep_mask_,removed_positions=model.removed_positions_)
        result['seconds']=time.monotonic()-started
    except Exception as exc:
        result={'status':'failed','error':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-started}
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return 0 if result['status']=='completed' else 1


if __name__=='__main__': sys.exit(run(sys.argv[1]))
