"""Isolated SPEC-006 evaluation; outer targets never passed to method selection."""
import json
from pathlib import Path
import sys,time,traceback,warnings,platform
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from automl_data import load_task,score
from experiments.run_twenty_hypotheses import ROOT,DATE,save
from experiments.twenty_common import make_model

def old_context(task,positions):
    old='HYP-012-01-regression-log1p_target' if task=='regression' else 'HYP-011-02-multiclass-none'
    full='HYP-021-01-regression-full-v1' if task=='regression' else 'HYP-021-02-multiclass-full-v1'
    result={}
    for prefix,dirname in [('baseline',old),('full',full)]:
        with np.load(ROOT/f'results/{DATE}/{dirname}/validation_predictions.npz',allow_pickle=True) as data:
            if not np.array_equal(data['positions'],positions): raise ValueError('Saved baseline position mismatch')
            for key in ['prediction','probabilities','classes']:
                result[prefix+'_'+key]=data[key].copy() if key in data else None
    return result

def main(manifest):
    request=json.loads(Path(manifest).read_text(encoding='utf-8')); out=Path(request['output']); job=request['candidate']; started=time.monotonic()
    result={'status':'started'}
    try:
        bundle=load_task(request['task_path'])
        with np.load(request['split_path']) as split: fit,valid=split['fit'],split['valid']
        X,y=bundle.X.iloc[fit],bundle.y.iloc[fit]; Xvalid=bundle.X.iloc[valid]
        context={**old_context(job['task'],valid),'fit_positions':fit,'valid_positions':valid,'cache_dir':Path(request['cache_dir']),
                 'output':out,'candidate_artifacts':request.get('candidate_artifacts',{})}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if job['method']=='features':
                from experiments.twenty_features import make_features
                model=make_model(job['task'],policy=job['policy'],feature=make_features(job['number'])).fit(X,y)
                prediction=model.predict(Xvalid)
                probabilities=model.predict_proba(Xvalid) if job['task']=='multiclass' else None
                classes=model.classes_ if job['task']=='multiclass' else None
                audit=model.audit_
                np.savez_compressed(out/'training-filter.npz',positions=fit,keep_mask=model.keep_mask_)
            else:
                from experiments.twenty_advanced import evaluate_advanced
                evaluated=evaluate_advanced(job['number'],job['task'],X,y,Xvalid,context)
                prediction=np.asarray(evaluated['prediction']); probabilities=evaluated.get('probabilities'); classes=evaluated.get('classes'); audit=evaluated.get('audit',{})
        if prediction.shape!=(len(valid),): raise ValueError('Prediction row/shape mismatch')
        artifact={'positions':valid,'prediction':prediction.astype(str) if job['task']=='multiclass' else prediction,
                  'y':bundle.y.iloc[valid].astype(str).to_numpy() if job['task']=='multiclass' else bundle.y.iloc[valid].to_numpy()}
        if job['task']=='multiclass':
            probabilities=np.asarray(probabilities); classes=np.asarray(classes).astype(str)
            if probabilities.shape!=(len(valid),len(classes)) or not np.isfinite(probabilities).all(): raise ValueError('Invalid probabilities')
            if not np.isin(prediction,classes).all(): raise ValueError('Unknown predicted class')
            artifact.update(probabilities=probabilities,classes=classes)
            save(out/'classification-report.json',classification_report(bundle.y.iloc[valid],prediction,labels=classes,output_dict=True,zero_division=0))
            pd.DataFrame(confusion_matrix(bundle.y.iloc[valid],prediction,labels=classes),index=classes,columns=classes).to_csv(out/'confusion-matrix.csv')
        elif not np.isfinite(prediction).all(): raise ValueError('Nonfinite predictions')
        np.savez_compressed(out/'validation_predictions.npz',**artifact)
        save(out/'preprocessing-audit.json',{'input_features':list(X),'outer_validation_unmodified':True,'outer_y_not_passed_to_method':True,**audit})
        result={'status':'completed','metric':bundle.metric,'score':score(bundle.metric,bundle.y.iloc[valid],prediction),
                'fit_rows':len(fit),'valid_rows':len(valid),'warnings':[str(w.message) for w in caught][:15],
                'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()}}
    except Exception as exc:
        result={'status':'failed','error':type(exc).__name__+': '+str(exc)}
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result['seconds']=time.monotonic()-started; save(out/'metrics.json',result)
    return 0 if result['status']=='completed' else 1

if __name__=='__main__': sys.exit(main(sys.argv[1]))
