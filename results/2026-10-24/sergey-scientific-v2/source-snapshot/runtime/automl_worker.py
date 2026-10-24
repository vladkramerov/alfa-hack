"""Standalone child process: all imports work from a flat platform archive."""
import json
from pathlib import Path
import sys
import time
import traceback
import re
from collections import Counter
import numpy as np
import pandas as pd
from automl_data import load_task,score,make_submission,atomic_submission
from automl_models import build_pipeline,predict_for_metric


def error_diagnostics(kind,actual,prediction,X=None,fit_target=None,classes=None):
    """Compact aggregate validation errors; no row positions or raw targets.

    These summaries inform research proposals only. They are never inputs to
    preprocessing, estimators, calibration or refit.
    """
    actual,prediction=np.asarray(actual),np.asarray(prediction)
    if actual.ndim!=1 or prediction.shape!=actual.shape or not len(actual):
        raise ValueError('Diagnostic arrays must be nonempty and aligned')
    result={'source':'validation_aggregates_only','rows':len(actual)}
    if kind=='classification':
        from sklearn.metrics import precision_recall_fscore_support
        labels=np.asarray(classes) if classes is not None else np.unique(np.concatenate([actual,prediction]))
        precision,recall,f1,support=precision_recall_fscore_support(actual,prediction,labels=labels,zero_division=0)
        indices=sorted(range(len(labels)),key=lambda i:(f1[i],-support[i],str(labels[i])))[:24]
        result.update(type='classification',class_count=len(labels),
            per_class=[{'class':str(labels[i])[:80],'precision':float(precision[i]),'recall':float(recall[i]),
                        'f1':float(f1[i]),'support':int(support[i])} for i in indices],
            top_confusions=[{'actual':str(a)[:80],'predicted':str(b)[:80],'count':int(n)}
                            for (a,b),n in Counter(zip(actual[actual!=prediction],prediction[actual!=prediction])).most_common(12)],
            classes_truncated=len(labels)>24)
        return result
    if kind!='regression':raise ValueError('Unsupported diagnostic task')
    actual,prediction=actual.astype(float),prediction.astype(float)
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():raise ValueError('Nonfinite regression diagnostic')
    def summary(mask):
        residual=prediction[mask]-actual[mask]
        return {'rows':int(mask.sum()),'rmse':float(np.sqrt(np.mean(residual**2))),
                'mae':float(np.mean(np.abs(residual))),'bias':float(np.mean(residual))}
    result.update(type='regression',overall=summary(np.ones(len(actual),dtype=bool)))
    training=np.asarray(fit_target,dtype=float) if fit_target is not None else np.array([])
    training=training[np.isfinite(training)]
    result['target_buckets']=[]
    if len(training):
        thresholds=np.unique(np.quantile(training,[.2,.4,.6,.8]))
        bins=np.searchsorted(thresholds,actual,side='right')
        result['bucket_definition']='quintiles_of_fit_targets'
        for bucket in range(len(thresholds)+1):
            mask=bins==bucket
            if mask.any():result['target_buckets'].append({'bucket':int(bucket),**summary(mask)})
    result['period_slices']=[]
    if X is not None:
        for column in X:
            if not re.search(r'year|period|date|год|период',str(column),re.I) or re.search(r'built|birth|построй',str(column),re.I):continue
            values=X[column]
            if not pd.api.types.is_numeric_dtype(values) or values.nunique(dropna=True)>12:continue
            for value in sorted(values.dropna().unique())[-8:]:
                mask=values.eq(value).fillna(False).to_numpy(bool)
                if mask.any():result['period_slices'].append({'column':str(column)[:80],'period':str(value)[:40],**summary(mask)})
            break
    return result


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
            try:
                diagnostic_prediction=pred
                if bundle.task_type=='classification' and bundle.metric in {'roc_auc','log_loss'}:
                    diagnostic_prediction=model.classes_[np.argmax(artifact['probabilities'],axis=1)]
                result['diagnostics']=error_diagnostics(bundle.task_type,bundle.y.iloc[valid],diagnostic_prediction,
                    X=bundle.X.iloc[valid],fit_target=bundle.y.iloc[fit] if bundle.task_type=='regression' else None,
                    classes=model.classes_ if bundle.task_type=='classification' else None)
            except (ValueError,TypeError,OverflowError) as exc:
                result['diagnostics']={'status':'unavailable','error':type(exc).__name__}
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
