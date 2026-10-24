"""One isolated SPEC-004 run; no retries, baselines or hidden tuning."""
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from automl_data import load_task,score
from experiments.novel_models import fit_experiment,predictions,inverse_group_weights


def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def main(manifest):
    request=json.loads(Path(manifest).read_text(encoding='utf-8')); output=Path(request['output']); started=time.monotonic()
    candidate=request['candidate']; result={'status':'started','candidate':candidate}
    try:
        bundle=load_task(request['task_path'])
        with np.load(request['split_path']) as splits: fit,valid=splits['fit'],splits['valid']
        X,y=bundle.X.iloc[fit],bundle.y.iloc[fit]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model=fit_experiment(candidate,X,y)
            prediction,probabilities,classes=predictions(candidate,model,bundle.X.iloc[valid])
        result.update(status='completed',metric=bundle.metric,score=score(bundle.metric,bundle.y.iloc[valid],prediction),
                      fit_rows=len(fit),valid_rows=len(valid),warnings=[str(w.message) for w in caught][:30],
                      packages={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'xgboost':xgboost.__version__},platform=platform.platform())
        artifact={'positions':valid,'y':bundle.y.iloc[valid].astype(str).to_numpy() if candidate['task']=='multiclass' else bundle.y.iloc[valid].to_numpy(),'prediction':prediction.astype(str) if candidate['task']=='multiclass' else prediction}
        if probabilities is not None:
            artifact.update(probabilities=probabilities,classes=classes.astype(str))
            result['accuracy']=float(accuracy_score(bundle.y.iloc[valid],prediction)); result['classes']=classes.tolist()
            result['probability_row_sum_range']=[float(probabilities.sum(axis=1).min()),float(probabilities.sum(axis=1).max())]
            save(output/'classification-report.json',classification_report(bundle.y.iloc[valid],prediction,labels=classes,output_dict=True,zero_division=0))
            pd.DataFrame(confusion_matrix(bundle.y.iloc[valid],prediction,labels=classes),index=classes,columns=classes).to_csv(output/'confusion-matrix.csv')
        np.savez_compressed(output/'validation_predictions.npz',**artifact)
        audit={'input_features':list(X),'identifier_excluded':bundle.id_column not in X,'target_excluded':bundle.target not in X}
        if hasattr(model,'audit_'): audit.update(model.audit_)
        elif 'new_features' in model.named_steps:
            features=model.named_steps['new_features']; audit['new_features']=features.audit_ if hasattr(features,'audit_') else {'class':type(features).__name__}
            if hasattr(features,'fold_indices_'):
                arrays={}
                for i,(a,b) in enumerate(features.fold_indices_): arrays[f'fold{i}_fit']=a; arrays[f'fold{i}_valid']=b
                np.savez_compressed(output/'encoding-folds.npz',**arrays)
        if candidate['transform']=='inverse_group_weights':
            weights=inverse_group_weights(X); np.savez_compressed(output/'training-weights.npz',positions=fit,weights=weights)
            audit.update(weights_sum=float(weights.sum()),weight_min=float(weights.min()),weight_max=float(weights.max()),weights_normalized=False)
        if hasattr(model,'named_steps'):
            encoder=model.named_steps['encode']; audit.update(numeric_columns=encoder.numeric_,categorical_columns=encoder.categorical_,scaled=encoder.scaler_ is not None,native_categories=encoder.native)
        save(output/'preprocessing-audit.json',audit)
    except Exception as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc))
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result['seconds']=time.monotonic()-started; save(output/'metrics.json',result)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__': sys.exit(main(sys.argv[1]))
