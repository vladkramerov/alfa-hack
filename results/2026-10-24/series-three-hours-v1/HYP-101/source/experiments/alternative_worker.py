"""No outer labels cross the train-only alternative-method boundary."""
import json,sys,time,traceback,platform
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
from automl_data import load_task,score
from experiments.run_alternatives import ROOT,ORIGINAL,save
def main(request):
    start=time.monotonic();req=json.loads(Path(request).read_text(encoding='utf8'));out=Path(req['output']);job=req['job']
    try:
        bundle=load_task(req['task_path'])
        with np.load(req['split_path']) as z:fit,valid=z['fit'],z['valid']
        context={k:req[k] for k in ['output','cache_dir','teacher_cache_dir','legacy_cache_dir']};context.update(fit_positions=fit,valid_positions=valid,threads=4)
        base=ROOT/f'results/{ORIGINAL}/HYP-043-01-regression-combinations'
        if job['task']=='regression':
            with np.load(base/'validation_predictions.npz') as z:
                if not np.array_equal(z['positions'],valid):raise ValueError('H43 positions mismatch')
                context['baseline_prediction']=z['prediction'].copy()
            context['h43_model_path']=json.loads((base/'audit.json').read_text(encoding='utf8'))['model_cache']['artifact']
        if job['number']==49:
            context['teacher_meta_state']=json.loads((ROOT/f'results/{ORIGINAL}/HYP-048-01-multiclass-combinations/learned-state.json').read_text(encoding='utf8'))['meta_state']
        X,y=bundle.X.iloc[fit],np.asarray(bundle.y)[fit];Xvalid=bundle.X.iloc[valid]
        if job['number']==69:
            from automl_temporal_memory import evaluate_memory
            result=evaluate_memory(X,y,Xvalid,context['baseline_prediction'])
            np.savez_compressed(out/'profile-memory-components.npz',positions=valid,memory_mean=result.pop('memory_mean'),memory_weight=result.pop('memory_weight'),baseline=context['baseline_prediction'])
        elif job['task']=='regression':
            from experiments.next_regression import evaluate_reg
            result=evaluate_reg(job['number'],X,y,Xvalid,context)
        else:
            from experiments.next_classification import evaluate_class
            result=evaluate_class(job['number'],X,y,Xvalid,context)
        truth=np.asarray(bundle.y)[valid];pred=np.asarray(result['prediction']);metric='rmse' if job['task']=='regression' else 'macro_f1'
        if pred.shape!=(len(valid),):raise ValueError('Prediction dimensions invalid')
        value=float(score(metric,truth,pred))
        if not np.isfinite(value):raise ValueError('Nonfinite metric')
        arrays={'positions':valid,'y':truth,'prediction':pred}
        if job['task']=='multiclass':
            arrays.update(y=truth.astype(str),prediction=pred.astype(str),probabilities=result['probabilities'],classes=np.asarray(result['classes']).astype(str))
            save(out/'classification-report.json',classification_report(truth,pred,output_dict=True,zero_division=0))
        np.savez_compressed(out/'validation_predictions.npz',**arrays)
        save(out/'audit.json',result['audit']);save(out/'learned-state.json',result.get('learned_state',{}))
        metrics={'status':'completed','metric':metric,'score':value,'candidate':result.get('candidate',job['candidate']),
                 'learned_state':result.get('learned_state',{}),'fit_rows':len(fit),'valid_rows':len(valid),'outer_baseline_refits':0,
                 'seconds':time.monotonic()-start,'python':sys.version,'platform':platform.platform()}
    except Exception as exc:
        traceback.print_exc();metrics={'status':'failed','error':repr(exc),'seconds':time.monotonic()-start}
    save(out/'metrics.json',metrics)
    if metrics['status']!='completed':sys.exit(1)
if __name__=='__main__':main(sys.argv[1])
