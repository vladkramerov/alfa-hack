"""Outer labels are used only after independent train-only graph evaluation."""
import json,sys,time,traceback,platform
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
from automl_data import load_task,score
from experiments.run_combinations import ROOT,DATE,save

OLD={'regression':{'raw':'HYP-012-01-regression-log1p_target','experts':'HYP-035-01-regression-spec006',
 'season':'HYP-030-01-regression-spec006','relative':'HYP-031-01-regression-spec006'},
 'multiclass':{'raw':'HYP-011-02-multiclass-none','smote':'HYP-038-01-multiclass-spec006','frequency':'HYP-017-01-multiclass-frequency'}}
def main(request):
    start=time.monotonic();req=json.loads(Path(request).read_text(encoding='utf-8'));out=Path(req['output']);job=req['job']
    try:
        task=load_task(req['task_path'])
        with np.load(req['split_path']) as z:fit,valid=z['fit'],z['valid']
        context={k:req[k] for k in ('output','cache_dir','legacy_cache_dir')};context.update(fit_positions=fit,valid_positions=valid,threads=4)
        components={}
        for name,folder in OLD[job['task']].items():
            with np.load(ROOT/f'results/{DATE}'/folder/'validation_predictions.npz',allow_pickle=True) as z:
                if not np.array_equal(z['positions'],valid):raise ValueError('Saved outer positions mismatch')
                components[name]=z['prediction'].copy() if job['task']=='regression' else {k:z[k].copy() for k in ('positions','probabilities','classes')}
        context['outer_components']=components
        X=task.X.iloc[fit];y=np.asarray(task.y)[fit];Xvalid=task.X.iloc[valid]
        if job['task']=='regression':
            from experiments.reg_combinations import evaluate_reg
            result=evaluate_reg(job['number'],X,y,Xvalid,context)
        else:
            from experiments.class_combinations import evaluate_class
            result=evaluate_class(job['number'],X,y,Xvalid,context)
        prediction=np.asarray(result['prediction']);truth=np.asarray(task.y)[valid]
        if prediction.shape!=(len(valid),):raise ValueError('Invalid prediction shape')
        metric='rmse' if job['task']=='regression' else 'macro_f1'
        value=float(score(metric,truth,prediction))
        if not np.isfinite(value):raise ValueError('Nonfinite score')
        arrays={'positions':valid,'prediction':prediction,'y':truth}
        for name in ('probabilities','classes'):
            if name in result:arrays[name]=result[name]
        if job['task']=='multiclass':
            arrays['prediction']=prediction.astype(str);arrays['y']=truth.astype(str);arrays['classes']=np.asarray(arrays['classes']).astype(str)
            save(out/'classification-report.json',classification_report(truth,prediction,output_dict=True,zero_division=0))
        np.savez_compressed(out/'validation_predictions.npz',**arrays)
        save(out/'audit.json',result['audit']);save(out/'learned-state.json',{'candidate':result.get('candidate',job['candidate']),**result.get('learned_state',{})})
        record={'status':'completed','metric':metric,'score':value,'fit_rows':len(fit),'valid_rows':len(valid),
                'seconds':time.monotonic()-start,'candidate':result.get('candidate',job['candidate']),
                'learned_state':result.get('learned_state',{}),'python':sys.version,'platform':platform.platform(),'outer_baseline_refits':0}
    except Exception as exc:
        traceback.print_exc();record={'status':'failed','error':repr(exc),'seconds':time.monotonic()-start}
    save(out/'metrics.json',record)
    if record['status']!='completed':sys.exit(1)
if __name__=='__main__':main(sys.argv[1])
