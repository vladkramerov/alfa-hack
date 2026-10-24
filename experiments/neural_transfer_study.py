"""Finite HT4/HT5 transfer of team FullTabM onto the delivered own controls."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automl_torch_tabm import TorchTabM
from automl_data import load_task,validation_split,split_key,score
from automl_engine import _search_windows
from automl_models import build_pipeline
from automl_worker import class_decisions,train_priors

OUT=ROOT/'results/neural-transfer/study.json'
WORK=ROOT/'local_data/neural-transfer'


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    report=json.loads(OUT.read_text()) if OUT.exists() else {
        'status':'running','tasks':[],
        'protocol':'HT4/HT5 team FullTabM-PWL8x96,60epoch,120sec/fit,4CPU. Own frozen delivery teacher. Inner-first: regression analytic simplex weight; classification grid0/.05/.1/.2/.4/.7/1 retaining own prior-power. Gates relativeRMSEgain>=.005 or F1gain>=.002, positiveweight>=.05. Freeze weight before at most one outer/task. Public outer historically reused, no independence. No integration before wholeCLI gain. Source exact Ivanfe674da; H109/H110 planned with no completed results. Different fresh own teacher graph, no repeatH108.'}
    def save():OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if report['status']=='completed':raise RuntimeError('Reuse results, no repeated fits')
    save()
    delivery=json.loads((ROOT/'results/strongest-team/e2e-v2.json').read_text())
    for task,folder in [('regression','Регрессия'),('classification','Многоклассовая классификация')]:
        existing=next((r for r in report['tasks'] if r['task']==task),None)
        if existing and existing['status']=='completed':continue
        b=load_task(ROOT/'drive-download-20261023T201455Z-1-001'/folder/'task.md')
        of,ov,_=validation_split(b)
        candidate=next(r['selected'] for r in delivery['runs'] if r['task']==task and r['scope']=='shadow')
        entry=existing or {'task':task,'candidate':candidate,'hashes':b.hashes,'stages':[],'status':'running'}
        if existing is None:report['tasks'].append(entry)
        if task=='regression':
            cache=next(Path('/private/tmp/alfa-strongest-e2e-v2/regression-shadow/agent_runs').glob('*/experiment-16-fold01/validation_predictions.npz'))
            with np.load(cache) as z:
                valid=of[z['positions']];base=z['prediction']
            fit=np.flatnonzero(b.X[candidate['panel_time_column']].to_numpy()<4)
            assert np.array_equal(valid,np.flatnonzero(b.X[candidate['panel_time_column']].to_numpy()==4))
        else:
            window=_search_windows(b,of,1800)[0][0];fit,valid=window['fit'],window['valid']
            with np.load(ROOT/'local_data/team-runtime/inner-teacher.npz') as z:
                assert np.array_equal(z['fit'],fit) and np.array_equal(z['valid'],valid)
                base=z['probabilities']
        for stage in ['inner','outer']:
            if stage=='outer':
                if not entry.get('admitted_outer'):break
                fit,valid=of,ov
                if task=='regression':base=pd.read_csv(ROOT/'local_data/strongest-team/regression-shadow/submission.csv')[b.target].to_numpy()
                else:
                    teacher_cache=WORK/'classification-outer-teacher.npz'
                    if teacher_cache.exists():
                        with np.load(teacher_cache) as z:
                            assert np.array_equal(z['fit'],fit) and np.array_equal(z['valid'],valid)
                            base=z['probabilities']
                    else:
                        start=time.monotonic();teacher=build_pipeline(task,candidate,8).fit(b.X.iloc[fit],b.y.iloc[fit])
                        assert np.array_equal(teacher.classes_,b.classes)
                        base=teacher.predict_proba(b.X.iloc[valid])
                        np.savez_compressed(teacher_cache,fit=fit,valid=valid,probabilities=base)
                        entry['outer_teacher_seconds']=time.monotonic()-start;save()
            if any(s['stage']==stage for s in entry['stages']):continue
            cache=WORK/(task+'-'+stage+'.npz');audit_path=cache.with_suffix('.audit.json')
            if cache.exists():
                with np.load(cache) as z:
                    assert np.array_equal(z['fit'],fit) and np.array_equal(z['valid'],valid)
                    prediction=z['prediction']
                audit=json.loads(audit_path.read_text())
            else:
                model=TorchTabM(kind=task,seconds=120,threads=4).fit(b.X.iloc[fit],b.y.iloc[fit])
                prediction=model.predict(b.X.iloc[valid]) if task=='regression' else model.predict_proba(b.X.iloc[valid])
                if task=='classification':assert np.array_equal(model.classes_,b.classes)
                audit=model.audit_
                # Persist expensive outputs before scoring or selecting weights.
                audit_path.write_text(json.dumps(audit,indent=2))
                np.savez_compressed(cache,fit=fit,valid=valid,prediction=prediction)
            actual=b.y.iloc[valid].to_numpy()
            if task=='regression':
                delta=prediction-base
                weight=float(np.clip(delta@(actual-base)/max(delta@delta,1e-12),0,1)) if stage=='inner' else entry['weight']
                labels=(1-weight)*base+weight*prediction
                baseline=score('rmse',actual,base);value=score('rmse',actual,labels)
                standalone=score('rmse',actual,prediction)
                admitted=weight>=.05 and value<=baseline*.995
            else:
                priors=train_priors(b.y.iloc[fit],b.classes)
                def value_at(weight):
                    return score('macro_f1',actual,class_decisions((1-weight)*base+weight*prediction,b.classes,priors,candidate.get('decision_power',0)))
                grid=[{'weight':w,'score':value_at(w)} for w in [0,.05,.1,.2,.4,.7,1]] if stage=='inner' else []
                weight=max(grid,key=lambda r:r['score'])['weight'] if grid else entry['weight']
                baseline=value_at(0);value=value_at(weight);standalone=value_at(1)
                admitted=weight>=.05 and value>=baseline+.002
                if grid:entry['inner_grid']=grid
            result={'stage':stage,'key':split_key(fit,valid),'base':baseline,'score':value,
                    'standalone':standalone,'weight':weight,'fit_audit':audit,
                    'rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
            entry['stages'].append(result)
            if stage=='inner':entry.update(weight=weight,admitted_outer=bool(admitted))
            save();print(json.dumps({k:v for k,v in result.items() if k!='fit_audit'}),flush=True)
        entry['status']='completed';save()
    report['status']='completed';save()


if __name__=='__main__':main()
