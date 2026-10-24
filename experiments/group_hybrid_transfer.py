"""HT3: transfer H43/H73's partially pooled fallback into the generic hybrid."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from automl_models import GroupExpertModel
from automl_residual import CausalResidual

OUT = ROOT / 'results/team-runtime/group-hybrid.json'
WORK = ROOT / 'local_data/team-runtime'


def main():
    if OUT.exists():
        raise RuntimeError('Inspect existing receipt; never repeat completed fits')
    b = load_task(ROOT / 'drive-download-20261023T201455Z-1-001/Регрессия/task.md')
    of, ov, _ = validation_split(b)
    evidence = json.loads((ROOT / 'results/strongest-team/e2e-v2.json').read_text())
    candidate = next(x['selected'] for x in evidence['runs'] if x['task']=='regression' and x['scope']=='shadow')
    candidate = {k:v for k,v in candidate.items() if k!='panel_strategy'}
    record = {'status':'running', 'candidate':candidate, 'hashes':b.hashes, 'stages':[],
              'protocol':'HT3 transfers H43/H73 pooled group experts into generic causal hybrid fallback. Choose lowest-cardinality categorical X-only partition2..8 with every group>=500. Shrink n/(n+500). Existing selected policy recipe retained, no new features/hyperparameters. Latestinner gain>=.3 RMSE before one outer; no outer tuning. These splits and recipe already adaptively explored.'}
    def save():
        OUT.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    save()
    def stage(fit,valid,base,name):
        X,y=b.X.iloc[fit],b.y.iloc[fit]
        options=[c for c in X if not pd.api.types.is_numeric_dtype(X[c]) and 2<=X[c].nunique()<=8
                 and X[c].value_counts(dropna=False).min()>=500]
        if not options:
            raise ValueError('No eligible fit-only expert partition')
        column=min(options,key=lambda c:(X[c].nunique(),c))
        started=time.monotonic()
        group=GroupExpertModel('regression',{**candidate,'expert_column':column,
            'expert_min_rows':500,'max_expert_groups':8},8).fit(X,y)
        query=b.X.iloc[valid]
        global_prediction=group.global_.predict(query)
        local=group.predict(query)
        counts=group._keys(X).value_counts()
        n=group._keys(query).map(counts).fillna(0).to_numpy(float)
        fallback=global_prediction+n/(n+500)*(local-global_prediction)
        residual=CausalResidual(candidate['panel_time_column'],8,strict_key=True).fit(X,y)
        prediction=.5*residual.stable(query,fallback)+.5*residual.predict(query,fallback)
        seconds=time.monotonic()-started
        np.savez_compressed(WORK/('group-hybrid-'+name+'.npz'),positions=valid,prediction=prediction)
        result={'stage':name,'key':split_key(fit,valid),'column':column,'base':score('rmse',b.y.iloc[valid],base),
                'score':score('rmse',b.y.iloc[valid],prediction),'seconds':seconds,
                'rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                'global_plus_expert_fits':1+len(group.experts_),'residual_fits':int(residual.model_ is not None)}
        record['stages'].append(result);save();print(json.dumps(result),flush=True)
        return result
    p=next(Path('/private/tmp/alfa-strongest-e2e-v2/regression-shadow/agent_runs').glob('*/experiment-16-fold01/validation_predictions.npz'))
    with np.load(p) as z:
        valid=of[z['positions']]
        base=z['prediction']
    periods=b.X[candidate['panel_time_column']].to_numpy()
    assert np.array_equal(valid,np.flatnonzero(periods==4))
    fit=np.flatnonzero(periods<4)
    inner=stage(fit,valid,base,'inner')
    if inner['score']<=inner['base']-.3:
        base=pd.read_csv(ROOT/'local_data/strongest-team/regression-shadow/submission.csv')[b.target].to_numpy()
        stage(of,ov,base,'outer')
    else:
        record['decision']='reject_inner'
    record['status']='completed';save()


if __name__=='__main__':
    main()
