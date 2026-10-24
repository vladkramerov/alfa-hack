"""Embed measured evidence only; this script does not fit any model."""
import json
from pathlib import Path
import pprint
import numpy as np
from automl_data import load_task,split_key,score,better
from automl_history import candidate_key

ROOT=Path(__file__).resolve().parents[1]


def main():
    records=json.loads((ROOT/'results/2026-10-23/new-family-records.json').read_text(encoding='utf-8'))
    known=[]; comparisons=[]
    for task in ['regression','multiclass']:
        bundle=load_task(ROOT/'data'/task/'task.md')
        split=np.load(ROOT/f'results/2026-10-23/series-{task}-preprocessing-v1/splits.npz')
        history=[]
        for record in records:
            if record['task']!=task: continue
            path=ROOT/record['directory']
            with np.load(path/'validation_predictions.npz',allow_pickle=True) as arrays:
                assert np.array_equal(arrays['positions'],split['outer_valid'])
                assert np.array_equal(arrays['y'],bundle.y.iloc[split['outer_valid']].to_numpy())
                independent=score(bundle.metric,bundle.y.iloc[split['outer_valid']],arrays['prediction'])
            assert abs(independent-record['score'])<1e-12
            history.append({'candidate':record['candidate'],'score':independent,'source':record['directory']+'/evidence.json'})
            comparisons.append({'task':task,'family':record['candidate']['family'],'metric':bundle.metric,'score':independent,'verified_without_fit':True})
        if task=='regression':
            candidate={'family':'catboost','parameters':{},'preprocessing':'legacy_standard','scale':True,
                       'feature_code':'def engineer(X):\n    X = X.copy()\n    X["year_built"] = X["year_built"].replace(0, np.nan)\n    return X\n',
                       'hypothesis':'Historical H4-zero-year, final refit only; no repeated validation'}
            history.append({'candidate':candidate,'score':43.0046386479306,'source':'results/history/2026-10-21/historical-experiments.json; sergey_branch reports/H4-zero-year, input hashes and outer rows verified'})
        else:
            # Store compatible baseline as evidence, not the incomparable historical random split.
            history.append({'candidate':{'family':'random_forest','parameters':{},'feature_code':''},'score':.572183507286843,
                            'source':'results/2026-10-23/HYP-001-12-multiclass-outer-baseline-forest/metrics.json'})
        best=None
        for r in history:
            r['candidate_key']=candidate_key(r['candidate'])
            if best is None or better(bundle.metric,r['score'],best['score']): best=r
        known.append({'hashes':{k:v for k,v in bundle.hashes.items() if k!='data_dictionary.csv'},'split_key':split_key(split['outer_fit'],split['outer_valid']),
                      'metric':bundle.metric,'records':history,'previous_directions':['CatBoost/RF baselines and tuning','L1/L2','correlation/VIF','scaling/Yeo-Johnson','missing indicators','flow ratios','equal ensembles'],
                      'selection_limitation':'Best observed compatible validation score; exploratory reuse of validation, not independent confirmation.'})
        print(task,best['candidate']['family'],best['score'])
    (ROOT/'automl_known.py').write_text('"""Measured prior evidence. Generated without training by experiments/build_known_history.py."""\nKNOWN_TASKS = '+pprint.pformat(known,width=120,sort_dicts=False)+'\n',encoding='utf-8')
    (ROOT/'results/2026-10-23/new-metric-verification.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')


if __name__=='__main__': main()
