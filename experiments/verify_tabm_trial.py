"""Independent no-fit replay of H108 artifacts and the frozen comparison."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    from research_memory import atomic_json
    suite = ROOT/'results/2026-10-24/series-tabm-mini-v1'
    attempts = json.loads((suite/'attempts.json').read_text(encoding='utf8'))
    history = json.loads((suite/'fit-history.json').read_text(encoding='utf8'))
    assert len({r['fingerprint'] for r in history}) == len(history)
    assert all(r['status']=='completed' for r in history)
    checks = []
    for run in attempts:
        out = ROOT/run['directory']
        request = json.loads((out/'request.json').read_text(encoding='utf8'))
        metrics = json.loads((out/'metrics.json').read_text(encoding='utf8'))
        assert metrics['status']=='completed'
        for name,digest in request['source_sha256'].items():
            assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==digest
        for name,digest in request['input_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        assert any(r['fingerprint']==metrics['fingerprint'] for r in history)
        with np.load(out/'splits.npz',allow_pickle=False) as z:
            fit,valid = z['fit'],z['valid']
        archive = out/'validation_predictions_portable.npz'
        if not archive.exists(): archive = out/'validation_predictions.npz'
        with np.load(archive,allow_pickle=False) as z:
            data = {k:z[k].copy() for k in z.files}
        assert not np.intersect1d(fit,valid).size
        assert np.array_equal(valid,data['positions'])
        train = pd.read_csv(ROOT/request['paths']['data'])
        target = train[request['paths']['target']].iloc[valid].to_numpy()
        X = train.drop(columns=['id',request['paths']['target']])
        model = joblib.load(out/'model.joblib')
        assert metrics['audit']['fit_only_preprocessing']
        assert metrics['audit']['checkpoint_criterion']=='training_loss_only'
        w = metrics['selected_weight']
        if run['kind']=='regression':
            np.testing.assert_allclose(data['y'],target,atol=1e-10,rtol=0)
            # Cast before weighting: NumPy scalar * float32 array stays float32,
            # whereas the runner subtracts a float64 control before multiplication.
            # Keep an independently arranged blend, but do all arithmetic in float64.
            pred = (1-w)*data['control'].astype(float)+w*data['raw_prediction'].astype(float)
            score = float(np.sqrt(np.mean((pred-target)**2)))
            pure = float(np.sqrt(np.mean((data['raw_prediction']-target)**2)))
            np.testing.assert_allclose(model.predict(X.iloc[valid]),data['raw_prediction'],atol=1e-5,rtol=1e-6)
            assert X.iloc[fit].Year_Factor.max()<X.iloc[valid].Year_Factor.min()
        else:
            np.testing.assert_array_equal(data['y'].astype(str),target.astype(str))
            combined = (1-w)*data['control_probabilities']+w*data['probabilities']
            pred = data['classes'][combined.argmax(1)]
            score = float(f1_score(target,pred,average='macro',zero_division=0))
            pure = float(f1_score(target,data['raw_prediction'],average='macro',zero_division=0))
            np.testing.assert_allclose(model.predict_proba(X.iloc[valid]),data['probabilities'],atol=1e-6,rtol=1e-5)
            groups = pd.util.hash_pandas_object(X,index=False).to_numpy()
            assert not np.intersect1d(groups[fit],groups[valid]).size
        if pred.dtype.kind in 'OUS': np.testing.assert_array_equal(pred,data['prediction'])
        else: np.testing.assert_allclose(pred,data['prediction'],atol=1e-10,rtol=1e-12)
        assert abs(score-metrics['score'])<1e-10
        assert abs(pure-metrics['pure_score'])<1e-10
        if run['stage']=='outer':
            assert request['frozen_weight']==w
            inner = next(r for r in attempts if r['kind']==run['kind'] and r['stage']=='inner')
            inner_result = json.loads((ROOT/inner['directory']/'metrics.json').read_text(encoding='utf8'))
            assert inner_result['eligible_for_outer'] and inner_result['selected_weight']==w
        checks.append({'kind':run['kind'],'stage':run['stage'],'score':score,'pure_score':pure,
                       'source_sha':True,'input_sha':True,'split':True,'target_alignment':True,
                       'prediction_replay':True,'metric_replay':True,'no_leakage':True,'frozen_weight':True})
    result = {'passed':True,'runs':checks,'unique_new_learner_fits':len(history),
              'old_model_refits':0,'verification_model_fits':0}
    atomic_json(suite/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
