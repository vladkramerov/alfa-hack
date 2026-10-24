"""Independent arithmetic/source/data checks and no-fit prediction replay."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2026-10-24/external-repositories-review-v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from external_multilag import MultiLagRegressor
    from external_key import AutoKeyMemory
    train=pd.read_csv(ROOT/'data/regression/train.csv');X=train.drop(columns=['id','site_eui'])
    verified=[]
    for number,series,slug in [(105,'series-external-ideas-v1','external-multilag'),(106,'series-external-key-v1','external-key')]:
        suite=ROOT/'results/2026-10-24'/series
        frozen=read(suite/'frozen.json')
        for name,digest in frozen['sources'].items():assert sha(suite/'source'/name)==digest,name
        attempts=read(suite/'attempts.json');assert len(attempts)==1 and attempts[0]['status']=='completed'
        out=ROOT/attempts[0]['directory'];request=read(out/'request.json');metrics=read(out/'metrics.json')
        for name,digest in request['input_sha256'].items():assert sha(ROOT/name)==digest,name
        with np.load(out/'splits.npz',allow_pickle=False) as z:fit=z['fit'];valid=z['valid']
        with np.load(out/'validation_predictions.npz',allow_pickle=False) as z:values={k:z[k] for k in z.files}
        assert not np.intersect1d(fit,valid).size and np.array_equal(values['positions'],valid)
        assert (X.iloc[fit].Year_Factor<4).all() and (X.iloc[valid].Year_Factor==4).all()
        np.testing.assert_array_equal(train.site_eui.iloc[valid],values['y'])
        np.testing.assert_array_equal(values['pure'],values['raw_prediction'])
        np.testing.assert_allclose(values['blend75'],.75*values['raw_prediction']+.25*values['control'],rtol=0,atol=1e-12)
        scores={arm:float(np.sqrt(np.mean((values[arm]-values['y'])**2))) for arm in ['pure','blend75']}
        for arm,score in scores.items():assert abs(score-metrics['arm_scores'][arm])<1e-12
        assert metrics['selected_arm']==min(scores,key=scores.get) and metrics['eligible_for_outer'] is False
        assert abs(min(scores.values())-metrics['score'])<1e-12
        with (patch.object(MultiLagRegressor,'fit',side_effect=AssertionError('Verifier cannot fit')),
              patch.object(AutoKeyMemory,'fit',side_effect=AssertionError('Verifier cannot fit'))):
            model=joblib.load(out/'model.joblib')
            replay=model.predict(X.iloc[valid]) if number==105 else model.predict_with_fallback(X.iloc[valid],values['control'])
        np.testing.assert_allclose(replay,values['raw_prediction'],rtol=1e-12,atol=1e-10)
        claims=read(suite/'fit-history.json')
        assert len(claims)==1 and claims[0]['status']=='completed' and claims[0]['ledger_sha256']
        verified.append({'hypothesis':f'HYP-{number:03d}','stage':'inner','scores':scores,
            'control':metrics['control_score'],'relative_gain_vs_Sergey':(metrics['control_score']-metrics['score'])/metrics['control_score'],
            'passes_preregistered_gate':False,'source_data_splits_verified':True,'replay_verified':True,
            'learner_fits':metrics['learner_fits'],'report':str((out/'report.md').relative_to(ROOT)),
            'prediction_sha256':sha(out/'validation_predictions.npz')})
    result={'status':'passed','verification_model_fits':0,'runs':verified,
            'production_promoted':False,'outer_not_run_due_inner_gate':True,
            'source_sha256':sha(__file__),'hidden_or_platform_score_verified':False}
    target=OUT/'verification.json';assert not target.exists()
    target.open('x',encoding='utf8').write(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
