"""Independent arithmetic, truth/split and causality checks, with no fitting."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'results/2026-10-24/sergey-improvement-v1'


def main():
    train=pd.read_csv(ROOT/'data/regression/train.csv')
    rows=[];checks=0
    for number in [70,71,72,73,74]:
        for stage in ['primary','secondary']:
            out=ROOT/f'results/2026-10-24/HYP-{number:03d}-{1 if stage=="primary" else 2:02d}-regression-panel-{stage}'
            metric=json.loads((out/'metrics.json').read_text())
            with np.load(out/'validation_predictions.npz',allow_pickle=False) as z:
                positions=z['positions'];truth=z['y'];prediction=z['prediction'];control=z['control']
                assert np.array_equal(truth,train.iloc[positions].site_eui)
                assert np.isfinite(prediction).all() and len(prediction)==len(truth)
                expected_year=5 if stage=='primary' else 4
                assert np.all(train.iloc[positions].Year_Factor.to_numpy()==expected_year)
                with np.load(out/'splits.npz') as splits:
                    assert not set(splits['fit']) & set(splits['valid'])
                    assert np.all(train.iloc[splits['fit']].Year_Factor<expected_year)
                current=float(np.sqrt(np.mean((prediction-truth)**2)))
                old=float(np.sqrt(np.mean((control-truth)**2)))
                assert abs(current-metric['score'])<1e-10 and abs(old-metric['control'])<1e-10
                if number in [70,71]:
                    assert np.all((z['memory_weight']>=0)&(z['memory_weight']<=1))
                    np.testing.assert_allclose(prediction,z['memory_weight']*z['memory_mean']+(1-z['memory_weight'])*z['base_prediction'],rtol=1e-12,atol=1e-12)
                if number==73:
                    parts=[]
                    for n in [70,72]:
                        source=ROOT/f'results/2026-10-24/HYP-{n:03d}-{1 if stage=="primary" else 2:02d}-regression-panel-{stage}/validation_predictions.npz'
                        with np.load(source) as part:parts.append(part['prediction'])
                    np.testing.assert_array_equal(prediction,.5*parts[0]+.5*parts[1])
                    # Resample complete stable-profile clusters, not individual rows.
                    keys=pd.util.hash_pandas_object(train.iloc[positions][['State_Factor','building_class','facility_type','floor_area','year_built']],index=False)
                    codes,labels=pd.factorize(keys,sort=True)
                    count=np.bincount(codes);sse_old=np.bincount(codes,weights=(control-truth)**2);sse_new=np.bincount(codes,weights=(prediction-truth)**2)
                    rng=np.random.default_rng(20261024);gains=[]
                    for _ in range(500):
                        selected=rng.integers(0,len(count),len(count))
                        rmse_old=np.sqrt(sse_old[selected].sum()/count[selected].sum())
                        rmse_new=np.sqrt(sse_new[selected].sum()/count[selected].sum())
                        gains.append((rmse_old-rmse_new)/rmse_old)
                    interval=np.quantile(gains,[.025,.975]).tolist()
                    assert interval[0]>0,interval
                else:interval=None
            checks+=9+(number in [70,71])+(number==73)*2
            rows.append({'hypothesis':f'HYP-{number:03d}','stage':stage,'rmse':current,'control_rmse':old,
                         'relative_gain':(old-current)/old,'cluster_bootstrap_95pct_gain_interval':interval})
    classification=[]
    truth=pd.read_csv(ROOT/'data/multiclass/train.csv').attack_cat.astype(str).to_numpy()
    for stage in ['primary','secondary']:
        out=SUITE/f'control-multiclass-{stage}'
        with np.load(out/'validation_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['y'],truth[z['positions']])
            np.testing.assert_allclose(z['probabilities'].sum(axis=1),1.,atol=1e-6)
            assert np.array_equal(z['prediction'],z['classes'][z['probabilities'].argmax(axis=1)])
            value=float(f1_score(z['y'],z['prediction'],average='macro',zero_division=0))
        assert abs(value-json.loads((out/'metrics.json').read_text())['score'])<1e-12
        checks+=4;classification.append({'stage':stage,'macro_f1':value,'changed_classifier':False})
    with zipfile.ZipFile(ROOT/'dist/solution_alternative_v2.zip') as old:
        unchanged=['automl_class_combo.py','automl_reg_combo.py','automl_advanced.py','automl_preprocessing.py','automl_combo_policy.py','automl_feature_sets.py']
        for name in unchanged:
            assert old.read(name).replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            checks+=1
    result={'status':'passed','checks':checks,'regression':rows,'classification':classification,
            'classifier_dependencies_identical_to_reference':unchanged,
            'bootstrap_replicates':500,'bootstrap_unit':'stable physical-profile cluster',
            'bootstrap_limitation':'Descriptive paired uncertainty on repeatedly examined public data, not private generalization or selection-adjusted inference.',
            'new_model_fits_in_verifier':0,'hidden_platform_score_verified':False}
    (SUITE/'independent-metric-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'status':'passed','checks':checks,'finalist':[r for r in rows if r['hypothesis']=='HYP-073'],'classification':classification}))


if __name__=='__main__':main()
