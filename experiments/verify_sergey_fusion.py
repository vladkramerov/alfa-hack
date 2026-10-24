"""Independent no-fit verification of H107 saved arithmetic and input provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DAY=ROOT/'results/2026-10-24'
SUITE=DAY/'series-sergey-ivan-fusion-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    frozen=read(SUITE/'frozen-weights.json');weights=np.asarray(frozen['weights'])
    assert frozen['source_sha256']==sha(SUITE/'source/fusion_weights.py')
    assert (weights>=0).all() and abs(weights.sum()-1)<1e-10
    train=pd.read_csv(ROOT/'data/regression/train.csv')
    checked=[]
    for stage,index,year in [('inner',1,4),('outer',2,5)]:
        out=DAY/f'HYP-107-0{index}-{stage}-sergey-fusion'
        request=read(out/'request.json');metrics=read(out/'metrics.json')
        for name,digest in request['inputs_sha256'].items():assert sha(ROOT/name)==digest,name
        for name,digest in request['source_sha256'].items():assert sha(SUITE/'source'/name)==digest,name
        assert sha(ROOT/'data/regression/train.csv')==request['train_sha256']
        with np.load(out/'validation_predictions.npz',allow_pickle=False) as z:values={key:z[key] for key in z.files}
        with np.load(out/'splits.npz',allow_pickle=False) as z:fit=z['fit'];valid=z['valid']
        np.testing.assert_array_equal(valid,values['positions'])
        assert not np.intersect1d(fit,valid).size
        assert (train.Year_Factor.iloc[fit]<year).all() and (train.Year_Factor.iloc[valid]==year).all()
        np.testing.assert_allclose(train.site_eui.iloc[valid],values['y'],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(values['weights'],weights)
        np.testing.assert_allclose(values['prediction'],values['components']@weights,rtol=0,atol=1e-12)
        rmse=float(np.sqrt(np.mean((values['y']-values['prediction'])**2)))
        control=float(np.sqrt(np.mean((values['y']-values['components'][:,0])**2)))
        assert abs(rmse-metrics['score'])<1e-12 and abs(control-metrics['Sergey_control'])<1e-12
        assert metrics['passes_gate'] and rmse<metrics['Ivan_best_control'] and rmse<control
        checked.append({'stage':stage,'rmse':rmse,'Sergey':control,'gain':(control-rmse)/control,
            'source_data_truth_splits_verified':True,'prediction_sha256':sha(out/'validation_predictions.npz')})
    result={'status':'passed','learner_fits':0,'parameter_fits':0,'runs':checked,
        'cross_os_limitation':True,'public_outer_adaptively_reused':True,
        'runtime_graph_quality_separately_measured':False,'platform_improvement_verified':False}
    target=SUITE/'independent-verification.json';assert not target.exists()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
