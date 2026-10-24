"""H107: one inner-only parameter fit, cached components, no learner refits."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fusion_weights import fit_simplex, combine

DAY=ROOT/'results/2026-10-24'
SERGEY=ROOT/'tmp/sergey-24e3311f6ba9/results/2026-10-24'
SUITE=DAY/'series-sergey-ivan-fusion-v1'
COMPONENTS=['SergeyH73','IvanH70','IvanH83','IvanH85']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def now():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def ledger(text):
    for file in ['HYPOTHESES.md','results/README.md']:
        with (ROOT/file).open('a',encoding='utf8') as stream:stream.write('\n'+text+'\n')


def paths(stage):
    if stage=='inner':
        return [SERGEY/'HYP-073-02-regression-panel-secondary',
            DAY/'HYP-070-03-inner-high-sprint', DAY/'HYP-083-01-inner-default-sprint',
            DAY/'HYP-085-01-inner-default-sprint']
    return [SERGEY/'HYP-073-01-regression-panel-primary',
        DAY/'HYP-070-04-outer-high-sprint',DAY/'HYP-083-02-outer-default-sprint',
        DAY/'HYP-085-02-outer-default-sprint']


def load_stage(stage):
    records=[]
    for directory in paths(stage):
        with np.load(directory/'validation_predictions.npz',allow_pickle=False) as z:
            record={key:z[key] for key in ['positions','y','prediction']}
        with np.load(directory/'splits.npz',allow_pickle=False) as z:
            record['fit']=z['fit'];record['valid']=z['valid']
        records.append(record)
    reference=records[0]
    for record in records:
        for key in ['positions','fit','valid']:np.testing.assert_array_equal(record[key],reference[key])
        np.testing.assert_allclose(record['y'],reference['y'],rtol=0,atol=1e-10)
        assert np.isfinite(record['prediction']).all()
    assert not np.intersect1d(reference['fit'],reference['valid']).size
    assert np.array_equal(reference['positions'],reference['valid'])
    return np.column_stack([r['prediction'] for r in records]),reference


def record(stage,matrix,data,weights):
    out=DAY/f'HYP-107-0{1 if stage=="inner" else 2}-{stage}-sergey-fusion'
    out.mkdir()
    save(out/'request.json',{'hypothesis':'HYP-107','stage':stage,'started':now(),
        'source_sha256':{f.name:sha(f) for f in (SUITE/'source').iterdir()},
        'inputs_sha256':{str((p/name).relative_to(ROOT)):sha(p/name) for p in paths(stage)
            for name in ['validation_predictions.npz','splits.npz']},
        'train_sha256':sha(ROOT/'data/regression/train.csv'),
        'ledger_sha256':sha(ROOT/'HYPOTHESES.md')})
    prediction=combine(matrix,weights)
    score=float(np.sqrt(np.mean((prediction-data['y'])**2)))
    control=float(np.sqrt(np.mean((matrix[:,0]-data['y'])**2)))
    benchmark=30.636363541581975 if stage=='inner' else 34.21382789041039
    passed=score<min(control,benchmark) and int((weights>1e-8).sum())>=2
    metrics={'status':'completed','hypothesis':'HYP-107','stage':stage,'metric':'rmse','score':score,
        'Sergey_control':control,'Ivan_best_control':benchmark,'passes_gate':passed,
        'weights':weights.tolist(),'components':COMPONENTS,'learner_fits':0,
        'parameter_fits':int(stage=='inner'),'old_model_refits':0,
        'relative_gain_vs_Sergey':(control-score)/control,'relative_gain_vs_Ivan':(benchmark-score)/benchmark,
        'fit_rows':len(data['fit']),'valid_rows':len(data['valid']),
        'cross_os_cached_components':True,'hidden_score_verified':False,'completed':now()}
    np.savez_compressed(out/'validation_predictions.npz',positions=data['positions'],y=data['y'],
        prediction=prediction,components=matrix,weights=weights)
    np.savez_compressed(out/'splits.npz',fit=data['fit'],valid=data['valid'])
    save(out/'metrics.json',metrics)
    report=f'''# HYP-107 / {stage}

SPEC014, PLAN014. Команда: `.venv-ml/Scripts/python.exe -X utf8 experiments/sergey_fusion.py`.
Исход: RMSE {score:.12f}; Сергей {control:.12f}; наш прежний лучший {benchmark:.12f}.
Весы ({', '.join(COMPONENTS)}): {weights.tolist()}.
Гейт пройден: {passed}. Старые learner refits0; parameter fits {int(stage=='inner')}.
Все positions/splits совпали; y CSV roundtrip atol1e-10. Входные/source SHA в request.json.
Inner-веса зафиксированы до открытия outer. Public outer уже адаптивно использовался; компоненты Mac/Windows.
Это проверка композиции сохранённых прогнозов, не доказательство выигрыша поставки на hidden platform.
'''
    (out/'report.md').write_text(report,encoding='utf8')
    ledger(f'HYP-107/{stage}: RMSE={score}; Sergey={control}; Ivan={benchmark}; passes_gate={passed}; '
        f'[report]({out.relative_to(ROOT).as_posix()}/report.md). Проверка завершена, не повторять.')
    print(json.dumps(metrics,ensure_ascii=False),flush=True)
    return metrics


def main():
    started=time.monotonic()
    history=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    sergey_history=(ROOT/'tmp/sergey-24e3311f6ba9/HYPOTHESES.md').read_text(encoding='utf8')
    assert 'HYP-107 registered' in history and 'HYP-107/inner:' not in history and sergey_history
    SUITE.mkdir()
    (SUITE/'source').mkdir()
    for source in [Path(__file__),ROOT/'fusion_weights.py',ROOT/'docs/hypotheses/HYP-107-sergey-ivan-fusion.md']:
        shutil.copyfile(source,SUITE/'source'/source.name)
    save(SUITE/'started.json',{'started':now(),'ledger_sha256':sha(ROOT/'HYPOTHESES.md'),
        'Sergey_ledger_sha256':sha(ROOT/'tmp/sergey-24e3311f6ba9/HYPOTHESES.md'),
        'train_sha256':sha(ROOT/'data/regression/train.csv'),
        'components':COMPONENTS,'optimizer':{'method':'SLSQP','maxiter':200,'ftol':1e-12},
        'budget_seconds':900,'learner_fits':0})
    try:
        matrix,data=load_stage('inner')
        # Registration/started evidence is on disk before the only parameter fit.
        weights=fit_simplex(matrix,data['y'])
        save(SUITE/'frozen-weights.json',{'weights':weights.tolist(),'selected_on':'Year4_inner_only',
            'frozen':now(),'source_sha256':sha(ROOT/'fusion_weights.py')})
        inner=record('inner',matrix,data,weights)
        outer=None
        if inner['passes_gate']:
            assert time.monotonic()-started<900
            # No outer predictions are opened before the weights are frozen above.
            matrix,data=load_stage('outer')
            outer=record('outer',matrix,data,weights)
        save(SUITE/'outcome.json',{'status':'completed','seconds':time.monotonic()-started,
            'inner':inner,'outer':outer,'production_promoted':False})
    except Exception as error:
        save(SUITE/'failure.json',{'status':'failed','error':repr(error),'seconds':time.monotonic()-started})
        ledger(f'HYP-107 technical failure: {error!r}; {SUITE.relative_to(ROOT).as_posix()}/failure.json. Не повторять parameter fit.')
        raise


if __name__=='__main__':main()
