"""Bounded H108 research: cached controls, inner-frozen blends, no old fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_memory import Memory, atomic_json, timestamp

SUITE = ROOT / 'results/2026-10-24/series-tabm-mini-v1'
CARD = ROOT / 'docs/hypotheses/HYP-108-tabm-mini.md'


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_arrays(arrays):
    """Keep numeric values exact and encode string labels without object pickle."""
    import numpy as np
    result = {}
    for name,value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind == 'O':
            if not all(isinstance(v,str) for v in value.flat):
                raise ValueError('Only string object arrays may be exported: '+name)
            value = value.astype(str)
        result[name] = value
    return result


def paths(kind, stage):
    if kind == 'regression':
        control = f'results/2026-10-24/HYP-107-{1 if stage=="inner" else 2:02d}-{stage}-sergey-fusion/validation_predictions.npz'
        split = ('results/2026-10-24/HYP-070-03-inner-high-sprint/splits.npz' if stage=='inner'
                 else 'results/2026-10-24/HYP-070-04-outer-high-sprint/splits.npz')
        data, target = 'data/regression/train.csv', 'site_eui'
    else:
        directory = f'results/2026-10-24/HYP-078-{1 if stage=="inner" else 2:02d}-{stage}-default-sprint'
        control, split = directory+'/validation_predictions.npz', directory+'/splits.npz'
        data, target = 'data/multiclass/train.csv', 'attack_cat'
    return {'control':control,'split':split,'data':data,'target':target}


def worker(request_path):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import f1_score
    from experimental_tabm import TabMMini
    request = read(request_path)
    out = Path(request['output'])
    root = Path(request['root'])
    started = time.monotonic()
    memory = Memory(request['history'], root/'HYPOTHESES.md')
    claim = None
    try:
        for name,digest in request['input_sha256'].items():
            if sha(root/name) != digest: raise ValueError('Input changed: '+name)
        for name,digest in request['source_sha256'].items():
            if sha(root/name) != digest: raise ValueError('Source changed: '+name)
        frame = pd.read_csv(root/request['paths']['data'])
        y = frame[request['paths']['target']]
        X = frame.drop(columns=['id',request['paths']['target']])
        with np.load(root/request['paths']['split'],allow_pickle=False) as z:
            fit, valid = z['fit'].copy(), z['valid'].copy()
        with np.load(root/request['paths']['control'],allow_pickle=False) as z:
            if not np.array_equal(z['positions'],valid): raise ValueError('Control positions mismatch')
            saved_y = z['y'].copy()
            control = z['prediction'].copy()
            cp = z['probabilities'].copy() if request['kind']=='classification' else None
            classes = z['classes'].astype(str) if cp is not None else None
        target = y.iloc[valid].to_numpy()
        if request['kind']=='regression':
            np.testing.assert_allclose(saved_y,target,rtol=0,atol=1e-10)
            if X.iloc[fit].Year_Factor.max() >= X.iloc[valid].Year_Factor.min():
                raise ValueError('Noncausal regression split')
        else:
            np.testing.assert_array_equal(saved_y.astype(str),target.astype(str))
            np.testing.assert_array_equal(classes[cp.argmax(axis=1)],control.astype(str))
            groups = pd.util.hash_pandas_object(X,index=False).to_numpy()
            if np.intersect1d(groups[fit],groups[valid]).size: raise ValueError('Duplicate-X group leakage')
        if np.intersect1d(fit,valid).size: raise ValueError('Split overlap')
        model = TabMMini(kind=request['kind'], seconds=request['fit_seconds'])
        config = model.configuration()
        atomic_json(out/'configuration.json',config)
        np.savez_compressed(out/'splits.npz',fit=fit,valid=valid)
        claim = memory.claim(config,X.iloc[fit],y.iloc[fit],stage=request['kind']+'-'+request['stage'])
        model.fit(X.iloc[fit],y.iloc[fit])
        raw = model.predict(X.iloc[valid])
        if request['kind']=='regression':
            control = control.astype(float)
            delta = raw-control
            score = lambda p: float(np.sqrt(np.mean((p-target)**2)))
            w = float(np.clip(np.dot(target-control,delta)/max(np.dot(delta,delta),1e-30),0,1))
            weights = [0.,1.,w]
            if request['stage']=='outer': weights = [float(request['frozen_weight'])]
            arms = [(w,score(control+w*delta)) for w in weights]
            selected_weight = min(arms,key=lambda p:p[1])[0]
            prediction = control+selected_weight*delta
            pure_score = score(raw)
            saved = {'raw_prediction':raw,'control':control}
        else:
            np.testing.assert_array_equal(model.classes_.astype(str),classes)
            probabilities = model.predict_proba(X.iloc[valid]).astype(float)
            if not np.isfinite(cp).all() or np.any(cp<0): raise ValueError('Invalid control probabilities')
            np.testing.assert_allclose(cp.sum(1),1,atol=1e-6)
            score = lambda p: float(f1_score(target,p,average='macro',zero_division=0))
            weights = [0.,.1,.25,.5,.75,1.] if request['stage']=='inner' else [float(request['frozen_weight'])]
            arms = [(w,score(classes[((1-w)*cp+w*probabilities).argmax(1)])) for w in weights]
            selected_weight = max(arms,key=lambda p:p[1])[0]
            prediction = classes[((1-selected_weight)*cp+selected_weight*probabilities).argmax(1)]
            pure_score = score(raw)
            saved = {'raw_prediction':raw.astype(str),'control':control.astype(str),
                     'probabilities':probabilities,'control_probabilities':cp,'classes':classes}
        selected_score = score(prediction)
        control_score = score(control)
        improves = selected_score < control_score if request['kind']=='regression' else selected_score > control_score
        result = {'status':'completed','hypothesis':'HYP-108','kind':request['kind'],'stage':request['stage'],
                  'metric':'rmse' if request['kind']=='regression' else 'macro_f1',
                  'score':selected_score,'pure_score':pure_score,'control_score':control_score,
                  'selected_weight':selected_weight,'arm_scores':arms,
                  'eligible_for_outer':bool(improves and selected_weight>0),
                  'fit_rows':len(fit),'valid_rows':len(valid),'new_learner_fits':1,'old_model_refits':0,
                  'seconds':time.monotonic()-started,'fingerprint':claim,'audit':model.audit_}
        np.savez_compressed(out/'validation_predictions.npz',**prediction_arrays(
            {'positions':valid,'y':target,'prediction':prediction,**saved}))
        joblib.dump(model,out/'model.joblib',compress=3)
        atomic_json(out/'audit.json',model.audit_)
        memory.finish(claim,result)
    except Exception as exc:
        result = {'status':'failed','hypothesis':'HYP-108','kind':request['kind'],'stage':request['stage'],
                  'error':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-started}
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        if claim: memory.finish(claim,result)
    atomic_json(out/'metrics.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['status']=='completed' else 1


def main(kind,stage):
    from experimental_tabm import TabMMini
    ledger = (ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    if 'HYP-108 registered' not in ledger: raise RuntimeError('Unregistered H108')
    SUITE.mkdir(parents=True,exist_ok=True)
    registry = SUITE/'attempts.json'
    attempts = read(registry) if registry.exists() else []
    if any(r['kind']==kind and r['stage']==stage for r in attempts): raise RuntimeError('Attempt already exists; no replay')
    frozen_weight = None
    if stage=='outer':
        inner = next(r for r in attempts if r['kind']==kind and r['stage']=='inner')
        outcome = read(ROOT/inner['directory']/'metrics.json')
        if not outcome.get('eligible_for_outer'): raise RuntimeError('Inner gate not passed')
        frozen_weight = outcome['selected_weight']
    number = 1+len(attempts)
    out = ROOT/f'results/2026-10-24/HYP-108-{number:02d}-{kind}-{stage}-tabm-mini'
    out.mkdir(exist_ok=False)
    sources = ['experimental_tabm.py','experiments/tabm_trial.py','research_memory.py',
               'docs/specs/SPEC-015-tabm-trial.md','docs/hypotheses/HYP-108-tabm-mini.md']
    source_hashes = {name:sha(ROOT/name) for name in sources}
    for name in sources:
        destination = out/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    used = paths(kind,stage)
    fit_seconds = min(240, int(900-sum(r.get('seconds',0) for r in attempts)-20))
    if fit_seconds<=0: raise RuntimeError('H108 cumulative budget exhausted')
    request = {'root':str(ROOT),'output':str(out),'history':str(SUITE/'fit-history.json'),
               'kind':kind,'stage':stage,'fit_seconds':fit_seconds,'frozen_weight':frozen_weight,'paths':used,
               'source_sha256':source_hashes,'input_sha256':{p:sha(ROOT/p) for k,p in used.items() if k!='target'}}
    record = {'kind':kind,'stage':stage,'status':'started','started':timestamp(),
              'directory':str(out.relative_to(ROOT)),'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest()}
    attempts.append(record)
    atomic_json(registry,attempts)
    atomic_json(out/'request.json',request)
    atomic_json(out/'evidence.json',record)
    atomic_json(out/'configuration-before-fit.json',TabMMini(kind=kind,seconds=fit_seconds).configuration())
    (out/'report.md').write_text('# HYP-108: '+kind+' '+stage+'\n\nНачало: '+record['started']+
        '\n\nСтатус: started до fit. [SPEC015](../../../docs/specs/SPEC-015-tabm-trial.md). '
        'Standalone NumPy TabM-mini; cached controls, 0 old refits.\n',encoding='utf8')
    env = {k:v for k,v in os.environ.items() if 'API_KEY' not in k and not k.startswith('OPENAI_')}
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf8')
    started = time.monotonic()
    try:
        with (out/'worker.log').open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'experiments/tabm_trial.py'),
                            '--worker',str(out/'request.json')],cwd=ROOT,env=env,stdout=log,
                           stderr=subprocess.STDOUT,timeout=fit_seconds+35,check=False)
        result = read(out/'metrics.json')
    except Exception as exc:
        result = {'status':'timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed',
                  'kind':kind,'stage':stage,'error':type(exc).__name__+': '+str(exc)}
        atomic_json(out/'metrics.json',result)
        memory = Memory(SUITE/'fit-history.json',ROOT/'HYPOTHESES.md')
        for row in memory.records():
            if row['status']=='started': memory.finish(row['fingerprint'],result)
    record.update(status=result['status'],seconds=time.monotonic()-started,finished=timestamp())
    atomic_json(registry,attempts)
    completed = result['status']=='completed'
    conclusion = ('Подтверждена в проверенных условиях' if result.get('eligible_for_outer') else
                  'Не подтверждена по заданному критерию') if completed else 'Недостаточно данных'
    command = f'.venv-ml/Scripts/python.exe -X utf8 -m experiments.tabm_trial --kind {kind} --stage {stage}'
    with (out/'report.md').open('a',encoding='utf8') as f:
        f.write('\nОкончание: '+record['finished']+'\n\nКоманда: `'+command+'`\n\n```json\n'+
                json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\nВывод: '+conclusion+
                '\n\nСреда: Python3.12/Windows, NumPy1.26.4/sklearn1.5.2/pandas2.2.3. '
                'SHA данных/кода: request.json; split: splits.npz; predictions: validation_predictions.npz; '
                'лог: worker.log; model: model.joblib. Нет platform/hidden проверки.\n')
    relative = out.relative_to(ROOT/'results').as_posix()+'/report.md'
    compact = {k:v for k,v in result.items() if k!='audit'}
    for path,prefix in ((ROOT/'HYPOTHESES.md','results/'),(ROOT/'results/README.md',''),(CARD,'../../results/')):
        with path.open('a',encoding='utf8') as f:
            f.write('\nHYP-108 '+kind+'/'+stage+': '+json.dumps(compact,ensure_ascii=False)+
                    '; '+conclusion+'; [report]('+prefix+relative+'). Не повторять.\n')
    print(json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker': raise SystemExit(worker(sys.argv[2]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['regression','classification'],required=True)
    parser.add_argument('--stage',choices=['inner','outer'],default='inner')
    args=parser.parse_args()
    main(args.kind,args.stage)
