"""SPEC013 bounded one-configuration trial; cached controls, no old refits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research_memory import Memory,atomic_json,timestamp


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(request_path):
    import numpy as np
    import pandas as pd
    request=read(request_path);out=Path(request['output']);root=Path(request['root'])
    number=request.get('number',105)
    if number==105:
        from external_multilag import MultiLagRegressor
        model=MultiLagRegressor(threads=4)
    else:
        from external_key import AutoKeyMemory
        model=AutoKeyMemory()
    memory=Memory(request['history'],root/'HYPOTHESES.md');claim=None;started=time.monotonic()
    try:
        for name,digest in request['input_sha256'].items():assert sha(root/name)==digest,name
        data=pd.read_csv(root/'data/regression/train.csv')
        X=data.drop(columns=['id','site_eui']);y=data.site_eui
        with np.load(root/request['split'],allow_pickle=False) as z:fit=z['fit'];valid=z['valid']
        with np.load(root/request['control'],allow_pickle=False) as z:
            assert np.array_equal(z['positions'],valid)
            np.testing.assert_allclose(y.iloc[valid],z['y'],rtol=0,atol=1e-10)
            control=z['prediction'].copy()
        assert not np.intersect1d(fit,valid).size
        configuration=model.configuration()
        atomic_json(out/'configuration.json',configuration)
        np.savez_compressed(out/'splits.npz',fit=fit,valid=valid)
        claim=memory.claim(configuration,X.iloc[fit],y.iloc[fit],request['stage'])
        model.fit(X.iloc[fit],y.iloc[fit])
        pred=model.predict(X.iloc[valid]) if number==105 else model.predict_with_fallback(X.iloc[valid],control)
        target=y.iloc[valid].to_numpy()
        candidates={'pure':pred,'blend75':.75*pred+.25*control}
        scores={name:float(np.sqrt(np.mean((p-target)**2))) for name,p in candidates.items()}
        arm=min(scores,key=scores.get) if request['stage']=='inner' else request['selected_arm']
        score=scores[arm]
        saved={'positions':valid,'fit':fit,'y':target,'prediction':candidates[arm],'raw_prediction':pred,
               'control':control,'pure':candidates['pure'],'blend75':candidates['blend75']}
        np.savez_compressed(out/'validation_predictions.npz',**saved)
        atomic_json(out/'audit.json',model.audit_)
        import joblib
        joblib.dump(model,out/'model.joblib',compress=3)
        result={'status':'completed','hypothesis':f'HYP-{number:03d}','stage':request['stage'],'metric':'rmse',
            'score':score,'arm_scores':scores,'selected_arm':arm,'control_score':float(np.sqrt(np.mean((control-target)**2))),
            'selected_on':'inner_only','learner_fits':int(number==105),'parameter_fits':int(number==106),'old_model_refits':0,'fit_rows':len(fit),'valid_rows':len(valid),
            'eligible_for_outer':score<30.636363541581975 if request['stage']=='inner' else None,
            'seconds':time.monotonic()-started,'fingerprint':claim}
        memory.finish(claim,result)
    except Exception as exc:
        result={'status':'failed','hypothesis':f'HYP-{number:03d}','stage':request['stage'],
                'error':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-started}
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        if claim:memory.finish(claim,result)
    atomic_json(out/'metrics.json',result)
    return 0 if result['status']=='completed' else 1


def main(stage,number=105):
    ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    (ROOT/'tmp/sergey-24e3311f6ba9/HYPOTHESES.md').read_text(encoding='utf8')
    assert f'HYP-{number:03d}' in ledger
    suite=ROOT/('results/2026-10-24/series-external-ideas-v1' if number==105 else 'results/2026-10-24/series-external-key-v1');suite.mkdir(exist_ok=True)
    registry=suite/'attempts.json';attempts=read(registry) if registry.exists() else []
    assert not any(r['stage']==stage for r in attempts),'No replays'
    sources=['external_multilag.py','research_memory.py','experiments/external_ideas_run.py']
    if number==106:sources+=['external_key.py']
    frozen=suite/'frozen.json'
    if not frozen.exists():
        hashes={name:sha(ROOT/name) for name in sources}
        for name in sources:
            dest=suite/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        atomic_json(frozen,{'created':timestamp(),'sources':hashes,'spec':'SPEC013','hypothesis':f'HYP-{number:03d}'})
    for name,digest in read(frozen)['sources'].items():assert sha(suite/'source'/name)==digest
    sergey='tmp/sergey-24e3311f6ba9/results/2026-10-24/'+('HYP-073-02-regression-panel-secondary' if stage=='inner' else 'HYP-073-01-regression-panel-primary')
    split=sergey+'/splits.npz';control=sergey+'/validation_predictions.npz'
    selected=None
    if stage=='outer':
        inner=next(r for r in attempts if r['stage']=='inner');selected_result=read(ROOT/inner['directory']/'metrics.json')
        assert selected_result['status']=='completed' and selected_result['eligible_for_outer']
        selected=selected_result['selected_arm']
    slug='external-multilag' if number==105 else 'external-key'
    out=ROOT/f'results/2026-10-24/HYP-{number:03d}-{1 if stage=="inner" else 2:02d}-{stage}-{slug}'
    out.mkdir(exist_ok=False)
    request={'root':str(ROOT),'output':str(out),'number':number,'stage':stage,'split':split,'control':control,
        'history':str(suite/'fit-history.json'),'selected_arm':selected,
        'input_sha256':{name:sha(ROOT/name) for name in [split,control,'data/regression/train.csv','data/regression/test.csv']}}
    record={'hypothesis':f'HYP-{number:03d}','stage':stage,'status':'started','started':timestamp(),
        'directory':str(out.relative_to(ROOT)),'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),
        'frozen_source_sha256':sha(frozen),'request':request}
    attempts.append(record);atomic_json(registry,attempts);atomic_json(out/'evidence.json',record)
    atomic_json(out/'request.json',request)
    (out/'report.md').write_text(f'# HYP-{number:03d} '+stage+'\n\nStarted before fit. SPEC013. Saved Sergey controls; no old refits.\n',encoding='utf8')
    remaining=900-sum(r.get('seconds',0) for r in attempts[:-1]);started=time.monotonic()
    env={k:v for k,v in os.environ.items() if 'API_KEY' not in k and not k.startswith('OPENAI_')}
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf8')
    try:
        if remaining<=0:raise TimeoutError('Cumulative900sec budget spent')
        with (out/'worker.log').open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-X','utf8','-u',str(suite/'source/experiments/external_ideas_run.py'),
                '--worker',str(out/'request.json')],env=env,cwd=suite/'source',stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
        result=read(out/'metrics.json')
    except Exception as exc:
        result={'status':'timeout' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'failed',
                'hypothesis':f'HYP-{number:03d}','stage':stage,'error':type(exc).__name__+': '+str(exc)}
        atomic_json(out/'metrics.json',result)
        memory=Memory(suite/'fit-history.json',ROOT/'HYPOTHESES.md')
        for row in memory.records():
            if row['status']=='started':memory.finish(row['fingerprint'],result)
    record.update(status=result['status'],seconds=time.monotonic()-started)
    atomic_json(registry,attempts);atomic_json(out/'outcome.json',result)
    with (out/'report.md').open('a',encoding='utf8') as f:
        f.write('\n'+json.dumps(result,ensure_ascii=False,indent=2)+f'\n\nCommand: `.venv-ml/Scripts/python.exe -X utf8 -m experiments.external_ideas_run --number {number} --stage '+stage+'`.\n')
    card='HYP-105-multilag-native-boosting.md' if number==105 else 'HYP-106-training-key-discovery.md'
    for path,prefix in [('HYPOTHESES.md','results/'),('results/README.md',''),('docs/hypotheses/'+card,'../../results/')]:
        link=prefix+str(out.relative_to(ROOT/'results')).replace('\\','/')+'/report.md'
        with (ROOT/path).open('a',encoding='utf8') as f:f.write(f'\nHYP-{number:03d}/'+stage+': '+json.dumps(result,ensure_ascii=False)+'; [report]('+link+'). Не повторять.\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':raise SystemExit(worker(sys.argv[2]))
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['inner','outer'],default='inner');parser.add_argument('--number',type=int,choices=[105,106],default=105)
    args=parser.parse_args();main(args.stage,args.number)
