"""Explicitly authorized one-time measurements of the actual previous ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'tmp/torch-tabm-baseline-v1'
ARCHIVE = ROOT/'dist/solution_sergey_ivan_fusion_v1.zip'
SUITE = ROOT/'results/2026-10-24/series-torch-tabm-v1'
EXPECTED = '9a3924a53f6bf6191d27dd0717c44afc4137757557e469efe77c3eac28d1756a'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def worker(request_path):
    request = read(request_path)
    sys.path.insert(0,str(BASE))
    sys.path.insert(1,str(ROOT))
    from research_memory import timestamp
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import f1_score
    import automl_models
    assert Path(automl_models.__file__).parent==BASE
    out=Path(request['output']);started=time.monotonic()
    try:
        assert sha(ARCHIVE)==EXPECTED
        ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
        assert 'HYP-109 registered' in ledger
        for name,digest in request['input_sha256'].items(): assert sha(ROOT/name)==digest,name
        kind=request['kind']
        target='site_eui' if kind=='regression' else 'attack_cat'
        frame=pd.read_csv(ROOT/request['data']);y=frame[target];X=frame.drop(columns=['id',target])
        with np.load(ROOT/request['split'],allow_pickle=False) as z:fit=z['fit'];valid=z['valid']
        assert not np.intersect1d(fit,valid).size
        if kind=='classification':
            groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
            assert not np.intersect1d(groups[fit],groups[valid]).size
        else: assert X.iloc[fit].Year_Factor.max()<X.iloc[valid].Year_Factor.min()
        candidate={'family':'extra_trees','strategy':'sergey_ivan_fusion','parameters':{}} if kind=='regression' else {'family':'xgboost','strategy':'probability_stacking','parameters':{}}
        write(out/'configuration.json',candidate)
        write(out/'fit-started.json',{'status':'started','started':timestamp(),'role':'explicit_previous_zip_control',
              'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'rows':len(fit),'candidate':candidate})
        os.environ.update(COMBO_RUN_DIR=str(out/'fits'),COMBO_LEDGER_PATH=str(ROOT/'HYPOTHESES.md'),
            COMBO_HYPOTHESIS='SPEC016-explicit-ZIP-control-'+kind+'-'+request['stage'])
        model=automl_models.build_pipeline(kind,candidate,4).fit(X.iloc[fit],y.iloc[fit])
        prediction=model.predict(X.iloc[valid])
        arrays={'positions':valid,'fit':fit,'y':y.iloc[valid].to_numpy(),'prediction':prediction}
        if kind=='classification':
            arrays.update(probabilities=model.predict_proba(X.iloc[valid]),classes=model.classes_)
            score=float(f1_score(y.iloc[valid],prediction,average='macro',zero_division=0))
        else: score=float(np.sqrt(np.mean((prediction-y.iloc[valid].to_numpy())**2)))
        arrays={k:(np.asarray(v).astype(str) if np.asarray(v).dtype.kind=='O' else np.asarray(v)) for k,v in arrays.items()}
        np.savez_compressed(out/'validation_predictions.npz',**arrays)
        np.savez_compressed(out/'splits.npz',fit=fit,valid=valid)
        joblib.dump(model,out/'model.joblib',compress=3)
        write(out/'audit.json',model.audit_)
        result={'status':'completed','kind':kind,'stage':request['stage'],'role':'explicit_previous_zip_control',
                'metric':'rmse' if kind=='regression' else 'macro_f1','score':score,'fit_rows':len(fit),
                'valid_rows':len(valid),'seconds':time.monotonic()-started,'zip_sha256':EXPECTED,
                'actual_archived_factory':True,'previous_quality_trials_replayed':False}
    except Exception as exc:
        result={'status':'failed','kind':request['kind'],'stage':request['stage'],'error':repr(exc),'seconds':time.monotonic()-started}
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
    write(out/'metrics.json',result)
    return 0 if result['status']=='completed' else 1


def main(kind,stage):
    sys.path.insert(0,str(ROOT))
    from research_memory import timestamp
    assert sha(ARCHIVE)==EXPECTED
    if not BASE.exists():
        BASE.mkdir(parents=True)
        with zipfile.ZipFile(ARCHIVE) as archive:
            for name in archive.namelist():
                assert '/' not in name and '\\' not in name
                (BASE/name).write_bytes(archive.read(name))
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist(): assert (BASE/name).read_bytes()==archive.read(name),name
    ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    assert 'HYP-109 registered' in ledger
    prefix='HYP-070-03-inner-high-sprint' if stage=='inner' else 'HYP-070-04-outer-high-sprint'
    if kind=='classification':prefix=f'HYP-078-{1 if stage=="inner" else 2:02d}-{stage}-default-sprint'
    split='results/2026-10-24/'+prefix+'/splits.npz'
    data='data/'+('regression' if kind=='regression' else 'multiclass')+'/train.csv'
    out=ROOT/f'results/2026-10-24/ZIP-CONTROL-{kind}-{stage}-torch-v1'
    out.mkdir(exist_ok=False)
    request={'kind':kind,'stage':stage,'output':str(out),'split':split,'data':data,'zip_sha256':EXPECTED,
             'started':timestamp(),'input_sha256':{p:sha(ROOT/p) for p in [split,data]},
             'source_sha256':{p.name:sha(p) for p in BASE.iterdir() if p.is_file()},'runner_sha256':sha(__file__)}
    write(out/'request.json',request)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'report.md').write_text('# Previous ZIP control '+kind+'/'+stage+'\n\nStarted before fit: '+request['started']+
        '\n\nSPEC016 REQ001: explicitly authorized actual ZIP comparison; not a replay of cached H107/H108.\n',encoding='utf8')
    env={k:v for k,v in os.environ.items() if 'API_KEY' not in k and not k.startswith('OPENAI_')}
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONIOENCODING='utf8')
    try:
        with (out/'worker.log').open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-X','utf8','-u',__file__,'--worker',str(out/'request.json')],
                cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=False)
        result=read(out/'metrics.json')
    except Exception as exc:
        result={'status':'timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed','kind':kind,'stage':stage,'error':repr(exc)}
        write(out/'metrics.json',result)
    with (out/'report.md').open('a',encoding='utf8',newline='\n') as f:
        f.write('\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\nCommand: `.venv-ml/Scripts/python.exe -X utf8 -m experiments.torch_zip_control --kind '+kind+' --stage '+stage+'`.\n')
    relative=out.relative_to(ROOT).as_posix()+'/report.md'
    for path,prefix in [(ROOT/'HYPOTHESES.md',''),(ROOT/'results/README.md','../')]:
        with path.open('a',encoding='utf8',newline='\n') as f:f.write('\nSPEC016 explicit ZIP control '+kind+'/'+stage+': '+json.dumps(result,ensure_ascii=False)+'; [report]('+prefix+relative+'). Не повторять.\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':raise SystemExit(worker(sys.argv[2]))
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['regression','classification'],required=True)
    parser.add_argument('--stage',choices=['inner','outer'],default='inner')
    args=parser.parse_args();main(args.kind,args.stage)
