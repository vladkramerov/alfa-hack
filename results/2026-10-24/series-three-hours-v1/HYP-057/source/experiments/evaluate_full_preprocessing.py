"""HYP-021: two new combined full-v1 evaluations, no previous baseline fits."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from automl_data import load_task,split_key
from automl_models import canonical_candidate
from automl_history import candidate_key


ROOT=Path(__file__).resolve().parents[1]
DATE=datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d')
SOURCES=['automl_models.py','automl_preprocessing.py','automl_worker.py','automl_data.py','automl_history.py','experiments/evaluate_full_preprocessing.py']


def save(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp'); temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); os.replace(temporary,path)


def main():
    parent=ROOT/'results'/DATE; parent.mkdir(parents=True,exist_ok=True)
    registry_path=ROOT/'results/spec005-attempts.json'
    previous=json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else []
    records=[]
    for task,family,transform,reference in [('regression','extra_trees','log1p',42.19670002719412),('multiclass','xgboost',None,.5914513709191194)]:
        ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf-8')
        if 'HYP-021' not in ledger: raise RuntimeError('Hypothesis not registered before fit')
        bundle=load_task(ROOT/'data'/task/'task.md')
        with np.load(ROOT/f'results/2026-10-23/series-{task}-preprocessing-v1/splits.npz') as split: fit,valid=split['outer_fit'],split['outer_valid']
        old=json.loads((ROOT/'results/2026-10-23/new-family-records.json').read_text(encoding='utf-8'))
        control=next(r for r in old if r['task']==task)
        if bundle.hashes!=control['hashes'] or split_key(fit,valid)!=control['split_key']: raise ValueError('Inputs/splits changed')
        candidate=canonical_candidate({'family':family,'parameters':{},'target_transform':transform,'hypothesis':'HYP-021: mandatory full-v1 normalization, bounded normal quantiles and <=5% training outliers'})
        identity=candidate_key(candidate)
        if any(r['task']==task or r['candidate_key']==identity for r in previous): raise RuntimeError('HYP-021 already attempted; no repeated fits')
        index=1
        while list(parent.glob(f'HYP-021-{index:02d}-*')): index+=1
        directory=parent/f'HYP-021-{index:02d}-{task}-full-v1'; directory.mkdir()
        source=directory/'source'; source.mkdir(); hashes={}
        for name in SOURCES:
            data=(ROOT/name).read_bytes(); (source/Path(name).name).write_bytes(data); hashes[name]=hashlib.sha256(data).hexdigest()
        np.savez_compressed(directory/'splits.npz',fit=fit,valid=valid)
        request={'task_path':str(bundle.task_path),'candidate':candidate,'mode':'evaluate','output':str(directory),'split_path':str(directory/'splits.npz'),'threads':4}
        save(directory/'request.json',request)
        record={'task':task,'candidate':candidate,'candidate_key':identity,'reference':reference,'hashes':bundle.hashes,'split_key':split_key(fit,valid),
                'ledger_sha256_before_fit':hashlib.sha256(ledger.encode()).hexdigest(),'code_sha256':hashes,'directory':directory.relative_to(ROOT).as_posix(),
                'started':datetime.now(timezone(timedelta(hours=3))).isoformat(),'status':'started'}
        previous.append(record); save(registry_path,previous); save(directory/'evidence.json',record)
        (directory/'report.md').write_text('# HYP-021 '+task+'\n\nСтатус: started до первого fit; исходная конфигурация и SHA записаны. [SPEC-005](../../../docs/specs/SPEC-005-agent-preprocessing.md).\n',encoding='utf-8')
        env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
        for key in list(env):
            if 'API_KEY' in key: env.pop(key,None)
        start=time.monotonic()
        try:
            with (directory/'worker.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-X','utf8',str(ROOT/'automl_worker.py'),str(directory/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=False)
            result=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        except Exception as exc: result={'status':'failed','error':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-start}; save(directory/'metrics.json',result)
        record.update(result,finished=datetime.now(timezone(timedelta(hours=3))).isoformat())
        improved=result['status']=='completed' and (result['score']<reference if task=='regression' else result['score']>reference)
        record['conclusion']='Подтверждена в проверенных условиях' if improved else 'Не подтверждена по заданному критерию' if result['status']=='completed' else 'Недостаточно данных'
        save(directory/'evidence.json',record); save(registry_path,previous); records.append(record); save(parent/'full-preprocessing-records.json',records)
        text=f'\nЗавершение:{record["finished"]}; {result["status"]}; {result.get("metric")}={result.get("score")}; {record["conclusion"]}.\n\nКоманда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.evaluate_full_preprocessing`. Контроль{reference} переиспользован без обучения. Validation сохраняет все строки; составной эффект без отдельного приписывания нормализации или выбросам. [Данные/код/split](evidence.json), [аудит](preprocessing-audit.json), [снимок кода](source/), [метрики](metrics.json), [лог](worker.log).\n'
        with (directory/'report.md').open('a',encoding='utf-8') as stream: stream.write(text)
        for file in [ROOT/'HYPOTHESES.md',ROOT/'docs/hypotheses/HYP-021.md']:
            link=record['directory']+'/report.md' if file.name=='HYPOTHESES.md' else '../../'+record['directory']+'/report.md'
            with file.open('a',encoding='utf-8') as stream: stream.write(f'\n- HYP-021/{task}: {result["status"]}, {result.get("metric")}={result.get("score")}; {record["conclusion"]}; [отчёт]({link}). Повтор не запускать.\n')
        print(json.dumps({k:record.get(k) for k in ['task','status','score','seconds','fit_retained_rows','conclusion']},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
