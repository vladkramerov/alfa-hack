"""SPEC-006: 23 frozen evaluations for twenty methods; safe resume, no retries."""
import argparse
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
from experiments.twenty_common import feature_identity

ROOT=Path(__file__).resolve().parents[1]
DATE='2026-10-23'
SUITE=ROOT/f'results/{DATE}/series-twenty-hypotheses-v1'
REGISTRY=ROOT/'results/spec006-attempts.json'
REFERENCES={'regression':42.19670002719412,'multiclass':.5914513709191194}
SOURCES=['automl_data.py','automl_models.py','automl_preprocessing.py','experiments/novel_models.py',
 'experiments/twenty_features.py','experiments/twenty_common.py','experiments/twenty_advanced.py',
 'experiments/twenty_worker.py','experiments/run_twenty_hypotheses.py']

def now(): return datetime.now(timezone(timedelta(hours=3))).isoformat()
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(temporary,path)

def experiment_plan():
    jobs=[]
    for n in range(22,42):
        tasks=['regression','multiclass'] if n in [22,28,41] else ['multiclass'] if n in [25,26,27,37,38,39,40] else ['regression']
        for task in tasks:
            jobs.append({'number':n,'hypothesis':f'HYP-{n:03d}','task':task,'version':'spec006-v1',
                         'policy':'gaussian' if n==22 else 'light','reference':REFERENCES[task],
                         'method':'features' if n in [22,23,24,25,26,27,29,30,31,32] else 'advanced',
                         'configuration_source':f'experiments/twenty_{"features" if n in [22,23,24,25,26,27,29,30,31,32] else "advanced"}.py'})
    return jobs

def load_rows(): return json.loads(REGISTRY.read_text(encoding='utf-8')) if REGISTRY.exists() else []

def main(only=None):
    SUITE.mkdir(parents=True,exist_ok=True)
    jobs=experiment_plan()
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    frozen_path=SUITE/'frozen-plan.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
        if frozen['candidates']!=jobs or frozen['sources']!=sources:
            raise RuntimeError('Frozen source/config changed; do not silently repeat experiments')
    else:
        save(frozen_path,{'created':now(),'candidates':jobs,'sources':sources,'baseline_refits':0})
        for name in SOURCES:
            dst=SUITE/'source'/name; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes((ROOT/name).read_bytes())
    for job in jobs:
        if only and job['number'] not in only: continue
        rows=load_rows()
        prior=[r for r in rows if r['candidate']['number']==job['number'] and r['candidate']['task']==job['task']]
        if prior:
            print(json.dumps({'skip_attempted':job['hypothesis'],'task':job['task'],'status':prior[-1]['status']}),flush=True); continue
        ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf-8')
        if job['hypothesis'] not in ledger: raise RuntimeError('Unregistered hypothesis')
        bundle=load_task(ROOT/'data'/job['task']/'task.md')
        with np.load(ROOT/f'results/{DATE}/series-{job["task"]}-preprocessing-v1/splits.npz') as split:
            fit,valid=split['outer_fit'],split['outer_valid']
        old=json.loads((ROOT/f'results/{DATE}/new-family-records.json').read_text(encoding='utf-8'))
        control=next(r for r in old if r['task']==job['task'])
        if bundle.hashes!=control['hashes'] or split_key(fit,valid)!=control['split_key']: raise ValueError('Changed data/split')
        identity={'candidate':job,'hashes':bundle.hashes,'split_key':split_key(fit,valid)}
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if any(r['fingerprint']==fingerprint for r in rows): raise RuntimeError('Repeated attempt')
        index=1
        while list(SUITE.parent.glob(f'{job["hypothesis"]}-{index:02d}-*')): index+=1
        directory=SUITE.parent/f'{job["hypothesis"]}-{index:02d}-{job["task"]}-spec006'; directory.mkdir()
        np.savez_compressed(directory/'splits.npz',fit=fit,valid=valid)
        artifact_map={f'{r["candidate"]["number"]}:{r["candidate"]["task"]}':str(ROOT/r['directory']/'validation_predictions.npz') for r in rows if r['status']=='completed'}
        request={'candidate':job,'task_path':str(bundle.task_path),'output':str(directory),'split_path':str(directory/'splits.npz'),
                 'cache_dir':str(SUITE/'cache'),'candidate_artifacts':artifact_map}
        save(directory/'request.json',request)
        record={**identity,'candidate':job,'fingerprint':fingerprint,'directory':directory.relative_to(ROOT).as_posix(),
                'status':'started','started':now(),'sources':sources,'ledger_sha256_before_fit':hashlib.sha256(ledger.encode()).hexdigest()}
        rows.append(record); save(REGISTRY,rows); save(directory/'evidence.json',record)
        card=ROOT/f'docs/hypotheses/{job["hypothesis"]}.md'
        if not card.exists():
            card.write_text(f'# {job["hypothesis"]}\n\n[SPEC-006](../specs/SPEC-006-twenty-hypotheses.md), [метод/новизна/риски](HYP-022-041-proposals.md). Пользователь поручил проверку всех20.\n\nПлан до fit:\n```json\n'+json.dumps([j for j in jobs if j['number']==job['number']],indent=2,ensure_ascii=False)+'\n```\n\nСравнимые пороги: RMSE<42.19670002719412 / grouped Macro-F1>0.5914513709191194. Старыеbaselineнепереобучать; источникконкретныхпараметров заморожен вsource/ серии.\n\n## Запуски\n',encoding='utf-8')
        (directory/'report.md').write_text(f'# {job["hypothesis"]} / {job["task"]}\n\nСтатус started до первого fit. [SPEC-006](../../../docs/specs/SPEC-006-twenty-hypotheses.md).\n\nКоманда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_twenty_hypotheses`.\n\nДанные/split/код: [evidence](evidence.json), [конфигурация](request.json), [снимок кода](../series-twenty-hypotheses-v1/source/).\n',encoding='utf-8')
        env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',
            SPEC006_RUN_DIR=str(SUITE),SPEC006_HYPOTHESIS=job['hypothesis']+'/'+job['task'],SPEC006_ENFORCE_BASELINE_GUARD='1')
        for key in list(env):
            if 'API_KEY' in key: env.pop(key,None)
        start=time.monotonic()
        try:
            with (directory/'worker.log').open('w',encoding='utf-8') as log:
                process=subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.twenty_worker',str(directory/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=False)
            result=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        except Exception as exc:
            result={'status':'timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed','error':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-start}
            save(directory/'metrics.json',result)
        succeeded=result['status']=='completed'
        improved=succeeded and (result['score']<REFERENCES[job['task']] if job['task']=='regression' else result['score']>REFERENCES[job['task']])
        conclusion='Подтверждена в проверенных условиях' if improved else 'Не подтверждена по заданному критерию' if succeeded else 'Недостаточно данных'
        record.update(result,conclusion=conclusion,finished=now()); rows[-1]=record
        save(directory/'evidence.json',record); save(REGISTRY,rows); save(SUITE/'records.json',rows)
        rel=record['directory']; detail=f'{result["status"]}, {result.get("metric")}={result.get("score")}; {conclusion}'
        with (directory/'report.md').open('a',encoding='utf-8') as f:
            f.write(f'\nИтог {record["finished"]}: **{detail}**.\n\nПрежний лучший {REFERENCES[job["task"]]} переиспользован безbaselinefit. [Метрики](metrics.json), [аудит](preprocessing-audit.json), [лог](worker.log), [predictions](validation_predictions.npz).\n\nОграничения: повторно использованный outer, без независимогоtest; локальнаяметрика не scoreплатформы.\n')
        with card.open('a',encoding='utf-8') as f: f.write(f'\n- {job["task"]}: {detail}; [отчёт](../../{rel}/report.md). Повтор не запускать.\n')
        with (ROOT/'HYPOTHESES.md').open('a',encoding='utf-8') as f: f.write(f'\n- SPEC-006 {record["finished"]}: **{job["hypothesis"]}/{job["task"]}**: {detail}; [отчёт]({rel}/report.md). Проверка завершена; повтор не запускать.\n')
        with (ROOT/'results/README.md').open('a',encoding='utf-8') as f: f.write(f'\n- SPEC-006 {job["hypothesis"]}/{job["task"]}: {detail}; [отчёт]({rel.removeprefix("results/")}/report.md).\n')
        print(json.dumps({'hypothesis':job['hypothesis'],'task':job['task'],**result,'conclusion':conclusion},ensure_ascii=False),flush=True)
    rows=load_rows()
    save(SUITE/'summary.json',{'planned_evaluations':len(jobs),'attempted':len(rows),'completed':sum(r['status']=='completed' for r in rows),
        'unique_hypotheses':len({r['candidate']['number'] for r in rows}),'baseline_refits':0,'finished':now()})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--only',nargs='*',type=int); args=parser.parse_args(); main(args.only)
