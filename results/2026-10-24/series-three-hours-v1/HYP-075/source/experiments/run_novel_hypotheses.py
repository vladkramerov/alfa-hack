"""Sequential new-hypothesis runner with read-before-fit persistent memory."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from automl_data import load_task,split_key
from experiments.novel_models import experiment_plan,semantic_key


ROOT=Path(__file__).resolve().parents[1]
MSK=timezone(timedelta(hours=3))
DATE=datetime.now(MSK).strftime('%Y-%m-%d')
OLD_DATE='2026-10-23'
REG_BEST=43.0046386479306
CLASS_BEST=.5838559593680208
SOURCES=['automl_data.py','automl_models.py','experiments/novel_models.py','experiments/novel_features.py','experiments/novel_worker.py','experiments/run_novel_hypotheses.py']


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp'); temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8'); os.replace(temporary,path)


def now(): return datetime.now(MSK).isoformat()


def claim_attempt(root,candidate,hashes,validation_key):
    root=Path(root)
    ledger=(root/'HYPOTHESES.md').read_text(encoding='utf-8') # Mandatory complete read before EVERY hypothesis.
    if candidate['hypothesis'] not in ledger and not any(p['hypothesis'] in ledger and semantic_key(p)==semantic_key(candidate) for p in experiment_plan()): raise RuntimeError('Hypothesis not registered in HYPOTHESES.md')
    registry=root/'results/spec004-attempts.json'
    previous=json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else []
    key=hashlib.sha256(json.dumps({'data':hashes,'split':validation_key,'candidate':semantic_key(candidate)},sort_keys=True).encode()).hexdigest()
    if any(r['fingerprint']==key or (r['candidate']['hypothesis']==candidate['hypothesis'] and r['candidate']['task']==candidate['task'] and r['hashes']==hashes and r['split_key']==validation_key) for r in previous):
        raise RuntimeError('Already attempted: use saved evidence; seed/name/config changes do not authorize a repeat')
    record={'fingerprint':key,'candidate_key':semantic_key(candidate),'candidate':candidate,'hashes':hashes,'split_key':validation_key,'status':'started','started':now(),
            'ledger_sha256_before_fit':hashlib.sha256(ledger.encode()).hexdigest()}
    previous.append(record); save(registry,previous); return record


def finish_attempt(root,fingerprint,result):
    path=Path(root)/'results/spec004-attempts.json'; rows=json.loads(path.read_text(encoding='utf-8'))
    for row in rows:
        if row['fingerprint']==fingerprint: row.update(result,finished=now())
    save(path,rows)


def write_card(candidate):
    identifier=candidate['hypothesis']; path=ROOT/f'docs/hypotheses/{identifier}.md'
    related=[p for p in experiment_plan() if p['hypothesis']==identifier]
    if path.exists(): return
    text=f'# {identifier}: фиксированная новая проверка\n\nДата:{DATE}. Статус: Запланирована по прямому поручению пользователя. [SPEC-004](../specs/SPEC-004-ten-new-hypotheses.md), [исходное утверждение и риски](HYP-011-020-proposals.md).\n\n'
    text+='Смысл изменения и отличие от истории сохранены в карточке предложения; seed/имя/другой split новизной не считаются.\n\nФиксированные конфигурации до первого fit:\n\n```json\n'+json.dumps(related,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='Данные: исходные публичные SHA и outer indices из series-*-preprocessing-v1; fit только на outer-train. Контрольные результаты не обучаются заново. Критерий практического успеха: RMSE<43.0046386479306 / grouped Macro F1>0.5838559593680208. Предсказания оцениваются в исходной шкале, validation не фильтруется/перевзвешивается. Это разведочная проверка, без независимого подтверждения.\n\nДо первого запуска: Не оценивалась.\n\n## Запуски\n'
    path.write_text(text,encoding='utf-8')


def update_journals(candidate,directory,result):
    identifier=candidate['hypothesis']; rel=directory.relative_to(ROOT).as_posix(); score=result.get('score')
    success=result['status']=='completed'
    meets=success and (score<REG_BEST if candidate['task']=='regression' else score>CLASS_BEST)
    conclusion='Подтверждена в проверенных условиях' if meets else 'Не подтверждена по заданному критерию' if success else 'Недостаточно данных'
    card=ROOT/f'docs/hypotheses/{identifier}.md'
    with card.open('a',encoding='utf-8') as stream: stream.write(f'\n- {now()}: {candidate["task"]}, {result["status"]}, {result.get("metric")}={score}, {conclusion}; [отчёт](../../{rel}/report.md).\n')
    with (ROOT/'HYPOTHESES.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n- SPEC-004 {now()}: **{identifier}/{candidate["task"]}**, {result["status"]}, {result.get("metric")}={score}; {conclusion}; [отчёт]({rel}/report.md). Повтор не запускать.\n')
    with (ROOT/'results/README.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n- SPEC-004 {identifier}/{candidate["task"]}: {result["status"]}, {result.get("metric")}={score}; [отчёт]({directory.relative_to(ROOT/"results").as_posix()}/report.md).\n')
    return conclusion


def main(only=None):
    suite=ROOT/f'results/{DATE}/series-ten-hypotheses-v1'; suite.mkdir(parents=True,exist_ok=True)
    source=suite/'source'
    if source.exists(): raise RuntimeError('Series already started; do not repeat checked configurations')
    source.mkdir(); hashes={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes(); (source/Path(name).name).write_bytes(data); hashes[name]=hashlib.sha256(data).hexdigest()
    save(suite/'source-manifest.json',hashes)
    planned=[p for p in experiment_plan() if only is None or p['hypothesis'] in only]
    save(suite/'frozen-plan.json',{'created':now(),'candidates':planned,'reference_regression_rmse':REG_BEST,'reference_grouped_macro_f1':CLASS_BEST,'source_sha256':hashes})
    for candidate in planned: write_card(candidate)
    records=[]
    for candidate in planned:
        for name,expected in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: raise RuntimeError('Source changed during frozen series: '+name)
        task=candidate['task']; bundle=load_task(ROOT/'data'/task/'task.md')
        original=ROOT/f'results/{OLD_DATE}/series-{task}-preprocessing-v1/splits.npz'
        with np.load(original) as splits: fit,valid=splits['outer_fit'],splits['outer_valid']
        previous=json.loads((ROOT/f'results/{OLD_DATE}/new-family-records.json').read_text(encoding='utf-8'))
        reference=next(r for r in previous if r['task']==task)
        if bundle.hashes!=reference['hashes'] or split_key(fit,valid)!=reference['split_key']: raise ValueError('Data SHA/split differ from saved compatible controls')
        expected={'regression':(35000,18308),'multiclass':(70529,17141)}[task]
        if (len(fit),len(valid))!=expected: raise ValueError('Unexpected fixed split shape')
        if set(fit)&set(valid): raise ValueError('Overlapping split')
        groups=__import__('pandas').util.hash_pandas_object(bundle.X,index=False).to_numpy()
        if task=='multiclass' and set(groups[fit])&set(groups[valid]): raise ValueError('Feature groups cross validation')
        identifier=candidate['hypothesis']; parent=suite.parent; index=1
        while list(parent.glob(f'{identifier}-{index:02d}-*')): index+=1
        directory=parent/f'{identifier}-{index:02d}-{task}-{candidate["transform"]}'
        directory.mkdir(); (directory/'report.md').write_text(f'# {directory.name}\n\nСтатус: подготовлен до fit. [SPEC-004](../../../docs/specs/SPEC-004-ten-new-hypotheses.md).\n',encoding='utf-8')
        record=claim_attempt(ROOT,candidate,bundle.hashes,split_key(fit,valid)); record['directory']=directory.relative_to(ROOT).as_posix()
        np.savez_compressed(directory/'splits.npz',fit=fit,valid=valid)
        save(directory/'source-manifest.json',hashes)
        save(directory/'request.json',{'task_path':str(bundle.task_path),'candidate':candidate,'output':str(directory),'split_path':str(directory/'splits.npz')})
        save(directory/'evidence.json',record)
        env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
        for key in list(env):
            if 'API_KEY' in key: env.pop(key,None)
        started=time.monotonic()
        try:
            with (directory/'worker.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.novel_worker',str(directory/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=False)
            result=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        except Exception as exc:
            result={'status':'failed','error':type(exc).__name__+': '+str(exc),'seconds':time.monotonic()-started}; save(directory/'metrics.json',result)
        result['directory']=record['directory']; result['planned_model_fit_count']=2 if candidate['transform']=='hierarchical' else 1
        finish_attempt(ROOT,record['fingerprint'],result)
        conclusion=update_journals(candidate,directory,result)
        with (directory/'report.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\nНачало:{record["started"]}; завершение:{now()}. Статус:{result["status"]}; {result.get("metric")}={result.get("score")}; {conclusion}.\n\nКоманда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_novel_hypotheses`.\n\n[Конфигурация](request.json), [SHA данных/split и чтение журнала до fit](evidence.json), [хеши стартового снимка](source-manifest.json), [полный snapshot](../series-ten-hypotheses-v1/source/), [метрики](metrics.json), [лог](worker.log).\n\nСохранённые контроли RMSE43.0046386479306 / Macro F1 0.5838559593680208 не переобучались. Медианы/статистики fit только на train; validation исходный и неизменённый. Повторно использованный outer: исследовательский вывод без независимого подтверждения.\n')
        record.update(result,conclusion=conclusion); save(directory/'evidence.json',record); records.append(record); save(suite/'records.json',records)
        print(json.dumps({k:record.get(k) for k in ['candidate','status','score','seconds','conclusion']},ensure_ascii=False),flush=True)
    save(suite/'summary.json',{'status':'completed' if all(r['status']=='completed' for r in records) else 'completed_with_failures','records':records,'baseline_refits':0,'planned_evaluations':len(planned),'finished':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--only',nargs='*'); args=parser.parse_args(); main(args.only)
