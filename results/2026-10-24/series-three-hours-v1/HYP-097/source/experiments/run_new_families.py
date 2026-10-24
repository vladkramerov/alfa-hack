"""HYP-009/010: four new fits, reuse saved splits; never refit controls."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
from automl_data import load_task,split_key,validation_split
from automl_history import candidate_key
from solution import worker,save


ROOT=Path(__file__).resolve().parents[1]


def main():
    records=[]
    for task in ['regression','multiclass']:
        bundle=load_task(ROOT/'data'/task/'task.md')
        saved=np.load(ROOT/f'results/2026-10-23/series-{task}-preprocessing-v1/splits.npz')
        a,b=saved['outer_fit'],saved['outer_valid']
        actual_a,actual_b,_=validation_split(bundle)
        if split_key(a,b)!=split_key(actual_a,actual_b): raise ValueError('Runtime split differs from historical protocol')
        for number,family in [(9,'lightgbm'),(10,'extra_trees')]:
            parent=ROOT/'results/2026-10-23'
            index=1
            while list(parent.glob(f'HYP-{number:03d}-{index:02d}-*')): index+=1
            directory=parent/f'HYP-{number:03d}-{index:02d}-{task}-{family}'
            if list(parent.glob(f'HYP-{number:03d}-*-{task}-{family}')): raise RuntimeError('Already attempted; inspect history instead of repeating')
            staging=parent/f'new-split-{task}.npz'
            np.savez_compressed(staging,fit=a,valid=b)
            candidate={'family':family,'parameters':{},'hypothesis':f'HYP-{number:03d}: new model family'}
            result=worker(bundle,candidate,'evaluate',directory,staging,1800)
            source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['automl_models.py','automl_data.py','automl_worker.py','experiments/run_new_families.py']}
            record={'task':task,'candidate':candidate,'candidate_key':candidate_key(candidate),'hashes':bundle.hashes,'split_key':split_key(a,b),'python':sys.version,'platform':platform.platform(),'code_sha256':source,**result,'directory':str(directory.relative_to(ROOT))}
            save(directory/'evidence.json',record)
            text=f'# HYP-{number:03d}: {task} / {family}\n\n[Спецификация](../../../docs/specs/SPEC-003-platform-agent.md), [гипотеза](../../../docs/hypotheses/HYP-{number:03d}.md).\n\nНовая модель; известные контрольные конфигурации повторно не обучались. Разбиение побайтно совпало с сохранённым outer. Предобработка: train-only median, native categories (LightGBM) / one-hot (ExtraTrees).\n\nКоманда: `.venv-ml/Scripts/python.exe -m experiments.run_new_families`.\n\nСтатус: {result["status"]}; метрика: {result.get("metric")}; значение: {result.get("score")}; секунд: {result.get("seconds")}.\n\n[Хеши данных/кода, окружение и конфигурация](evidence.json), [метрика](metrics.json), [лог](worker.log). Повторно используемый validation: исследовательский выбор без независимого подтверждения. Вывод и сравнение: общий NEW_EXPERIMENT_REPORT.md.\n'
            (directory/'report.md').write_text(text,encoding='utf-8')
            records.append(record); save(parent/'new-family-records.json',records)
            print(json.dumps({k:record.get(k) for k in ['task','candidate','status','score','seconds']}),flush=True)
            if result['status']!='completed': raise RuntimeError('New fit failed; preserve evidence and diagnose before continuing')


if __name__=='__main__': main()
