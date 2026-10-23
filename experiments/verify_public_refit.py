"""Final full-training checks only; never recompute known validation scores."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import pandas as pd
from automl_data import load_task,validate_submission

ROOT=Path(__file__).resolve().parents[1]


def main():
    directory=ROOT/'results/2026-10-23'/('public-refit-verification-'+str(time.time_ns())); directory.mkdir()
    extracted=ROOT/'tmp'/('public-flat-'+str(time.time_ns())); extracted.mkdir(parents=True)
    with zipfile.ZipFile(ROOT/'dist/solution_ivan_branch.zip') as archive: archive.extractall(extracted)
    records=[]
    for task in ['regression','multiclass']:
        bundle=load_task(ROOT/'data'/task/'task.md')
        env=os.environ.copy(); env.update(AUTOML_MAX_EXPERIMENTS='0',AUTOML_TIME_LIMIT='900',PYTHONIOENCODING='utf-8')
        start=time.monotonic()
        process=subprocess.run([sys.executable,str(extracted/'solution.py'),str(bundle.task_path)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=900)
        (directory/(task+'.log')).write_text(process.stdout,encoding='utf-8')
        assert process.returncode==0,process.stdout
        summary_path=sorted((bundle.task_path.parent/'agent_runs').glob('*/summary.json'))[-1]
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        assert summary['cached_history'] and summary['selected_score'] is not None
        assert not summary['experiments'],'Unexpected repeated validation fit'
        refits=json.loads((summary_path.parent/'initial_refit.json').read_text(encoding='utf-8'))
        assert len(refits)==1 and refits[0]['result']['status']=='completed',refits
        frame=pd.read_csv(bundle.task_path.parent/'submission.csv'); validate_submission(bundle,frame)
        record={'task':task,'selected_family':summary['selected']['family'],'cached_validation_metric':summary['metric'],'cached_validation_score':summary['selected_score'],
                'validation_fit_count':0,'full_refit_count':1,'rows':len(frame),'columns':list(frame),'seconds':time.monotonic()-start,
                'submission_sha256':hashlib.sha256((bundle.task_path.parent/'submission.csv').read_bytes()).hexdigest(),'summary':summary,'refits':refits}
        (directory/(task+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8'); records.append(record)
        print(json.dumps({k:record[k] for k in ['task','selected_family','cached_validation_score','rows','seconds']}),flush=True)
    (directory/'verification.json').write_text(json.dumps({'status':'passed','records':records},indent=2,ensure_ascii=False),encoding='utf-8')
    (directory/'report.md').write_text('# Полное обучение выбранных моделей\n\nЗапуск из чистой распаковки ZIP. Для обоих публичных датасетов: 0 validation fit, 1 full train fit; использованы сохранённые совместимые метрики. Submission проверен: размеры, имена столбцов, ID/порядок, допустимые значения. Файлы submission.csv сохранены рядом с исходными task.md (data/ не публикуется в Git). Реальный LLM API не вызывался: конфигурация отсутствует, экспериментальный бюджет установлен в0.\n',encoding='utf-8')


if __name__=='__main__': main()
