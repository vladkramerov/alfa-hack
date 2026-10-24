"""Two public full refits from clean ZIP; zero repeated validation fits."""
import hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from automl_data import load_task,validate_submission

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'regression':('group_experts',41.480963027907194,22449),'multiclass':('macro_f1_calibrated',.6286047899141216,41166)}

def main():
    archive=ROOT/'dist/solution_improved_metrics.zip'
    parent=ROOT/'results/2026-10-23'/('improved-delivery-'+str(time.time_ns())); parent.mkdir()
    extracted=ROOT/'tmp'/('improved-flat-'+str(time.time_ns())); extracted.mkdir(parents=True)
    with zipfile.ZipFile(archive) as package: package.extractall(extracted)
    manifest=json.loads((ROOT/'results/2026-10-23/package-manifest-improved.json').read_text(encoding='utf-8'))
    assert manifest['sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
    records=[]
    for task,(strategy,measured,nrows) in EXPECTED.items():
        bundle=load_task(ROOT/'data'/task/'task.md')
        env=os.environ.copy(); env.update(AUTOML_MAX_EXPERIMENTS='0',AUTOML_PORTFOLIO='0',AUTOML_TIME_LIMIT='900',PYTHONIOENCODING='utf-8')
        env.pop('PYTHONPATH',None)
        for key in list(env):
            if key.startswith('OPENAI_') or key.startswith('SPEC006_'): env.pop(key,None)
        started=time.monotonic()
        p=subprocess.run([sys.executable,'-X','utf8',str(extracted/'solution.py'),str(bundle.task_path)],cwd=extracted,env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=900)
        (parent/(task+'.log')).write_text(p.stdout,encoding='utf-8')
        assert p.returncode==0,p.stdout
        path=sorted((bundle.task_path.parent/'agent_runs').glob('*/summary.json'))[-1]
        summary=json.loads(path.read_text(encoding='utf-8'))
        assert summary['cached_history'] and summary['selected']['strategy']==strategy,summary
        assert abs(summary['selected_score']-measured)<1e-12
        assert not list(path.parent.glob('experiment-*/metrics.json')),'Repeated validation fit'
        refits=json.loads((path.parent/'initial_refit.json').read_text(encoding='utf-8'))
        assert len(refits)==1 and refits[0]['result']['status']=='completed',refits
        frame=pd.read_csv(bundle.task_path.parent/'submission.csv'); validate_submission(bundle,frame)
        assert len(frame)==nrows
        audit=json.loads((path.parent/'refit-01/preprocessing-audit.json').read_text(encoding='utf-8'))
        if task=='multiclass':
            from automl_known import KNOWN_TASKS
            expected=next(r['candidate']['class_multipliers'] for t in KNOWN_TASKS if t['metric']=='macro_f1' for r in t['records'] if r['candidate'].get('strategy')==strategy)
            assert audit['calibration_multipliers']==expected or audit.get('class_multipliers')==expected
            assert audit['calibration_source']=='frozen_candidate' and not audit['folds'],'Known frozen calibration should not be retrained'
        result={'task':task,'strategy':strategy,'selected_validation_score':measured,'validation_fit_count':0,'full_refit_count':1,
                'rows':len(frame),'columns':list(frame),'seconds':time.monotonic()-started,'summary':summary,'audit':audit,
                'submission_sha256':hashlib.sha256((bundle.task_path.parent/'submission.csv').read_bytes()).hexdigest()}
        (parent/(task+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); records.append(result)
        print(json.dumps({k:result[k] for k in ['task','strategy','selected_validation_score','rows','seconds']},ensure_ascii=False),flush=True)
    result={'status':'passed','archive':str(archive),'archive_sha256':manifest['sha256'],'clean_extraction':str(extracted),'records':records,
            'real_platform_api_tested':False,'hidden_score_verified':False,'validation_fit_count':0}
    (parent/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (parent/'report.md').write_text('# Проверка улучшенного ZIP\n\nДве модели полностью обучены из чистой распаковки; обе стратегии выбраны по сохранённым H35/H37 метрикам. Повторных validationfit нет; submission.csv имеет правильные строки/столбцы/ID, классификационныекоэффициентызаморожены. Это проверка вывода, не новая оценкакачества и не проверкаhiddenplatformscore.\n\n[Доказательства](verification.json).\n',encoding='utf-8')
    print(json.dumps({'status':'passed','verification':str(parent/'verification.json')},ensure_ascii=False))

if __name__=='__main__': main()
