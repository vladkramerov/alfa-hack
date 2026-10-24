"""Frozen ZIPs, honest shadow submissions, then both original public tasks.

Shadow test labels live only in this parent evaluator. Each arm is a fresh CLI
process with the same 8 CPUs, 1200s/16-trial bounds and no LLM configuration.
Original tasks use 1800s and the candidate's default trial cap. No Boosters upload.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from package_solution import build

OUT=ROOT/'results/score-recovery/e2e.json'
WORK=Path('/private/tmp/alfa-score-recovery-e2e')
DATA=ROOT/'drive-download-20261023T201455Z-1-001'


def run():
    if OUT.exists(): raise RuntimeError('Reuse existing e2e report; no repeated fits')
    WORK.mkdir(exist_ok=False)
    archives={'control':WORK/'control.zip','candidate':WORK/'candidate.zip'}
    archives['control'].write_bytes(subprocess.check_output(['git','show','be9d9ae:dist/alfa_autoresearch_submission.zip']))
    manifest=build(archives['candidate'])
    # Preserve this exact delivered byte stream; docs/commit metadata do not alter it.
    shutil.copy2(archives['candidate'],ROOT/'dist/alfa_autoresearch_submission.zip')
    (ROOT/'dist/alfa_autoresearch_submission.manifest.json').write_text(json.dumps(manifest,indent=2))
    digests={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in archives.items()}
    report={'protocol':'Source-frozen CLI; masked external holdout then original public tasks. No API; no Boosters upload.',
            'python':sys.version,'archive_sha256':digests,'candidate_manifest':manifest,'runs':[],
            'limitations':['Local macOS, not official Docker.','Original test labels unknown.','Public holdouts have historical exposure.']}
    def save():OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    save()
    for task,folder in [('regression','Регрессия'),('multiclass','Многоклассовая классификация')]:
        bundle=load_task(DATA/folder/'task.md');fit,valid,description=validation_split(bundle)
        truths=bundle.train.iloc[valid].copy()
        for scope,arm in [('shadow','control'),('shadow','candidate'),('original','candidate')]:
            directory=WORK/f'{task}-{scope}-{arm}';directory.mkdir()
            if scope=='shadow':
                bundle.train.iloc[fit].to_csv(directory/'train.csv',index=False)
                truths.drop(columns=bundle.target).to_csv(directory/'test.csv',index=False)
                sample=truths[[bundle.id_column]].copy() if bundle.id_column else pd.DataFrame(index=np.arange(len(valid)))
                sample[bundle.target]=float(bundle.y.iloc[fit].mean()) if task=='regression' else bundle.y.iloc[fit].mode().iloc[0]
                sample.to_csv(directory/'sample_submission.csv',index=False)
                if bundle.paths['dictionary']:shutil.copy2(bundle.paths['dictionary'],directory/'data_dictionary.csv')
                metric='RMSE' if task=='regression' else 'Macro F1'
                (directory/'task.md').write_text(f'Primary metric: {metric}\nTarget: {bundle.target}\nIdentifier: {bundle.id_column}\nTraining data: train.csv\nTest data: test.csv\nSample submission: sample_submission.csv\nData dictionary: data_dictionary.csv\n')
            else:
                for name in ['task.md','train.csv','test.csv','sample_submission.csv','data_dictionary.csv']:
                    shutil.copy2(DATA/folder/name,directory/name)
            with zipfile.ZipFile(archives[arm]) as z:z.extractall(directory)
            env={k:v for k,v in os.environ.items() if not k.startswith('OPENAI_')}
            budget=1200 if scope=='shadow' else 1800
            env.update(AUTOML_TIME_LIMIT=str(budget),AUTOML_THREADS='8',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(WORK/'matplotlib'))
            if scope=='shadow':env['AUTOML_MAX_EXPERIMENTS']='16'
            else:env.pop('AUTOML_MAX_EXPERIMENTS',None)
            started=time.monotonic();peak=0.
            with (directory/'cli.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'solution.py','task.md'],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                process=psutil.Process(child.pid)
                while child.poll() is None:
                    try:
                        processes=[process]+process.children(recursive=True)
                        rss=sum(p.memory_info().rss for p in processes if p.is_running())/1024**2;peak=max(peak,rss)
                    except (psutil.Error,PermissionError):pass
                    if time.monotonic()-started>budget+30 or peak>32000:
                        os.killpg(child.pid,signal.SIGKILL);child.wait();break
                    time.sleep(.5)
            summaries=sorted((directory/'agent_runs').glob('*/summary.json'))
            if child.returncode or not summaries:raise RuntimeError(f'CLI failed; inspect {directory}/cli.log')
            summary=json.loads(summaries[-1].read_text())
            submission=pd.read_csv(directory/'submission.csv');expected=pd.read_csv(directory/'sample_submission.csv')
            assert list(submission)==list(expected) and len(submission)==len(expected)
            assert not submission.isna().any().any()
            if bundle.id_column:assert submission[bundle.id_column].equals(expected[bundle.id_column])
            value=None
            if scope=='shadow':
                prediction=submission[bundle.target].to_numpy()
                if task=='regression':prediction=prediction.astype(float)
                value=score(bundle.metric,truths[bundle.target].to_numpy(),prediction,classes=bundle.classes)
            record={'task':task,'scope':scope,'arm':arm,'archive_sha256':digests[arm],
                    'wall_seconds':time.monotonic()-started,'peak_rss_mb':peak,'budget_seconds':budget,
                    'trial_cap':16 if scope=='shadow' else 36,'threads':8,'api_calls':summary.get('llm_http_calls_reported'),
                    'external_split_key':split_key(fit,valid) if scope=='shadow' else None,
                    'submission_score':value,'metric':bundle.metric,'submission_rows':len(submission),
                    'submission_sha256':hashlib.sha256((directory/'submission.csv').read_bytes()).hexdigest(),
                    'local_directory':str(directory),'summary':summary}
            report['runs'].append(record);save()
            print(json.dumps({k:record[k] for k in ['task','scope','arm','wall_seconds','peak_rss_mb','submission_score']},ensure_ascii=False),flush=True)
            destination=OUT.parent/f'{task}-{scope}-{arm}';destination.mkdir()
            for name in ['summary.json','experiments.json','validation.json','audit.json','frozen_selection.json','llm.json']:
                source=summaries[-1].parent/name
                if source.exists():shutil.copy2(source,destination/name)
    report['status']='completed';save()


if __name__=='__main__':run()
