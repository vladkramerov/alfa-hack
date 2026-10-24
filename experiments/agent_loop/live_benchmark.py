"""Equal-budget real-Qwen masked CLI benchmark with preserved checkpoints."""
import argparse
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
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parents[2]
LEGACY=Path('/Users/maxchibisov/Projects/HACKS/alfa-hack')
WORK=Path('/private/tmp/alfa-another-live')
sys.path.insert(0,str(LEGACY))
from automl_data import load_task,validation_split,split_key,score


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('variant')
    parser.add_argument('--zip',required=True)
    parser.add_argument('--task',choices=['regression','classification'])
    args=parser.parse_args()
    archive=Path(args.zip).resolve()
    destination=ROOT/'results/agent-loop'/(args.variant+'.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    report=json.loads(destination.read_text()) if destination.exists() else {
        'status':'running','variant':args.variant,'zip_sha256':digest(archive),'runs':[],
        'protocol':'Same masked CSV and official dictionary; real organizer Qwen; 600s/8CPU/60iterations/TUNE_TRIALS0/20000tokens; identical seed42. Read-only launcher instrumentation; external targets absent from child. Historical outer repeatedly explored, not independent evidence.'}
    assert report['zip_sha256']==digest(archive)
    def save():destination.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str))
    save()
    for task,folder in [('regression','Регрессия'),('classification','Многоклассовая классификация')]:
        if args.task and args.task!=task:continue
        if any(r['task']==task for r in report['runs']):continue
        run=WORK/args.variant/task
        receipts=run/'receipts';directory=run/'task'
        receipts.mkdir(parents=True,exist_ok=True);directory.mkdir(exist_ok=True)
        state_path=run/'process.json'
        if state_path.exists():raise RuntimeError('Inspect process/receipts; never restart an existing run automatically')
        with zipfile.ZipFile(archive) as z:
            assert all('/' not in n and (n.endswith('.py') or n=='requirements.txt') for n in z.namelist())
            z.extractall(directory)
        fixture=Path('/private/tmp/alfa-strongest-e2e-v2')/(task+'-shadow')
        for name in ['train.csv','test.csv','sample_submission.csv','task.md']:
            shutil.copy2(fixture/name,directory/name)
        original=LEGACY/'drive-download-20261023T201455Z-1-001'/folder
        shutil.copy2(original/'data_dictionary.csv',directory/'data_dictionary.csv')
        bundle=load_task(original/'task.md');fit,valid,_=validation_split(bundle)
        source_hashes={p.name:digest(p) for p in directory.glob('*.py')}
        data_hashes={p.name:digest(p) for p in directory.glob('*.csv')}
        secret=None
        for line in (LEGACY/'.env.local').read_text().splitlines():
            if line.strip().startswith('OPENAI_API_KEY='):secret=line.split('=',1)[1].strip().strip('\"\'')
        assert secret and len(secret)>8
        env={k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_','AUTOML_','LLM_','COMBO_'))
             and k not in ['TARGET_SCORE','METRIC','CUSTOM_METRIC','PYTHONPATH']}
        env.update(OPENAI_API_KEY=secret,OPENAI_BASE_URL='https://changellenge.up.railway.app/v1',
            OPENAI_MODEL='qwen/qwen3.8-27b',LLM_DISABLED='0',LLM_MAX_TOKENS='20000',
            TIME_LIMIT='600',MAX_ITERATIONS='60',TUNE_TRIALS='0',N_JOBS='8',
            OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',
            PYTHONHASHSEED='0',MPLCONFIGDIR=str(WORK/'matplotlib'),BENCH_RECEIPTS=str(receipts))
        before={'source_hashes':source_hashes,'data_hashes':data_hashes,'split_key':split_key(fit,valid),
                'resources':{'seconds':600,'cpu':8,'iterations':60,'tune_trials':0,'llm_tokens':20000},
                'llm_model':env['OPENAI_MODEL'],'llm_base_url':env['OPENAI_BASE_URL']}
        (run/'provenance.json').write_text(json.dumps(before,indent=2))
        started=time.monotonic();peak=0;stopped=None
        with (run/'command.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'experiments/agent_loop/live_launcher.py'),'task.md'],
                cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state_path.write_text(json.dumps({'pid':child.pid,'created':time.time(),'cwd':str(directory)}))
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try:peak=max(peak,sum(q.memory_info().rss for q in [process]+process.children(recursive=True))/1024**2)
            except psutil.Error:pass
            if time.monotonic()-started>620 or peak>32000:
                stopped='time' if peak<=32000 else 'memory'
                os.killpg(child.pid,signal.SIGKILL);child.wait();break
            time.sleep(.5)
        agent_path=receipts/'agent.json'
        agent=json.loads(agent_path.read_text()) if agent_path.exists() else None
        result={'task':task,'exit_code':child.returncode,'seconds':time.monotonic()-started,
                'peak_rss_mb':peak,'stopped':stopped,'source_unchanged':all(digest(directory/n)==h for n,h in source_hashes.items()),
                'split_key':split_key(fit,valid),'directory':str(run),'agent':agent}
        submission=directory/'submission.csv'
        if submission.exists():
            sub=pd.read_csv(submission);sample=pd.read_csv(directory/'sample_submission.csv')
            assert list(sub)==list(sample) and len(sub)==len(valid) and not sub.isna().any().any()
            assert sub[bundle.id_column].equals(sample[bundle.id_column])
            result.update(score=score(bundle.metric,bundle.y.iloc[valid],sub[bundle.target].to_numpy(),classes=bundle.classes),
                          submission_sha256=digest(submission),submission_valid=True)
        else:result['submission_valid']=False
        events=[json.loads(line) for line in (receipts/'events.jsonl').read_text().splitlines()] if (receipts/'events.jsonl').exists() else []
        result['llm_requests']=[e for e in events if e['event']=='request']
        result['planner_responses']=[e for e in events if e['event']=='proposal']
        result['decisions']=[e for e in events if e['event']=='decision']
        result['json_responses']=[e for e in events if e['event']=='json']
        payload=json.dumps(result,indent=2,ensure_ascii=False,default=str)
        assert secret not in payload
        report['runs'].append(result);save()
        print(json.dumps({k:v for k,v in result.items() if k not in ['agent','llm_requests','planner_responses','decisions','json_responses']},ensure_ascii=False),flush=True)
    report['status']='completed' if len(report['runs'])==2 else 'partial'
    save()


if __name__=='__main__':main()
