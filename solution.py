"""Universal bounded AutoResearch entry point: python solution.py /path/task.md."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from automl_data import load_task,validation_split,split_key,better,fallback_submission,atomic_submission,validate_submission
from automl_history import candidate_key,known_task,already_known,ledger_case,read_ledger,write_ledger
from automl_llm import propose


def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def worker(bundle,candidate,mode,directory,splits,timeout):
    directory.mkdir(parents=True,exist_ok=False)
    save(directory/'request.json',{'task_path':str(bundle.task_path),'candidate':candidate,'mode':mode,'output':str(directory),'split_path':str(splits),'threads':4})
    env=os.environ.copy()
    for key in list(env):
        if 'API_KEY' in key: env.pop(key,None)
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
    try:
        with (directory/'worker.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('automl_worker.py')),str(directory/'request.json')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,timeout),check=False)
        result=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    except Exception as exc:
        result={'status':'failed','error':type(exc).__name__}
        save(directory/'metrics.json',result)
    (directory/'report.md').write_text('# '+mode+'\n\n'+json.dumps({'candidate':candidate,'result':result},ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main(task_path):
    started=time.monotonic(); budget=min(10200,max(60,int(os.getenv('AUTOML_TIME_LIMIT','10200'))))
    bundle=load_task(task_path)
    atomic_submission(bundle,fallback_submission(bundle))
    root=bundle.task_path.parent/'agent_runs'/datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    root.mkdir(parents=True); records=[]; llm=[]
    fit,valid,description=validation_split(bundle)
    np.savez_compressed(root/'splits.npz',fit=fit,valid=valid)
    prior=known_task(bundle.hashes,split_key(fit,valid),bundle.metric)
    ledger=bundle.task_path.parent/'hypotheses.json'
    case=ledger_case(bundle.hashes,split_key(fit,valid),bundle.metric)
    records=read_ledger(ledger,case)['records']
    best=None; best_score=None
    if prior:
        for record in prior['records']:
            if record.get('status','completed')=='completed' and better(bundle.metric,record['score'],best_score): best,best_score=record['candidate'],record['score']
    profile={'task':bundle.text,'target':bundle.target,'id':bundle.id_column,'type':bundle.task_type,'metric':bundle.metric,'split':description,
             'shape':list(bundle.X.shape),'dtypes':bundle.X.dtypes.astype(str).to_dict(),'missing':bundle.X.isna().mean().to_dict(),
             'categories':{c:bundle.X[c].nunique() for c in bundle.X.select_dtypes(exclude='number')},'known':prior,'hashes':bundle.hashes,'split_key':split_key(fit,valid)}
    for earlier in sorted(root.parent.glob('*/profile.json')):
        if earlier.parent==root: continue
        try:
            old=json.loads(earlier.read_text(encoding='utf-8'))
            if old.get('hashes')==profile['hashes'] and old.get('split_key')==profile['split_key']:
                old_records=json.loads((earlier.parent/'experiments.json').read_text(encoding='utf-8'))
                for record in old_records:
                    if not any(r.get('candidate_key')==record.get('candidate_key') for r in records): records.append(record)
        except (OSError,ValueError): pass
    save(root/'profile.json',profile)
    def remember():
        write_ledger(ledger,case,{'hashes':bundle.hashes,'split_key':split_key(fit,valid),'metric':bundle.metric,'known':prior,'records':records})
        save(root/'experiments.json',records)
    remember()
    refit_count=0
    def refit(candidate):
        nonlocal refit_count
        refit_count+=1; directory=root/f'refit-{refit_count:02d}'
        remaining=budget-(time.monotonic()-started)
        result=worker(bundle,candidate,'refit',directory,root/'splits.npz',min(1800,remaining))
        if result['status']=='completed': atomic_submission(bundle,pd.read_csv(directory/'candidate_submission.csv'))
        return result
    available=(list(prior['records']) if prior else [])+[r for r in records if r.get('status')=='completed' and 'score' in r]
    available.sort(key=lambda r:r['score'],reverse=bundle.metric not in ['rmse','mae','log_loss'])
    best=None; best_score=None; attempted=set(); initial=[]
    for r in available:
        identity=candidate_key(r['candidate'])
        if identity in attempted: continue
        attempted.add(identity)
        result=refit(r['candidate']); initial.append({'candidate':r['candidate'],'result':result})
        if result['status']=='completed': best,best_score=r['candidate'],r['score']; break
        if len(initial)>=3: break
    save(root/'initial_refit.json',initial)
    candidates=[{'family':'lightgbm','parameters':{},'hypothesis':'Native categorical boosting, a new model family'},
                {'family':'extra_trees','parameters':{},'hypothesis':'Random thresholds reduce variance, a new model family'}]
    if os.getenv('AUTOML_PORTFOLIO','1')=='0': candidates=[]
    limit=min(6,max(0,int(os.getenv('AUTOML_MAX_EXPERIMENTS','4'))))
    llm_calls=0
    inherited=len(records)
    while len([r for r in records[inherited:] if r['status']!='skipped'])<limit and budget-(time.monotonic()-started)>max(30,min(600,budget*.2)):
        if candidates: candidate=candidates.pop(0)
        elif llm_calls<3:
            llm_calls+=1
            candidate,status=propose(profile,records,timeout=min(45,max(1,budget-(time.monotonic()-started))))
            llm.append(status); save(root/'llm.json',llm)
            if candidate is None: break
        else: break
        try:
            identity=candidate_key(candidate)
            checked=read_ledger(ledger,case) # Read the persistent file before EVERY attempted hypothesis.
            if already_known(candidate,checked['records'],prior):
                records.append({'candidate_key':identity,'candidate':candidate,'status':'skipped','reason':'already checked'}); remember(); continue
            record={'candidate_key':identity,'candidate':candidate,'status':'started'}; records.append(record); remember()
            directory=root/f'experiment-{len(records):02d}'
            result=worker(bundle,candidate,'evaluate',directory,root/'splits.npz',min(1200,budget-(time.monotonic()-started)-min(600,budget*.2)))
            record.update(result); remember()
            if result['status']=='completed' and better(bundle.metric,result['score'],best_score):
                proposal_score=result['score']; refit_result=refit(candidate)
                if refit_result['status']=='completed': best,best_score=candidate,proposal_score
        except Exception as exc:
            llm.append({'status':'candidate_rejected','error':type(exc).__name__}); save(root/'llm.json',llm)
            if len(llm)>8: break
    frame=pd.read_csv(bundle.task_path.parent/'submission.csv'); validate_submission(bundle,frame)
    summary={'status':'completed','metric':bundle.metric,'selected_score':best_score,'selected':best,'validation':description,'cached_history':bool(prior),
             'experiments':records,'llm':llm,'seconds':time.monotonic()-started,'submission_rows':len(frame),
             'limitation':'Validation selection is exploratory, not a hidden-platform score; real API requires platform environment.'}
    save(root/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','metric','selected_score','cached_history','seconds','submission_rows']}))


if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Usage: python solution.py /path/to/task.md')
    main(sys.argv[1])
