"""Seven SPEC-008 graphs, frozen sources and persistent no-replay claims."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
from automl_data import load_task,split_key

ROOT=Path(__file__).resolve().parents[1]
BASELINE={'regression':41.480963027907194,'multiclass':.6286047899141216}
DATE='2026-10-23'
SUITE=ROOT/f'results/{DATE}/series-combinations-v1'
SOURCES=['automl_data.py','automl_models.py','automl_preprocessing.py','automl_advanced.py',
 'automl_combo_memory.py','automl_feature_sets.py','automl_reg_combo.py','automl_class_combo.py',
 'experiments/novel_models.py','experiments/twenty_features.py','experiments/twenty_common.py',
 'experiments/twenty_advanced.py','experiments/reg_combinations.py','experiments/class_combinations.py',
 'experiments/combo_worker.py','experiments/run_combinations.py']
def now():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8');os.replace(temp,path)
def experiment_plan():
    strategies=['reg_feature_experts','reg_feature_experts','reg_convex_blend','target_tail_mixture','smote_macro_f1','frequency_blend_macro_f1','probability_stacking']
    jobs=[]
    for number,strategy in zip(range(42,49),strategies):
        task='regression' if number<46 else 'multiclass'
        c={'strategy':strategy,'parameters':{},'family':'extra_trees' if number<45 else 'xgboost'}
        if number in (42,43):c.update(feature_numbers=[30] if number==42 else [29,30,31],group_column='building_class',minimum_expert_rows=500,shrinkage=500.)
        if number==45:c['tail_quantile']=.8
        jobs.append({'number':number,'hypothesis':f'HYP-{number:03d}','task':task,'candidate':c,'version':'spec008-v1','reference':BASELINE[task]})
    return jobs
def claim_attempt(root,job,hashes,splitkey):
    root=Path(root);ledger=(root/'HYPOTHESES.md').read_text(encoding='utf-8')
    if job['hypothesis'] not in ledger:raise RuntimeError('Unregistered hypothesis')
    path=root/'results/spec008-attempts.json';rows=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    identity={'task':job['task'],'candidate':job['candidate'],'hashes':hashes,'split_key':splitkey}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if any(r['fingerprint']==fingerprint or (r['job']['number']==job['number'] and r['job']['task']==job['task']) for r in rows):raise RuntimeError('Already attempted; do not replay')
    row={**identity,'fingerprint':fingerprint,'job':job,'status':'started','started':now(),'ledger_sha256_before_fit':hashlib.sha256(ledger.encode()).hexdigest()}
    rows.append(row);save(path,rows);return row
def main(only=None):
    SUITE.mkdir(parents=True,exist_ok=True);jobs=experiment_plan()
    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}
    frozen=SUITE/'frozen-plan.json'
    if frozen.exists():
        old=json.loads(frozen.read_text(encoding='utf-8'))
        if old['sources']!=sources or old['jobs']!=jobs:raise RuntimeError('Frozen source/configuration changed')
    else:
        save(frozen,{'created':now(),'jobs':jobs,'sources':sources,'baseline_refits':0})
        for n in SOURCES:
            dst=SUITE/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    registry=ROOT/'results/spec008-attempts.json'
    for job in jobs:
        if only and job['number'] not in only:continue
        rows=json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else []
        if any(r['job']['number']==job['number'] for r in rows):print('Skip attempted '+job['hypothesis'],flush=True);continue
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=s for n,s in sources.items()):raise RuntimeError('Code changed during frozen experiment')
        (ROOT/'HYPOTHESES.md').read_text(encoding='utf-8')
        bundle=load_task(ROOT/'data'/job['task']/'task.md')
        with np.load(ROOT/f'results/{DATE}/series-{job["task"]}-preprocessing-v1/splits.npz') as z:fit,valid=z['outer_fit'],z['outer_valid']
        control=next(r for r in json.loads((ROOT/f'results/{DATE}/new-family-records.json').read_text(encoding='utf-8')) if r['task']==job['task'])
        if bundle.hashes!=control['hashes'] or split_key(fit,valid)!=control['split_key']:raise RuntimeError('Data/split changed')
        record=claim_attempt(ROOT,job,bundle.hashes,split_key(fit,valid))
        index=1;parent=ROOT/'results'/now()[:10];parent.mkdir(parents=True,exist_ok=True)
        while list(parent.glob(f'{job["hypothesis"]}-{index:02d}-*')):index+=1
        directory=parent/f'{job["hypothesis"]}-{index:02d}-{job["task"]}-combinations';directory.mkdir()
        np.savez_compressed(directory/'splits.npz',fit=fit,valid=valid)
        record.update(directory=directory.relative_to(ROOT).as_posix(),sources=sources)
        rows.append(record);save(registry,rows);save(directory/'evidence.json',record)
        request={'job':job,'task_path':str(bundle.task_path),'output':str(directory),'split_path':str(directory/'splits.npz'),
                 'cache_dir':str(SUITE/'cache'),'legacy_cache_dir':str(ROOT/f'results/{DATE}/series-twenty-hypotheses-v1/cache')}
        save(directory/'request.json',request)
        card=ROOT/f'docs/hypotheses/{job["hypothesis"]}.md'
        card.write_text('# '+job['hypothesis']+'\n\n[SPEC-008](../specs/SPEC-008-combinations.md), [frozen methods](HYP-042-048-combinations.md).\n\nДо fit:\n```json\n'+json.dumps(job,indent=2,ensure_ascii=False)+'\n```\n',encoding='utf-8')
        (directory/'report.md').write_text('# '+job['hypothesis']+'\n\nStarted before fit. [SPEC-008](../../../docs/specs/SPEC-008-combinations.md).\n\nCommand: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_combinations`.\n\n[evidence](evidence.json), [request](request.json), [frozen sources](../series-combinations-v1/source/).\n',encoding='utf-8')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',
            COMBO_RUN_DIR=str(SUITE),COMBO_LEDGER_PATH=str(ROOT/'HYPOTHESES.md'),COMBO_HYPOTHESIS=job['hypothesis'])
        for key in list(env):
            if 'API_KEY' in key:env.pop(key,None)
        start=time.monotonic();print('START '+job['hypothesis'],flush=True)
        try:
            with (directory/'worker.log').open('w',encoding='utf-8') as log:
                p=subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.combo_worker',str(directory/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            result=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
            if p.returncode and result['status']=='completed':raise RuntimeError('Worker nonzero exit')
        except Exception as exc:
            result={'status':'timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed','error':repr(exc),'seconds':time.monotonic()-start};save(directory/'metrics.json',result)
        ok=result['status']=='completed';improved=ok and (result['score']<BASELINE[job['task']]-1e-10 if job['task']=='regression' else result['score']>BASELINE[job['task']]+1e-10)
        conclusion='Подтверждена в проверенных условиях' if improved else 'Не подтверждена по заданному критерию' if ok else 'Недостаточно данных'
        record.update(result,conclusion=conclusion,improved=improved,finished=now());rows[-1]=record
        save(registry,rows);save(directory/'evidence.json',record);save(SUITE/'records.json',rows)
        detail=f'{result["status"]}, {result.get("metric")}={result.get("score")}; {conclusion}'
        rel=record['directory']
        with (directory/'report.md').open('a',encoding='utf-8') as f:f.write('\nResult: **'+detail+'**.\n\nBaseline '+str(BASELINE[job['task']])+' reused without fit. [metrics](metrics.json), [audit](audit.json), [log](worker.log), [predictions](validation_predictions.npz).\n\nRepeated exploratory outer; hidden platform score unknown. No outer target used for learned choices.\n')
        with card.open('a',encoding='utf-8') as f:f.write('\n'+detail+f'; [report](../../{rel}/report.md). Не повторять.\n')
        with (ROOT/'HYPOTHESES.md').open('a',encoding='utf-8') as f:f.write(f'\n- SPEC-008 {record["finished"]}: **{job["hypothesis"]}/{job["task"]}**: {detail}; [отчёт]({rel}/report.md). Повтор не запускать.\n')
        with (ROOT/'results/README.md').open('a',encoding='utf-8') as f:f.write(f'\n| {job["hypothesis"]} | {job["task"]} | {detail} | [report](../{rel}/report.md) |\n')
        print(json.dumps({'hypothesis':job['hypothesis'],**result,'improved':improved},ensure_ascii=False,default=str),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='*',type=int);main(p.parse_args().only)
