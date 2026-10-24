"""SPEC-009 frozen first wave; previous fits are read-only artifacts."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from automl_data import load_task,split_key
from experiments.run_combinations import save,now,ROOT
BASELINE={'regression':40.972997661684175,'multiclass':.6495311620074445}
ORIGINAL='2026-10-23';DATE='2026-10-24';SUITE=ROOT/f'results/{DATE}/series-alternatives-v1'
SOURCES=['automl_data.py','automl_models.py','automl_preprocessing.py','automl_advanced.py','automl_feature_sets.py',
 'automl_combo_memory.py','automl_reg_combo.py','automl_class_combo.py','automl_next_reg.py','automl_next_class.py','automl_temporal_memory.py',
 'experiments/twenty_common.py','experiments/twenty_advanced.py','experiments/twenty_features.py','experiments/novel_models.py',
 'experiments/next_regression.py','experiments/next_classification.py','experiments/run_combinations.py',
 'experiments/run_alternatives.py','experiments/alternative_worker.py']
def experiment_plan():
    methods=[(69,'regression','temporal_profile_memory'),(49,'multiclass','soft_distillation'),(51,'multiclass','balanced_softmax'),
      (53,'multiclass','nystrom_rbf'),(54,'regression','spline_gam'),(55,'regression','quantile_mean'),
      (56,'regression','h43_residual'),(58,'regression','covariate_shift'),(60,'regression','masked_regression'),(60,'multiclass','masked_augmentation')]
    return [{'number':n,'hypothesis':f'HYP-{n:03d}','task':t,'candidate':{'strategy':s,'parameters':{}},
             'reference':BASELINE[t],'version':'spec009-v1'} for n,t,s in methods]
def claim(root,job,hashes,key):
    root=Path(root);ledger=(root/'HYPOTHESES.md').read_text(encoding='utf8')
    if job['hypothesis'] not in ledger:raise RuntimeError('Unregistered hypothesis')
    path=root/'results/spec009-attempts.json';rows=json.loads(path.read_text(encoding='utf8')) if path.exists() else []
    identity={'task':job['task'],'candidate':job['candidate'],'hashes':hashes,'split_key':key}
    fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if any(r['fingerprint']==fp or (r['job']['number']==job['number'] and r['job']['task']==job['task']) for r in rows):raise RuntimeError('Attempt already exists; no replay')
    row={**identity,'job':job,'fingerprint':fp,'status':'started','started':now(),'ledger_sha256_before_fit':hashlib.sha256(ledger.encode()).hexdigest()}
    save(path,rows+[row]);return row
def main(only=None):
    SUITE.mkdir(parents=True,exist_ok=True);jobs=experiment_plan();registry=ROOT/'results/spec009-attempts.json'
    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES};frozen=SUITE/'frozen-plan.json'
    if frozen.exists():
        old=json.loads(frozen.read_text(encoding='utf8'))
        if old['sources']!=sources or old['jobs']!=jobs:raise RuntimeError('Frozen sources/config changed')
    else:
        save(frozen,{'created':now(),'jobs':jobs,'sources':sources,'baseline_validation_refits':0})
        for n in SOURCES:
            p=SUITE/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    for job in jobs:
        if only and job['number'] not in only:continue
        rows=json.loads(registry.read_text(encoding='utf8')) if registry.exists() else []
        if any(r['job']['number']==job['number'] and r['job']['task']==job['task'] for r in rows):print('SKIP ATTEMPTED '+job['hypothesis']+'/'+job['task'],flush=True);continue
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=s for n,s in sources.items()):raise RuntimeError('Training source drift')
        bundle=load_task(ROOT/'data'/job['task']/'task.md')
        with np.load(ROOT/f'results/{ORIGINAL}/series-{job["task"]}-preprocessing-v1/splits.npz') as z:fit,valid=z['outer_fit'],z['outer_valid']
        previous=next(r for r in json.loads((ROOT/f'results/{ORIGINAL}/new-family-records.json').read_text(encoding='utf8')) if r['task']==job['task'])
        if bundle.hashes!=previous['hashes'] or split_key(fit,valid)!=previous['split_key']:raise RuntimeError('Data/split drift')
        record=claim(ROOT,job,bundle.hashes,split_key(fit,valid));parent=ROOT/'results'/now()[:10];parent.mkdir(parents=True,exist_ok=True)
        i=1
        while list(parent.glob(f'{job["hypothesis"]}-{i:02d}-*')):i+=1
        out=parent/f'{job["hypothesis"]}-{i:02d}-{job["task"]}-alternatives';out.mkdir()
        record.update(directory=out.relative_to(ROOT).as_posix(),sources=sources);rows.append(record);save(registry,rows);save(out/'evidence.json',record)
        np.savez_compressed(out/'splits.npz',fit=fit,valid=valid)
        req={'job':job,'task_path':str(bundle.task_path),'output':str(out),'split_path':str(out/'splits.npz'),
             'cache_dir':str(SUITE/'cache'),'teacher_cache_dir':str(ROOT/f'results/{ORIGINAL}/series-combinations-v1/cache'),
             'legacy_cache_dir':str(ROOT/f'results/{ORIGINAL}/series-twenty-hypotheses-v1/cache')};save(out/'request.json',req)
        card=ROOT/f'docs/hypotheses/{job["hypothesis"]}.md'
        if not card.exists():card.write_text('# '+job['hypothesis']+'\n\n[SPEC009](../specs/SPEC-009-alternative-models.md); [новизна](HYP-049-068-proposals.md). H69описана вSPEC009.\n\nПлан до fit:\n```json\n'+json.dumps([j for j in jobs if j['number']==job['number']],indent=2,ensure_ascii=False)+'\n```\n',encoding='utf8')
        (out/'report.md').write_text('# '+job['hypothesis']+' / '+job['task']+'\n\nStarted before fit. [SPEC009](../../../docs/specs/SPEC-009-alternative-models.md), [request](request.json), [data/source/split](evidence.json).\n\nCommand: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_alternatives`. Фиксированные параметры, old baseline без повторного fit, outer повторно просмотрен.\n',encoding='utf8')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',COMBO_RUN_DIR=str(SUITE),COMBO_LEDGER_PATH=str(ROOT/'HYPOTHESES.md'),COMBO_HYPOTHESIS=job['hypothesis']+'/'+job['task'])
        for k in list(env):
            if 'API_KEY' in k:env.pop(k,None)
        started=time.monotonic();print('START '+job['hypothesis']+'/'+job['task'],flush=True)
        try:
            with (out/'worker.log').open('w',encoding='utf8') as log:
                p=subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.alternative_worker',str(out/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            result=json.loads((out/'metrics.json').read_text(encoding='utf8'))
            if p.returncode and result['status']=='completed':raise RuntimeError('Worker exit mismatch')
        except Exception as exc:
            result={'status':'timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed','error':repr(exc),'seconds':time.monotonic()-started};save(out/'metrics.json',result)
        ok=result['status']=='completed';improved=ok and (result['score']<BASELINE[job['task']]-1e-10 if job['task']=='regression' else result['score']>BASELINE[job['task']]+1e-10)
        conclusion='Подтверждена в проверенных условиях' if improved else 'Не подтверждена по заданному критерию' if ok else 'Недостаточно данных'
        record.update(result,improved=improved,conclusion=conclusion,finished=now());rows[-1]=record;save(registry,rows);save(out/'evidence.json',record);save(SUITE/'records.json',rows)
        detail=f'{result["status"]}, {result.get("metric")}={result.get("score")}; {conclusion}';rel=record['directory']
        with (out/'report.md').open('a',encoding='utf8') as f:f.write('\n**'+detail+'**.\n\n[metrics](metrics.json), [audit](audit.json), [log](worker.log).\n\nHiddenplatformscore не установлен; старые контрольныеvalidationfit не выполнялись. Повтор не запускать.\n')
        with card.open('a',encoding='utf8') as f:f.write('\n'+job['task']+': '+detail+f'; [report](../../{rel}/report.md). Не повторять.\n')
        with (ROOT/'HYPOTHESES.md').open('a',encoding='utf8') as f:f.write(f'\n- SPEC009 {record["finished"]}: **{job["hypothesis"]}/{job["task"]}**: {detail}; [отчёт]({rel}/report.md). Повтор запрещён.\n')
        with (ROOT/'results/README.md').open('a',encoding='utf8') as f:f.write(f'\n| {job["hypothesis"]} | {job["task"]} | {detail} | [report](../{rel}/report.md) |\n')
        print(json.dumps({'hypothesis':job['hypothesis'],'task':job['task'],**{k:result.get(k) for k in ['status','metric','score','error','seconds']},'improved':improved},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='*',type=int);main(p.parse_args().only)
