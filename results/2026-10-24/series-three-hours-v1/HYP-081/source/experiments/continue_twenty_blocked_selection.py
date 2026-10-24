"""Continue only H28 pre-fit guard stops; reuse every previously fitted inner model."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from automl_data import load_task
from experiments.run_twenty_hypotheses import ROOT,SUITE,REGISTRY,SOURCES,REFERENCES,save,now
from experiments.twenty_common import digest_frame
from experiments.twenty_advanced import inner_fingerprint,stable_feature_selection

def main():
    rows=json.loads(REGISTRY.read_text(encoding='utf-8'))
    failures=[r for r in rows if r['candidate']['number']==28 and r['status']=='failed' and
              r.get('error')=='RuntimeError: Known outer baseline fit prohibited; use saved predictions' and not r.get('continuation_of')]
    if not failures: print('No eligible pre-fit guard stops'); return
    source=SUITE/'continuations'/'source'; source.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in SOURCES+['experiments/continue_twenty_blocked_selection.py']:
        content=(ROOT/name).read_bytes(); hashes[name]=hashlib.sha256(content).hexdigest()
        target=source/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(content)
    save(source.parent/'source-manifest.json',hashes)
    for previous in failures:
        if any(r.get('continuation_of')==previous['fingerprint'] for r in rows): continue
        job=previous['candidate']; task=job['task']
        ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf-8')
        request=json.loads((ROOT/previous['directory']/'request.json').read_text(encoding='utf-8'))
        bundle=load_task(request['task_path'])
        with np.load(request['split_path']) as split: fit,valid=split['fit'],split['valid']
        X,y=bundle.X.iloc[fit],bundle.y.iloc[fit]
        key=inner_fingerprint(task,X,np.asarray(y),fit,{'policy':'legacy','repeats':2})
        cache=SUITE/'cache'/('permutation-'+key+'.npz')
        with np.load(cache) as data: keep=stable_feature_selection(list(X),data['importance'])
        if not keep or set(keep)==set(X): raise RuntimeError('Continuation requires proven new reduced feature matrix')
        selected_hash=digest_frame(X[keep])
        attempts=[json.loads(line) for line in (SUITE/'model-fit-attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        if any(a['identity']['X']==selected_hash for a in attempts):
            raise RuntimeError('Selected outer model already attempted; no repeat allowed')
        index=1
        while list(SUITE.parent.glob(f'HYP-028-{index:02d}-*')): index+=1
        directory=SUITE.parent/f'HYP-028-{index:02d}-{task}-guard-continuation'; directory.mkdir()
        request.update(output=str(directory)); save(directory/'request.json',request)
        np.savez_compressed(directory/'splits.npz',fit=fit,valid=valid)
        request['split_path']=str(directory/'splits.npz'); save(directory/'request.json',request)
        identity=hashlib.sha256((previous['fingerprint']+json.dumps(hashes,sort_keys=True)).encode()).hexdigest()
        record={**previous,'fingerprint':identity,'status':'started','started':now(),'directory':directory.relative_to(ROOT).as_posix(),
                'sources':hashes,'source_root':source.relative_to(ROOT).as_posix(),'continuation_of':previous['fingerprint'],
                'ledger_sha256_before_fit':hashlib.sha256(ledger.encode()).hexdigest()}
        for key in ['error','seconds','conclusion','finished']: record.pop(key,None)
        save(directory/'continuation-proof.json',{'original_guard_stop':previous['directory'],'training_started_for_reduced_matrix_before':False,
            'selected_columns':keep,'dropped_columns':[c for c in X if c not in keep],'selected_X_sha256':selected_hash,
            'permutation_cache':str(cache),'no_inner_refit':True,'same_original_hypothesis':True})
        rows.append(record); save(REGISTRY,rows); save(directory/'evidence.json',record)
        (directory/'report.md').write_text(f'# HYP-028 {task}: техническое продолжение\n\nПредыдущий запуск остановлен до training новой outer-модели чрезмерно строгой защитой. Его ошибка и артефакты сохранены. Новая гипотеза не придумана, прежние модели не повторяются.\n\n[Доказательство нового набора признаков](continuation-proof.json); все inner-предсказания/модели и permutation importance загружаются из кэша. [Исходники](../series-twenty-hypotheses-v1/continuations/source/), [evidence](evidence.json).\n\nКоманда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.continue_twenty_blocked_selection`.\n',encoding='utf-8')
        env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8',
            SPEC006_RUN_DIR=str(SUITE),SPEC006_HYPOTHESIS='HYP-028/'+task+'/guard-continuation',SPEC006_ENFORCE_BASELINE_GUARD='1')
        for name in list(env):
            if 'API_KEY' in name: env.pop(name,None)
        started=time.monotonic()
        try:
            with (directory/'worker.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.twenty_worker',str(directory/'request.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=False)
            result=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        except Exception as exc: result={'status':'failed','error':repr(exc),'seconds':time.monotonic()-started}; save(directory/'metrics.json',result)
        improvement=result['status']=='completed' and (result['score']<REFERENCES[task] if task=='regression' else result['score']>REFERENCES[task])
        conclusion='Подтверждена в проверенных условиях' if improvement else 'Не подтверждена по заданному критерию' if result['status']=='completed' else 'Недостаточно данных'
        record.update(result,finished=now(),conclusion=conclusion); rows[-1]=record
        save(REGISTRY,rows); save(SUITE/'records.json',rows); save(directory/'evidence.json',record)
        detail=f'{result["status"]}, {result.get("metric")}={result.get("score")}; {conclusion}'
        with (directory/'report.md').open('a',encoding='utf-8') as f: f.write(f'\nИтог: {detail}. [Метрики](metrics.json), [аудит](preprocessing-audit.json), [предсказания](validation_predictions.npz). Внешнийholdoutужеиспользовался; выводразведочный.\n')
        for path,prefix in [(ROOT/'HYPOTHESES.md',''),(ROOT/'docs/hypotheses/HYP-028.md','../../'),(ROOT/'results/README.md','../')]:
            with path.open('a',encoding='utf-8') as f: f.write(f'\n- SPEC-006 HYP-028/{task}, техническое продолжение безповтораinner: {detail}; [отчёт]({prefix}{record["directory"]}/report.md). Повтор не запускать.\n')
        print(json.dumps({'hypothesis':'HYP-028','task':task,**result},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
