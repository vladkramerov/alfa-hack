"""Recompute SPEC-006 metrics and verify artifacts without training any model."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,mean_squared_error
from automl_data import load_task,split_key
from experiments.run_twenty_hypotheses import ROOT,SUITE,REGISTRY,experiment_plan,REFERENCES,save

def main():
    rows=json.loads(REGISTRY.read_text(encoding='utf-8')); checks=[]
    def check(name,value): checks.append({'check':name,'passed':bool(value)})
    latest={(r['candidate']['number'],r['candidate']['task']):r for r in rows}
    check('23 planned cases',len(latest)==23)
    check('all planned cases have metric',all(r['status']=='completed' for r in latest.values()))
    check('all20 hypotheses',{r['candidate']['number'] for r in rows}==set(range(22,42)))
    check('no duplicate attempts',len({r['fingerprint'] for r in rows})==len(rows))
    frozen=json.loads((SUITE/'frozen-plan.json').read_text(encoding='utf-8'))
    supplied=Path('C:/Users/User/Downloads/requirements (1).txt')
    check('provided requirements unchanged',supplied.exists() and (ROOT/'requirements.txt').read_bytes()==supplied.read_bytes())
    for name,expected in frozen['sources'].items():
        check('source immutable '+name,hashlib.sha256((SUITE/'source'/name).read_bytes()).hexdigest()==expected)
    bundles={task:load_task(ROOT/'data'/task/'task.md') for task in ['regression','multiclass']}
    table=[]
    for r in rows:
        task=r['candidate']['task']; label=r['candidate']['hypothesis']+'/'+task; directory=ROOT/r['directory']; bundle=bundles[task]
        check(label+' terminal',r['status'] in ['completed','failed','timeout'])
        check(label+' data hashes',r['hashes']==bundle.hashes)
        check(label+' ledger read before fit',len(r['ledger_sha256_before_fit'])==64)
        if r.get('continuation_of'):
            check(label+' prior prefit stop retained',any(old['fingerprint']==r['continuation_of'] and old['status']=='failed' for old in rows))
            with (directory/'continuation-proof.json').open(encoding='utf-8') as f: proof=json.load(f)
            check(label+' continuation no inner refit',proof['no_inner_refit'] and not proof['training_started_for_reduced_matrix_before'])
            check(label+' reduced columns differ',bool(proof['dropped_columns']) and bool(proof['selected_columns']))
            for name,expected in r['sources'].items():
                check(label+' continuation source '+name,hashlib.sha256((ROOT/r['source_root']/name).read_bytes()).hexdigest()==expected)
        with np.load(directory/'splits.npz') as split: fit,valid=split['fit'],split['valid']
        check(label+' split hash',split_key(fit,valid)==r['split_key'])
        check(label+' split disjoint',not bool(set(fit)&set(valid)))
        if task=='multiclass':
            g=pd.util.hash_pandas_object(bundle.X,index=False).to_numpy()
            check(label+' raw groups disjoint',not bool(set(g[fit])&set(g[valid])))
        else:
            check(label+' time directed',bundle.X.iloc[fit]['Year_Factor'].max()<bundle.X.iloc[valid]['Year_Factor'].min())
        for file in ['report.md','evidence.json','metrics.json','request.json']:
            check(label+' '+file,(directory/file).is_file())
        number=r['candidate']['number']
        if r['status']=='completed' and number in [36,39]:
            with np.load(directory/'training-weights.npz') as weights:
                w=weights['weights']
                check(label+' weight positions',np.array_equal(weights['positions'],fit))
                check(label+' finite positive weights',len(w)==len(fit) and np.isfinite(w).all() and np.all(w>0))
                check(label+' normalized weights',abs(w.mean()-1)<1e-10)
                if number==36:
                    years=bundle.X.iloc[fit]['Year_Factor'].to_numpy(); expected=2.**(years-years.max()); expected/=expected.mean()
                else:
                    expected=np.where(weights['p_true']>=.05,1+2*(1-weights['p_true']),1.); expected/=expected.mean()
                check(label+' exact weight formula',np.allclose(w,expected,atol=1e-12))
        if r['status']=='completed' and number==38:
            audit=json.loads((directory/'preprocessing-audit.json').read_text(encoding='utf-8'))
            synthetic=pd.read_csv(directory/'accepted-synthetic.csv.gz')
            with np.load(directory/'synthetic-selection.npz') as artifact:
                check(label+' synthetic original positions',np.array_equal(artifact['original_fit_positions'],fit))
                check(label+' accepted mask count',int(artifact['accepted'].sum())==len(synthetic))
            counts=bundle.y.iloc[fit].value_counts()
            generated=synthetic['__synthetic_target__'].value_counts()
            check(label+' synthetic perclass cap',all(n<=min(counts[k],500) for k,n in generated.items()))
            for column in audit['domains']['rounded_integer_fields']:
                v=synthetic[column].dropna().to_numpy()
                check(label+' integer '+column,np.allclose(v,np.round(v),atol=1e-8))
            joint=[c for c in ['proto','service','state'] if c in synthetic]
            known=set(map(tuple,bundle.X.iloc[fit][joint].astype(str).to_numpy()))
            check(label+' category joint support',all(tuple(row) in known for row in synthetic[joint].astype(str).to_numpy()))
        if r['status']=='completed':
            with np.load(directory/'validation_predictions.npz',allow_pickle=True) as artifact:
                check(label+' exact validation order',np.array_equal(artifact['positions'],valid))
                truth=bundle.y.iloc[valid].to_numpy(); pred=artifact['prediction']
                check(label+' unchanged truth',np.array_equal(artifact['y'].astype(str),truth.astype(str)))
                check(label+' prediction shape',pred.shape==(len(valid),))
                metric=float(np.sqrt(mean_squared_error(truth,pred))) if task=='regression' else float(f1_score(truth,pred,average='macro',zero_division=0))
                check(label+' metric recomputation',abs(metric-r['score'])<1e-10)
                if task=='multiclass':
                    p=artifact['probabilities']; classes=artifact['classes']
                    check(label+' probability shape',p.shape==(len(valid),len(classes)))
                    check(label+' finite probabilities',np.isfinite(p).all() and np.all(p>=0))
                    check(label+' normalized probabilities',np.allclose(p.sum(axis=1),1,atol=2e-5))
                    check(label+' labels in classes',np.isin(pred,classes).all())
                else: check(label+' finite prediction',np.isfinite(pred).all())
            check(label+' audit',(directory/'preprocessing-audit.json').exists())
        previous=REFERENCES[task]
        improvement=(previous-r['score'] if task=='regression' else r['score']-previous) if r['status']=='completed' else None
        table.append({'hypothesis':r['candidate']['hypothesis'],'task':task,'metric':r.get('metric'),'score':r.get('score'),
            'previous_best':previous,'improvement':improvement,'status':r['status'],'conclusion':r['conclusion'],
            'seconds':r.get('seconds'),'directory':r['directory'],'error':r.get('error','')})
    attempts_path=SUITE/'model-fit-attempts.jsonl'
    attempts=[json.loads(line) for line in attempts_path.read_text(encoding='utf-8').splitlines()] if attempts_path.exists() else []
    outcomes=[json.loads(line) for line in (SUITE/'model-fit-outcomes.jsonl').read_text(encoding='utf-8').splitlines()]
    check('all model fingerprints unique',len({a['fingerprint'] for a in attempts})==len(attempts))
    check('all model attempts finished',{a['fingerprint'] for a in attempts}=={a['fingerprint'] for a in outcomes})
    check('all model fits read ledger',all(len(a['ledger_sha256'])==64 for a in attempts))
    continued=[a for a in attempts if '/guard-continuation' in str(a.get('hypothesis',''))]
    check('technical continuations only new outer fits',len(continued)==sum(bool(r.get('continuation_of')) for r in rows)
          and all(a['rows'] in [35000,70529] for a in continued))
    for path in (SUITE/'cache').glob('*/fold-*.npz'):
        if '-model' in path.name: continue
        with np.load(path,allow_pickle=False) as data:
            if not {'fit_positions','valid_positions'}.issubset(data.files): continue
            fit,valid=data['fit_positions'],data['valid_positions']
            metadata=json.loads((path.parent/'metadata.json').read_text()) if (path.parent/'metadata.json').exists() else {}
            # Raw positions refer to original full train and metadata may lack task.
            task='regression' if metadata.get('selection_source')=='training_temporal_oof' else 'multiclass'
            bundle=bundles[task]
            check(str(path.relative_to(SUITE))+' rowdisjoint',not bool(set(fit)&set(valid)))
            if task=='regression':
                check(str(path.relative_to(SUITE))+' temporal',bundle.X.iloc[fit]['Year_Factor'].max()<bundle.X.iloc[valid]['Year_Factor'].min())
            else:
                g=pd.util.hash_pandas_object(bundle.X,index=False).to_numpy()
                check(str(path.relative_to(SUITE))+' groupdisjoint',not bool(set(g[fit])&set(g[valid])))
    import sklearn,xgboost,lightgbm,imblearn
    verification={'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks_passed':sum(c['passed'] for c in checks),
                  'checks_failed':sum(not c['passed'] for c in checks),'checks':checks,'model_fit_attempts':len(attempts),
                  'completed_evaluations':sum(r['status']=='completed' for r in rows),'technical_stops':sum(r['status']!='completed' for r in rows),
                  'pending_cases':sum(r['status']!='completed' for r in latest.values()),
                  'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'packages':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,
                              'lightgbm':lightgbm.__version__,'imbalanced-learn':imblearn.__version__}}
    save(SUITE/'independent-verification.json',verification)
    save(SUITE/'summary.json',{'status':'completed' if verification['status']=='passed' else 'verification_failed',
        'planned_evaluations':23,'attempted':len(rows),'completed':verification['completed_evaluations'],
        'unique_hypotheses':20,'technical_stops':verification['technical_stops'],'pending_cases':verification['pending_cases'],
        'baseline_refits':0,'model_fit_attempts':len(attempts),'verification':verification['status']})
    with (SUITE.parent/'twenty-hypotheses-comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    lines=['# Двадцать новых гипотез HYP-022…041 — результаты','',
        f'Всего {len(rows)} попыток / {len({r["candidate"]["number"] for r in rows})} гипотез; оценок с метриками {verification["completed_evaluations"]}, технических остановок {verification["technical_stops"]}; случаев без завершённой метрики {verification["pending_cases"]}.',
        f'Независимая сверка {verification["status"]}: {verification["checks_passed"]} успешных / {verification["checks_failed"]} ошибок. Обучение при сверке не запускалось.','',
        '| Гипотеза | Задача | Метрика | Значение | Прежний лучший | Улучшение | Исход |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in table:
        val='—' if r['score'] is None else f'{r["score"]:.9f}'
        delta='—' if r['improvement'] is None else f'{r["improvement"]:+.9f}'
        lines.append(f'| [{r["hypothesis"]}]({Path(r["directory"]).name}/report.md) | {r["task"]} | {r["metric"]} | {val} | {r["previous_best"]:.9f} | {delta} | {r["status"]} |')
    lines+=['','Улучшение положительно для обеих метрик: прежний−новый RMSE, новый−прежний F1.','']
    for task in ['regression','multiclass']:
        successful=[r for r in table if r['task']==task and r['status']=='completed']
        if successful:
            best=min(successful,key=lambda r:r['score']) if task=='regression' else max(successful,key=lambda r:r['score'])
            lines.append(f'Лучший новый {task}: **{best["hypothesis"]}, {best["metric"]}={best["score"]:.12f}**, улучшение {best["improvement"]:+.12f}.')
    lines+=['','Восстановление только относительно full-v1 (RMSE43.837201829105254 / F1.5637871673383791) не означает новый рекорд.',
        'Преобразования и imputation обучаются только на соответствующем training subset (в итоговой модели на outer-fit); learned selection/thresholds/коррекции выбираются на inner. Веса H36 — фиксированная формула по training-годам. Исходный outer уже использовался и не является независимым подтверждением. Реальный платформенный score не оценивался. Точный эффект отдельных шагов старого HYP-021 здесь не устанавливается.',
        'Два первоначальных HYP-028 могли остановиться до нового outer-fit из-за чрезмерно строгого guard. Технические продолжения используют готовые inner-модели и permutation importance; исходные ошибки сохранены, новыми гипотезами или повторными baseline fits не считаются.',
        'HYP-024 включает групповой imputer и missing indicators, HYP-022 меняет и сохранение строк, и clipping; эффект отдельного элемента этих составных представлений не установлен.',
        '','История/карточки/results обновлены после каждого запуска, отрицательные исходы сохранены. Старые outer-baseline predictions переиспользовались; модельные fits новой серии и внутренние cross-fit отражены в журнале model-fit-attempts.jsonl.','',
        '[Независимая сверка](series-twenty-hypotheses-v1/independent-verification.json), [CSV](twenty-hypotheses-comparison.csv), [план до fit](series-twenty-hypotheses-v1/frozen-plan.json), [реестр](series-twenty-hypotheses-v1/records.json).']
    (SUITE.parent/'TWENTY_HYPOTHESES_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k!='checks'},ensure_ascii=False))
    return 0 if verification['status']=='passed' else 1

if __name__=='__main__': raise SystemExit(main())
