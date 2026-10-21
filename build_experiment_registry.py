"""Build a complete, reproducible team registry from saved experiment evidence."""
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent


def compact_parameters(m):
    p=m.get('model_parameters',{})
    keys=['n_estimators','max_depth','min_samples_leaf','max_features','class_weight',
          'iterations','depth','learning_rate','l2_leaf_reg','alpha','solver','tol','equal_weights']
    if not p:
        if m['model']=='Ridge':return 'alpha=1; solver=lsqr; tol=1e-6 (восстановлено по коду первого этапа)'
        if m['model']=='LogisticRegression':return 'solver=lbfgs; max_iter=3000; random_state=42 (по коду первого этапа)'
        if m['model']=='LinearRegression':return 'LinearRegression(); настройки по умолчанию, диагностический запуск'
        return 'Не записаны в старом JSON; см. источник и Git'
    return '; '.join(f'{k}={p[k]}' for k in keys if k in p)


def normalize(m,source,run_id):
    scores=m.get('scores',{})
    task=m['task']
    primary='rmse' if task=='regression' else 'macro_f1'
    hypothesis=m.get('hypothesis','')
    change=m.get('change')
    if not change:
        family=m.get('model_family')
        if family=='optuna':change='Итог подбора: сохранена прежняя модель' if m['tuning']['selected']=='incumbent' else 'Итог подбора: выбран кандидат'
        elif m['model']=='Ridge':change='Первый линейный ориентир; Ridge'
        elif m['model']=='LogisticRegression':change='Первый линейный классификатор'
        elif m['model'].startswith('RandomForest'):change='Замена линейной модели на Random Forest'
        elif m['model'].startswith('CatBoost'):change='Замена модели на CatBoost; категории OneHotEncoder'
        else:change=m['model']
    kind=m.get('validation_kind','primary')
    if m.get('audit_kind'):kind=m['audit_kind']
    comparison=m.get('comparison') or {}
    result={'better':'Лучше контроля','worse':'Хуже контроля','same':'Без изменения'}.get(comparison.get('conclusion'),'Исходный ориентир / без парного сравнения')
    if m.get('model_family')=='optuna':result='Подбор не улучшил основную метрику' if m['tuning']['selected']=='incumbent' else 'Подбор улучшил основную метрику'
    if kind=='diagnostic_failure':result='Численно нестабильная OLS; не использовать'
    elif kind=='technical_repeat':result='Технический повтор, не новая гипотеза'
    return dict(id=run_id,task=task,validation_kind=kind,hypothesis=hypothesis,
                model=m['model'],change=change,parameters=compact_parameters(m),
                full_parameters=m.get('model_parameters'),recipe=m.get('recipe'),
                split=m.get('split'),seed=m.get('seed'),n_fit=m.get('n_fit'),n_validation=m.get('n_validation'),
                rmse=scores.get('rmse'),macro_f1=scores.get('macro_f1'),accuracy=scores.get('accuracy'),weighted_f1=scores.get('weighted_f1'),
                metric=primary,value=scores.get(primary),reference=comparison.get('reference'),
                reference_value=comparison.get('reference_value'),gain=comparison.get('gain'),result=result,
                seconds=m.get('seconds'),pipeline_seconds=m.get('pipeline_seconds',m.get('seconds')),
                timing_note='Время усреднения; полная оценка стоимости включает компоненты' if 'component_seconds' in m else 'Весь запуск; дополнительная проверка без финального обучения' if kind=='secondary_sensitivity' else 'Полный запуск с финальным обучением, если не указано иное',
                source_commit=m.get('source_commit'),source_report=source,artifact_path=m.get('artifact_path'),
                input_sha256=m.get('input_sha256'),script_sha256=m.get('script_sha256'),
                submission_created=m.get('submission_created',kind not in ['secondary_sensitivity']),
                tuning=m.get('tuning'),provenance_note=m.get('provenance_note'))


def main():
    reports=ROOT/'reports'
    records=[]
    for path in sorted(reports.rglob('*.json')):
        m=json.loads(path.read_text(encoding='utf-8'))
        if m.get('kind') not in ['local_fixed_pipeline','hypothesis_experiment','archived_experiment']:
            continue
        if 'scores' not in m or 'task' not in m:continue
        run_id=m.get('run_id',path.stem)
        records.append(normalize(m,str(path.relative_to(ROOT)),run_id))
    # Constant predictors were measured inside the first runs, not standalone jobs.
    for task,parent_id in [('regression','regression-20261021-195542'),('multiclass','multiclass-20261021-195652')]:
        parent=next((r for r in records if r['id']==parent_id),None)
        if parent:
            raw=json.loads((ROOT/parent['source_report']).read_text(encoding='utf-8'))
            metric='rmse' if task=='regression' else 'macro_f1'
            r=dict(parent)
            r.update(id='constant-'+task+'-reference',model='DummyRegressor' if task=='regression' else 'DummyClassifier',
                     hypothesis='reference',change='Постоянное среднее train' if task=='regression' else 'Всегда самый частый класс train',
                     parameters='strategy=mean' if task=='regression' else 'strategy=most_frequent',
                     full_parameters={'strategy':'mean' if task=='regression' else 'most_frequent'},
                     value=raw['scores']['dummy_'+metric],rmse=raw['scores']['dummy_rmse'] if task=='regression' else None,
                     macro_f1=raw['scores']['dummy_macro_f1'] if task=='multiclass' else None,accuracy=None,weighted_f1=None,
                     seconds=None,pipeline_seconds=None,result='Простой ориентир, вычислен внутри первого запуска',
                     timing_note='Входит в первый запуск; отдельное время не измерялось',submission_created=False)
            records.append(r)
    # Capture completed local runs that have not yet been published.
    known={r['id'] for r in records}
    for path in sorted((ROOT/'runs').rglob('metrics.json')):
        m=json.loads(path.read_text(encoding='utf-8'))
        run_id=m.get('run_id',path.parent.name)
        if run_id in known:continue
        if 'scores' in m and 'task' in m:
            records.append(normalize(m,str(path.relative_to(ROOT)),run_id));known.add(run_id)
    # Rejected Optuna outer candidates are distinct from the selected final result.
    for parent in list(records):
        t=parent.get('tuning')
        if not t:continue
        r=dict(parent)
        r.update(id=parent['id']+'-candidate',change='Лучший по внутренней проверке кандидат Optuna, до решения о сохранении',
                 parameters=json.dumps(t['candidate_parameters'],ensure_ascii=False,sort_keys=True),full_parameters=t['candidate_parameters'],
                 seconds=None,pipeline_seconds=None,reference=parent['id']+'-incumbent',reference_value=t['outer_incumbent_score'],
                 value=t['outer_candidate_score'],rmse=t['outer_candidate_score'] if parent['task']=='regression' else None,
                 macro_f1=t['outer_candidate_score'] if parent['task']=='multiclass' else None,
                 accuracy=None,weighted_f1=None,submission_created=False,
                 result='Кандидат принят' if t['selected']=='candidate' else 'Кандидат отклонён',
                 timing_note='Оценка входит в родительский запуск; отдельное время не записано')
        if parent['task']=='multiclass':
            saved=ROOT/'runs'/parent['id']/'outer_predictions.csv'
            if saved.exists():
                predictions=pd.read_csv(saved)
                r['accuracy']=float(accuracy_score(predictions.true,predictions.candidate_prediction))
                r['weighted_f1']=float(f1_score(predictions.true,predictions.candidate_prediction,average='weighted'))
        r['gain']=t['outer_incumbent_score']-t['outer_candidate_score'] if parent['task']=='regression' else t['outer_candidate_score']-t['outer_incumbent_score']
        records.append(r)
    ids=[r['id'] for r in records]
    assert len(ids)==len(set(ids)), 'Duplicate registry IDs'
    # Every individual Optuna attempt remains visible, including duplicates.
    trials=[]
    for path in sorted(reports.glob('*-trials.csv')):
        task='regression' if path.name.startswith('regression') else 'multiclass'
        parent=path.name.removesuffix('-trials.csv')
        seen={}
        for row in pd.read_csv(path).to_dict('records'):
            params={k.removeprefix('params_'):None if pd.isna(v) else v for k,v in row.items() if k.startswith('params_')}
            for key in ['iterations','depth','n_estimators','max_depth','min_samples_leaf']:
                if key in params and params[key] is not None:params[key]=int(params[key])
            signature=json.dumps(params,sort_keys=True)
            number=int(row['number'])
            trials.append(dict(id=f'{parent}-trial-{number}',parent=parent,task=task,
                               validation_kind='inner_tuning',trial=number,value=float(row['value']),
                               metric='rmse' if task=='regression' else 'macro_f1',
                               parameters=params,seconds=row.get('user_attrs_seconds'),state=row['state'],
                               duplicate_of=seen.get(signature),source_report=str(path.relative_to(ROOT))))
            seen.setdefault(signature,number)
    order=['constant-regression-reference','constant-multiclass-reference',
           'regression-20261021-195542','multiclass-20261021-195652',
           'regression-forest-20261021-stage2','multiclass-forest-20261021-stage2',
           'regression-boosting-20261021-stage3','multiclass-boosting-20261021-stage3',
           'regression-optuna-20261021-stage4-candidate','regression-optuna-20261021-stage4',
           'multiclass-optuna-20261021-stage4-candidate','multiclass-optuna-20261021-stage4',
           'C00-forest-control','H1-balanced-forest','R00-catboost-control','H2-native-regression',
           'C00-catboost-control','H2-native-multiclass','H3-missing-flags','H4-zero-year','H5-flow-ratios',
           'R00-forest-control','H6-ensemble-regression','H6-ensemble-multiclass',
           'S-regression-control','S-regression-candidate','S-multiclass-control','S-multiclass-candidate']
    order={name:i for i,name in enumerate(order)}
    kinds={'primary':0,'secondary_sensitivity':1,'diagnostic_failure':2,'technical_repeat':3}
    records.sort(key=lambda r:(kinds.get(r['validation_kind'],9),order.get(r['id'],999),r['id']))
    primary=[r for r in records if r['validation_kind']=='primary' and r['value'] is not None]
    best={task:(min if task=='regression' else max)([r for r in primary if r['task']==task],key=lambda r:r['value']) for task in ['regression','multiclass']}
    # Prefer actual candidate experiments over tied repetitions in the leader display.
    for task in best:
        tied=[r for r in primary if r['task']==task and abs(r['value']-best[task]['value'])<1e-10]
        preferred=[r for r in tied if r['id'].startswith('H')]
        old_id='regression-boosting-20261021-stage3' if task=='regression' else 'multiclass-forest-20261021-stage2'
        incumbent=[r for r in tied if r['id']==old_id]
        if incumbent:best[task]=incumbent[0]
        elif preferred:best[task]=preferred[0]
    hypothesis_rows=[r for r in records if r['id'].startswith('H') and r['validation_kind']=='primary']
    secondary=[r for r in records if r['validation_kind']=='secondary_sensitivity']
    recommendations={}
    for task,old_id in [('regression','regression-boosting-20261021-stage3'),('multiclass','multiclass-forest-20261021-stage2')]:
        old=next(r for r in records if r['id']==old_id)
        check=next((r for r in secondary if r['task']==task and r['id'].endswith('candidate')),None)
        winner=best[task]
        primary_improves=winner['value']<old['value']-1e-10 if task=='regression' else winner['value']>old['value']+1e-10
        recommendations[task]=winner['id'] if primary_improves and check and (check.get('gain') or 0)>0 else old_id
    catalog=dict(kind='experiment_registry',date='2026-10-21',records=records,optuna_trials=trials,
                 best_primary={k:v['id'] for k,v in best.items()},recommendations=recommendations,hypotheses=[r['id'] for r in hypothesis_rows],
                 notes=['Все значения измерены локально. Ответы скрытого test неизвестны.',
                        'Внутренние попытки Optuna, основные проверки, диагностика и дополнительные split разделены.',
                        'Один запуск включает валидацию и финальное переобучение; отдельный fit без метрики не считается новым экспериментом.',
                        'Все завершённые запуски, обнаруженные в reports и runs, включены; незафиксированные в файлах действия восстановить нельзя.'])
    (reports/'experiment_registry.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    columns=['id','task','validation_kind','model','change','parameters','split','seed','rmse','macro_f1','accuracy','weighted_f1','reference','reference_value','gain','result','seconds','pipeline_seconds','source_commit','source_report']
    with (reports/'EXPERIMENTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        headers=['ID','Задача','Тип проверки','Модель','Что изменили','Параметры','Разделение','Seed','RMSE','Macro F1','Accuracy','Weighted F1','Контроль','Значение контроля','Улучшение','Вывод','Время запуска, с','Время с компонентами, с','Commit','Источник']
        writer.writerow(headers)
        writer.writerows([[r.get(c) for c in columns] for r in records])
    def text(v):return '—' if v is None else f'{v:.6f}' if isinstance(v,float) else str(v).replace('|','/').replace('\n',' ')
    lines=['# Все эксперименты команды','','Дата: 21 октября 2026. Источник — сохранённые метрики, конфигурации и истории запусков.',
           f'В журнале {len(records)} оценок экспериментов и {len(trials)} внутренних попыток Optuna. Отдельное финальное переобучение включено во время родительского запуска.','',
           'RMSE: ниже лучше. Macro F1: выше лучше, каждый класс имеет одинаковый вес. Accuracy — доля верных ответов; Weighted F1 учитывает размеры классов.', '', '## Лучшие результаты основной валидации','', '| Задача | Запуск | Модель | Метрика |','|---|---|---|---:|']
    for task,r in best.items():lines.append(f'| {task} | {r["id"]} | {r["model"]} | {r["metric"]}={r["value"]:.6f} |')
    lines+=['','Ориентиры для дальнейшей работы с учётом дополнительной проверки: '+', '.join(k+' — '+v for k,v in recommendations.items())+'. Малый основной прирост не считается устойчивым, если на другом split знак изменения обратный.']
    for kind,title in [('primary','Основная валидация'),('secondary_sensitivity','Дополнительные разделения'),('diagnostic_failure','Неудачная диагностика'),('technical_repeat','Технические повторы')]:
        group=[r for r in records if r['validation_kind']==kind]
        if not group:continue
        lines+=['',f'## {title}','','| ID | Задача / модель | Что проверили | Параметры | Разделение | RMSE | Macro F1 | Accuracy | Weighted F1 | Итог | Секунды |','|---|---|---|---|---|---:|---:|---:|---:|---|---:|']
        for r in group:
            link=f"[{r['id']}]({r['source_report'].removeprefix('reports/')})" if r['source_report'].startswith('reports/') else r['id']
            cells=[link,r['task']+' / '+r['model'],r['change'],r['parameters'],r['split'],r['rmse'],r['macro_f1'],r['accuracy'],r['weighted_f1'],r['result'],r['pipeline_seconds']]
            lines.append('| '+' | '.join(map(text,cells))+' |')
    lines+=['','## Все внутренние попытки Optuna','','Их метрики рассчитаны на внутренних split и не сравниваются напрямую с основной таблицей.','','| ID | Задача | Параметры | Метрика | Секунды | Повтор trial |','|---|---|---|---:|---:|---|']
    for r in trials:lines.append('| '+' | '.join(map(text,[r['id'],r['task'],json.dumps(r['parameters'],ensure_ascii=False),r['value'],r['seconds'],r['duplicate_of']]))+' |')
    lines+=['','Полные источники, commit, SHA-256 и конфигурации — в experiment_registry.json; плоский экспорт — EXPERIMENTS.csv. Технические повторы не являются независимыми подтверждениями.','']
    (reports/'EXPERIMENTS.md').write_text('\n'.join(lines),encoding='utf-8')
    lines=['# Итоги последовательной проверки гипотез','','Изменения проверены по одному. Ансамбли используют исходные модели с фиксированными весами 50/50.','','| Гипотеза | Задача | Изменение | Контроль | Контроль: метрика | Новая метрика | Улучшение | Итог |','|---|---|---|---|---:|---:|---:|---|']
    for r in hypothesis_rows:lines.append('| '+' | '.join(map(text,[r['hypothesis'],r['task'],r['change'],r['reference'],r['reference_value'],r['value'],r['gain'],r['result']]))+' |')
    lines+=['','Положительное улучшение означает лучший результат относительно указанного контроля, а не обязательно лучше всех моделей.','','## Проверка чувствительности','','Дополнительная регрессия использует год 4 (он уже применялся в Optuna); классификация — seed=2026. Это проверка чувствительности, не независимый скрытый тест. Кандидаты выбраны до просмотра этих результатов.','','| ID | Задача | Модель | Split | Значение | Контроль | Изменение к контролю |','|---|---|---|---|---:|---|---:|']
    for r in secondary:lines.append('| '+' | '.join(map(text,[r['id'],r['task'],r['model'],r['split'],r['value'],r['reference'],r['gain']]))+' |')
    lines+=['','Для дальнейшей работы: регрессия — обработка нулевого года постройки, классификация — прежний лес без отношений трафика. Очень малый прирост отношений на основной выборке не перенёсся на дополнительную.', '', 'Все подробности: [общий журнал](EXPERIMENTS.md). Новые комбинации признаков друг с другом не проверялись: складывать их эффекты без отдельного эксперимента нельзя.','']
    (reports/'HYPOTHESES.md').write_text('\n'.join(lines),encoding='utf-8')
    readme=['# Журнал экспериментов команды','',
            '[Итоги гипотез](HYPOTHESES.md) — что помогло и что не помогло.',
            '[Все эксперименты](EXPERIMENTS.md) — модели, изменения, настройки, метрики и выводы.',
            '[Excel с фильтрами](EXPERIMENTS.xlsx) · [CSV](EXPERIMENTS.csv) · [Полные данные JSON](experiment_registry.json)','',
            f'Учтено {len(records)} оценок экспериментов и {len(trials)} внутренних попыток Optuna.','',
            '| Задача | Лучший основной результат | Запуск |','|---|---:|---|']
    for task,r in best.items():readme.append(f'| {task} | {r["metric"]}={r["value"]:.6f} | {r["id"]} |')
    readme+=['','RMSE меньше — лучше. Macro F1 больше — лучше. Метрики скрытого test неизвестны.',
             'Основная валидация, внутренний подбор, дополнительная проверка и технические повторы разделены.',
             'Для обновления CSV/Markdown/JSON: python build_experiment_registry.py. Excel — снимок на дату сборки. Для публикации на GitHub нужны commit и push в sergey_branch.','']
    (reports/'README.md').write_text('\n'.join(readme),encoding='utf-8')
    print(f'Registry: {len(records)} experiment evaluations, {len(trials)} Optuna attempts; best:',catalog['best_primary'])


if __name__=='__main__':main()
