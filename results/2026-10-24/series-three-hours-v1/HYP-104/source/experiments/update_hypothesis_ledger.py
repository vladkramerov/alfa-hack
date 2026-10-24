"""Render the single human-readable hypothesis memory; no model fitting."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def clean(value): return str(value if value is not None else '—').replace('|','/').replace('\n',' ')


def main():
    historical=json.loads((ROOT/'results/history/2026-10-21/historical-experiments.json').read_text(encoding='utf-8'))
    text=['# Единый журнал гипотез','',
          'Обновлён: 2026-10-23. **Перед КАЖДОЙ гипотезой прочитать этот файл целиком; проверенное не повторять.**',
          'Новую гипотезу и критерий записать до запуска в docs/hypotheses/, затем внести сюда статус «Запланирована»; после проверки сразу сохранить исход и ссылку на отдельный results/. Новый seed/имя/техническая правка не создают новую гипотезу.',
          'Сравнивать данные (SHA256), признаки/предобработку, модель/параметры, split и смысл изменения. Технический full refit для итоговых предсказаний не является новой проверкой и не даёт новой метрики.',
          'На платформе эта же память — постоянный hypotheses.json рядом с task.md: агент читает его непосредственно перед каждым допуском кандидата, записывает started до fit и результат после. История разделена по SHA данных/split/метрике и читается при следующем запуске.',
          '', '## Данные и протоколы','',
          '- Регрессия train SHA256: `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`; outer: Year_Factor<5→5 (35000/18308).',
          '- Классификация train SHA256: `171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`; outer: SGKF5 first fold, seed42, группировка одинаковых сырых признаков (70529/17141).',
          '- Старые классификационные random-split scores несопоставимы с grouped. Остальные SHA и параметры: [история](results/history/2026-10-21/historical-experiments.json), evidence.json/metrics.json каждого run.',
          '', '## Уже проверенные направления — закрыты для повторов','',
          '| ID | Изменение | Статус / вывод | Подробности |','| --- | --- | --- | --- |']
    directions=[('HYP-001','Парные Pearson/Spearman корреляции, порог0.95','Проверена; улучшения зависят от split'),
                ('HYP-002','VIF>10, joint collinearity','Проверена; сильное ухудшение regression outer'),
                ('HYP-003','StandardScaler после отбора','Проверена; систематической пользы деревьям нет'),
                ('HYP-004','Yeo–Johnson для непрерывных столбцов','Проверена; нормальность всех данных не гарантирована'),
                ('HYP-005','Полный cumulative preprocessing + engineering','Проверена; не лучший outer'),
                ('HYP-006','CatBoost l2_leaf_reg1/3/10/30','Проверена; inner выбрал1, не гарантирует лучший outer'),
                ('HYP-007','L1/L2: Lasso/Ridge и LogisticRegression','Проверена; L1 лучше в проверенной сетке, classification разница мала'),
                ('HYP-008','Engineering без pruning/scaling/Gaussian','Проверена; classification RF0.577365824'),
                ('HYP-009','Нативный категориальный LightGBM500/.05/31/20/lambda1','Проверена; новый лучший grouped classifier0.583855959'),
                ('HYP-010','ExtraTrees200/depth24/leaf2/features.8','Проверена; regression43.567567738, class0.580495276')]
    for identifier,change,status in directions:
        text.append(f'| {identifier} | {change} | {status} | [гипотеза](docs/hypotheses/{identifier}.md) |')
    text+=['','Также проверены в старой истории: линейные/RF/CatBoost baselines, Optuna16 trials, balanced RF, нативные категории CatBoost, missing flags, year_built0→NaN, четыре flow-ratios с+1, равные ансамбли и другие splits. Их не повторять. Все записи ниже импортированы, без новых fit.',
           '', '## Исторические оценки — 33 записи и 16 Optuna trials','',
           '| ID | Задача | Изменение / параметры | Split | Метрика | Источник |','| --- | --- | --- | --- | --- | --- |']
    for r in historical['sheets']['Запуски']:
        metric=r.get('RMSE ↓') if r.get('RMSE ↓') is not None else r.get('Macro F1 ↑')
        text.append('| '+' | '.join(clean(x) for x in [r.get('ID'),r.get('Задача'),str(r.get('Что изменили'))+'; '+str(r.get('Основные параметры')),r.get('Разделение'),metric,'[JSON](results/history/2026-10-21/historical-experiments.json)'])+' |')
    text+=['','Optuna: все16 параметрических попыток сохранены в разделе «Optuna» [истории](results/history/2026-10-21/historical-experiments.json); новое название trial не разрешает повтор.',
           '', '## SPEC-002: все завершённые и неуспешные оценки','',
           '| Run | Конфигурация | Часть | Метрики | Статус | Результат |','| --- | --- | --- | --- | --- | --- |']
    for path in sorted((ROOT/'results/2026-10-23').glob('HYP-*/metrics.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        if 'run_id' not in r: continue
        if r.get('stage') not in ['inner','outer']: continue
        text.append('| '+' | '.join(clean(x) for x in [r['run_id'],json.dumps(r.get('config'),ensure_ascii=False),r.get('stage'),json.dumps(r.get('scores'),ensure_ascii=False),r.get('status'),f'[отчёт]({path.parent.relative_to(ROOT).as_posix()}/report.md)'])+' |')
    text+=['','## SPEC-003: только новые семейства — четыре проверки','',
           '| Run | Данные | Модель и параметры | Метрика | Статус | Результат |','| --- | --- | --- | --- | --- | --- |']
    for r in json.loads((ROOT/'results/2026-10-23/new-family-records.json').read_text(encoding='utf-8')):
        params='500trees, lr.05, leaves31, min_child20, lambda1; native categories' if r['candidate']['family']=='lightgbm' else '200trees, depth24, min_leaf2, features.8; median/OHE'
        text.append('| '+' | '.join(clean(x) for x in [Path(r['directory']).name,r['task'],r['candidate']['family']+': '+params,f"{r['metric']}={r['score']:.12f}",r['status'],f"[отчёт]({Path(r['directory']).as_posix()}/report.md)"])+ ' |')
    text+=['','## Текущий выбор','',
           'Регрессия: исторический CatBoost H4-zero-year, RMSE43.004638647931; классификация: новый LightGBM, grouped Macro F1 0.583855959368. Старые проверки не переобучались на validation; выполнен только технический full refit победителей.',
           '[Новые выводы](results/2026-10-23/NEW_EXPERIMENT_REPORT.md), [основная серия](results/2026-10-23/EXPERIMENT_REPORT.md), [журнал артефактов](results/README.md).',
           '', '## Следующие гипотезы','',
           'Новых запланированных запусков сейчас нет. Перед следующим fit прочитать журнал, записать конкретный новый механизм, данные/разбиение/параметры/критерий и убедиться, что он отсутствует выше.']
    (ROOT/'HYPOTHESES.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__': main()
