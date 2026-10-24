# HYP-086: causal state-space filtering истории энергопотребления

- Дата:2026-10-24, Москва.
- Связь: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: реализация и unit-проверки готовы; **real-data model fit и scientific score при передаче отсутствуют**.
- Основание: прямое поручение координатора реализовать новый conditional history filter; регистрация/sourcefreeze/научные запуски выполняются координатором.

## Утверждение и новизна

Оценка скрытого стабильного уровня здания по всей строго прошлой истории может уменьшить шум одиночного донора H70. Модель локального уровня фильтрует log consumption после поправки на известный rating effect и учитывает изменяющуюся posterior uncertainty. H69 усредняла исходные targets; H70 переносила latest donor; H83 обучала monotone within-profile ratio curve. Здесь отдельная последовательная Gaussian state-space recursion, а не повтор этих правил или H71 global rating normalization +ExtraTrees.

## Единственный конфиг до realfit

`RatingStateSpaceHistory(process_variance=.01, measurement_variance=.04, alpha=.95, support_shrinkage=.25, age_decay=.25, beta_cap=2.)`.

API: `fit(X,y,base_predictor=None,transport_state=None)`, `predict(X)`, `configuration()`, `audit_`, `last_prediction_audit_`. В опыте base_predictor должен быть совместимым **сохранённым H70high**, transport_state содержит его замороженные rating_beta/group_rating_betas. H70 не fit заново для получения state. В production wrapper допустим единственный H70 fit с последующей передачей predictor/state фильтру. Нет переданного baseline →fit mean только как технический standalone fallback; нет state →beta0.

1. Profile key ровно State_Factor/building_class/facility_type/floor_area/year_built с canonical representation H70. Любой sameyear collision отвергает весь профиль. ID и будущие query observations не участвуют в matching; пропуски key не wildcard.
2. Для каждого retained профиля сортируются training observations. Используются только y>0 и finite rating; глобальный y должен быть finite и≥0. Остальные строки остаются частью training schema/ambiguity проверки, но не являются Gaussian log-observations. Filtering не удаляет validation/query строки.
3. `q=ndtri(clip((100.5−rating)/100,.005,.995))`, `z=log(y)−beta_facility*q`. Frozen facility beta при наличии, иначе frozen global beta; cap2 проверяется, beta не переоценивается.
4. Фиксированная local-level модель: latent state random walk с variance Q=.01 за единицу Year_Factor, observation noise R=.04. Init first usable `mu=z1,P=R`. Для следующего usable года gap: `Ppred=P+Q*gap`, `K=Ppred/(Ppred+R)`, `mu=mu+K*(z−mu)`, `P=(1−K)*Ppred`. Missing/zero наблюдения пропускаются; gap считается от предыдущего **usable** наблюдения.
5. Fit сохраняет causal posterior на каждом usable training-году. Query выбирает только последний posteriorYear<queryYear. Более поздний training posterior не влияет на более ранний query. Число support=count usable observations до выбранного posterior.
6. Только finite query rating: conditional lognormal mean=`exp(mu+beta*q_query+.5*(P+Q*age+R))`, age=queryYear−posteriorYear. Это математическое ожидание в выбранной Gaussian log-модели, а не безусловно несмещённая оценка реальных данных. Weight=`.95*support/(support+.25)*exp(−.25*max(0,age−1))`; output=weight*filtered_mean+(1−weight)*cached H70 prediction.
7. Missing query rating, отсутствие usable strict-past history и ambiguous profile →H70 prediction остаётся точно прежним. Query не записывается в память. Невалидный baseline/state/неfinite output отклоняется.

Q/R выбраны фиксированно до realfit: 0.1 лог-единицы process standard deviation и0.2 measurement standard deviation выражают консервативное предпочтение сглаживания истории. Их пригодность — предположение для научной проверки. Робастная оценка Q/R по short-panel first/second differences не используется: при коротких и нерегулярных рядах она добавила бы неустойчивую идентификацию и дополнительные choices. Ни median-square Gaussian correction, ни validation target для Q/R не вычислялись; hyperparameter search отсутствует.

## Данные и критерий до запуска

Общий regression SHA256 из зарегистрированного provenance: `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`. Inner Year<4→4, compatible H70high RMSE30.663255379972725 / H69RMSE42.46005247026025; frozen H70high predictions/state переиспользуются. Если фильтр не улучшает H70, inherited baseline gain не выдаётся за пользу нового механизма.

Outer Year<5→5 только при отдельном допуске после выбора inner; сохранённый H70high34.318489927675174 / H69RMSE39.39219382766771. Цель SPEC011 outerRMSE≤31.51375506213417, меньший результат отмечается честно. Уже просмотренный outer остаётся разведочным; source/config/data/split фиксируются до fit, max900сек, один конфиг. Координатор сохраняет каждый фактический исход отдельно в results, включая технические ошибки.

Source SHA256 при передаче: `35f5008aaefc95182b883d65f64c412182e9d78b91adc11c57e48eec991ed72b`.

## Unit-проверки и ограничения

`.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_sprint_state_space.py -v`: **7/7 green**,0.035сек. Ручные ожидания проверяют K=5/9,P=1/45 для gap1; K=.6,P=.024 для gap2; predictive lognormal variance; rating normalization/restore с независимо указанными normal quantiles. Также sameyear/future exclusion, collision veto, missingquery unchanged, positive-y/knownrating gates, frozen state copy/no estimation, query independence, permutation/pickle, invalid y/state/baseline.

Полный `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q`: **425 tests, OK**,10.570сек; три известных PyparsingDeprecationWarning от matplotlib. Это unit-проверка, не scientific evaluation.

Логnormal Gaussian assumption и fixed Q/R могут усилить bias/heavy tails; это не гарантия RMSE улучшения. Missingrating subset вообще не меняется. Реальных fit0; вывод **Не оценивалась** до отдельного запуска координатора.

SPEC011 result 2026-10-24T02:23:18.967963+03:00 HYP-086/inner/default: completed rmse=31.12117379943106 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-086-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
