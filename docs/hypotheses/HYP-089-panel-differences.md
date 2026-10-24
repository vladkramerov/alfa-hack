# HYP-089: original-unit within-entity differences Ridge

- Дата:2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: реализация и синтетические проверки готовы; **реальных model fits / scientific scores при передаче нет**.
- Основание: прямое поручение координатора после отрицательного H86inner реализовать один фиксированный within-entity Δy/weather transport. Координатор регистрирует ID/source/data/config/split до realfit и сохраняет исход отдельно.

## Утверждение и новизна

Линейная модель изменения энергопотребления по изменениям погоды/рейтинга может переносить прошлое потребление стабильнее, чем direct-y CatBoost H76. Target здесь `current_y−donor_y` для соседних строго прошлых записей одного профиля. В H56 модель обучалась на остатке старого baseline; здесь baseline в training target не участвует. H7 Ridge на исходных абсолютных признаках не является этой моделью: изменены единица обучения, target и представление входа. H70 rating beta, H83 monotone ratio, H86 Gaussian state filtering не переобучаются и не используются как новые learned components.

## Один зафиксированный конфиг до realfit

`PanelDifferenceRegressor(ridge_alpha=100., alpha=.95, minimum_pairs=32, support_shrinkage=.25, age_decay=.25, log_change_cap=1.5, threads=4)`.

API `fit(X,y,base_predictor=None)`, `predict(X)`, `configuration()`, `audit_`, `last_prediction_audit_`. Реальный baseline — совместимый сохранённый H69. Метод не вызывает baseline.fit; без predictor технический fallback=fit mean.

1. Точный canonical key State_Factor/building_class/facility_type/floor_area/year_built, как H70; любой sameyear collision отвергает весь профиль. ID/target/label исключены; отсутствующий key не wildcard. Eligible pairs — все соседние уникальные training-годы одного retained профиля, donorYear<currentYear. y конечное и≥0; zero/missing-rating pairs сохраняются.
2. Features: donor_y, age, `donor_y*(q(current_rating)−q(donor_rating))`, donor/current rating-missing flags и current facility_type. `q=ndtri(clip((100.5−rating)/100,.005,.995))`. Если любой rating missing, interaction=0 с явными flags; rating не импутируется искусственным score.
3. Для всех numeric исходных X вне key/Year_Factor/rating/id/target/label добавляется только `current_value−donor_value`. Это погодные изменения и ELEVATION delta. Current absolute raw X и donor weather levels не входят в модель. Numeric columns выбираются по fit schema; query не определяет набор признаков.
4. Numeric pipeline: SimpleImputer(strategy=median,keep_empty_features=True) →StandardScaler; fit только по training-pair features. Facility отдельно OneHotEncoder(handle_unknown=ignore,sparse_output=False), без scaling OHE. Inf→NaN. Unknown query facility допустима, в том числе если её fit profile имел только одну строку и потому не попал в training pairs.
5. `Ridge(alpha=100,fit_intercept=True,solver='lsqr',tol=1e-6)` на исходном `Δy`; веса единичные, target normalization/log/smearing отсутствуют. Scaler обучается до Ridge только на этих training pairs. Матрица небольшая dense; threadpoolctl ограничивает fit/predict максимум4 потоками. Минимум32 пары; иначе model=None и query baseline без изменения.
6. Query donor — latest строго прошлый training-row retained профиля. `transport=clip(donor_y+delta_model,donor_y*exp(−1.5),donor_y*exp(1.5))`; при donor_y0 оба bounds0. Output=`weight*transport+(1−weight)*H69_prediction`, weight=`.95*past_support/(past_support+.25)*exp(−.25*max(0,age−1))`. Support включает все strict-past строки профиля, не только pair rows. No donor/ambiguous/no model →baseline.
7. Query observations никогда не добавляются в память; donor target берётся только из fit. Невалидный y/baseline/nonfinite output отклоняется. Validation/query строки не удаляются.

Умеренная Ridge regularization100 фиксирована до realfit для коррелированных погодных изменений и небольшой pair sample; это инженерное предположение, не найденный optimum. Unknown/missing handling соответствует train-only fit, других вариантов конфигурации не проверяется.

## Данные и критерий до запуска

Regression train SHA256 из зарегистрированного provenance `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`. Inner Year<4→4: H69RMSE42.46005247026025, H70high30.663255379972725, H83RMSE30.651578052812432 из сохранённых результатов. Для финального переноса нужен strict gain относительно лучшего совместимого кандидата, а не унаследованный baseline gain. Сначала inner; никаких повторных H69/H70 scientific fits.

Outer Year<5→5 только по решению координатора после inner selection. Совместимые готовые H69RMSE39.39219382766771/H70high34.318489927675174; цель SPEC011RMSE≤31.51375506213417. Outer уже открыт и разведочный. Один конфиг, max900сек, отдельные outcomes/predictions/report и source/config/data/split SHA; post-score hypersearch нет.

Source SHA256 при передаче: `c5a5cf8a7a896c3f7b529fcff00f45641059c803d5cfcc85860f22c552d48c9a`.

## Синтетическая проверка

`.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_sprint_differences.py -v`: **6/6 green**,0.989сек. Независимый ручной Ridge fixture: donor_y=[10,20] →scaled[−1,+1], delta_y=[10,20], coefficient=10/(2+100)=5/51, intercept15; итоговый predictor проверен в исходных единицах. Дополнительно manual delta features/normal quantiles, strict-past/sameyear/future и collision veto, minpairs fallback, missing ratings/weather, unseen facility, query independence, permutation/pickle, zero donor, multiplicative upper/lower bounds и invalid y/baseline.

Первый полный `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q` на параллельно изменяемом общем дереве:443 tests,4failures; все4 в test_sprint_missing_expert setUp (`H90 missing-rating expert absent`), так как другой агент ещё не создал sprint_missing_expert.py. Остальные439 tests, включая все6 H89, прошли; причина не относится к этому модулю. Известные три PyparsingDeprecationWarning от matplotlib не ошибки.

После появления модуля H90 повтор того же полного unit-suite: **443 tests, OK**,10.094сек. Научных model fits при этих unit checks нет.

Это unit fit на синтетике, scientific evaluation не выполнена. Реальные результаты после freeze/run сохраняет координатор; **Не оценивалась** при передаче.

SPEC011 result 2026-10-24T02:28:38.822395+03:00 HYP-089/inner/default: completed rmse=31.05521238320071 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-089-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
