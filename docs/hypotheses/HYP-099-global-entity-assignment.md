# HYP-099: global one-to-one temporal record assignment

- Дата:2026-10-24, Москва.
- Связь: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: design/config зафиксированы до реализации, код и синтетические unit-проверки готовы; научных fit/метрик при передаче нет.
- Основание: поручение координатора реализовать новый joint assignment; corrected assignment acceptance согласована его прямым сообщением до freeze. Реальный запуск только после регистрации/root approval.

## Утверждение и отличие от H81

Изменения реестровой площади/года постройки могут нарушать exact physical keys. Joint one-to-one matching целого query-year cohort не допускает повторного использования одного донорского здания и резервирует уже достоверные exact H70 links. Это меняет combinatorial matching problem по сравнению с H81 local nearest/margin; один донор, который был ближайшим двум query, теперь не может обслужить оба. Близость и оптимальное назначение **не доказывают идентичность здания**. Цель — отдельная научная проверка качества, не подтверждение утечки.

## Frozen default config

`GlobalEntityAssignment(alpha=.35, age_decay=.25, max_assignment_cells=4000000)`.

API `fit(X,y,base_predictor=None,transport_state=None)`, `predict(X)`, `configuration()`, `audit_`, `last_prediction_audit_`. Реальный base predictor — совместимый cachedH70high, state=замороженные rating_beta/group_rating_betas H70; H70 не fit заново. Global assignment не обучает beta/cost/alpha. Нет state →beta0; нет predictor →fit mean только технический fallback.

1. Canonical exact profile: State_Factor/building_class/facility_type/floor_area/year_built как H70. Training-profile с любым sameyear collision отвергается целиком. Matching не читает ID/y, rating не входит в стоимость. y сохраняется только для последующего переноса уже выбранного donor, finite и≥0.
2. Каждый queryYear обрабатывается отдельно. Выбирается **глобальный latest training year<queryYear**; donors только из этого года, без возврата к более старому году для отдельного блока. Block точно равен State_Factor/building_class/facility_type.
3. Query keys, имеющие retained exact H70 profile со strictly-past observation, сохраняют baseline и резервируют этот profile: соответствующий донор недоступен остальным queries. Query keys, совпавшие с rejected ambiguous training profiles, тоже остаются baseline из осторожности. Несколько query rows одного exact key в одном queryYear не участвуют в assignment; exact reservation при этом всё равно действует.
4. Остальные rows/donors сортируются по canonical exact feature-key, без ID/target/исходной позиции. После duplicate exclusion keys уникальны, поэтому row permutation не меняет keyed predictions. Упорядочение — только tie-breaking, не отдельная ML feature.
5. Для positive finite area query/donor gate ratio=`areaQ/areaD`∈[.95,1.05]. Если оба year_built finite и>0, abs gap≤20; иначе year penalty=.5. Cost=`abs(log(areaQ/areaD))/.05 + abs(yearQ−yearD)/20 + abs(elevationQ−elevationD)/100`; unknown/0year term=.5; missing любое elevation term=.5. ELEVATION — optional column, отсутствие означает missing penalty, а не ошибку. Известная числовая elevation0 допустима.
6. **Только edges cost≤1.0 допустимы**. Остальные получают forbidden cost. `scipy.optimize.linear_sum_assignment` решает матрицу N×(M+N) с N dummy columns cost1.000001; каждый query может остаться unmatched, каждый real donor используется максимум раз. Invalid cost1…2 edges вообще не участвуют, поэтому не могут блокировать допустимую пару. Это corrected design координатора; прежнее предложение dummy2/postfilter1 не реализуется. Оптимизируется общая сумма cost+dummies, не max-cardinality matching.
7. Memory cap включает **всю** Hungarian matrix N*(M+N)≤4,000,000 cells; больше →блок пропускается целиком с audit reason/counts, без fallback area-binning или новой эвристики. Нулевые candidate blocks остаются baseline без allocation.
8. Accepted donor y переносится с frozen H70 inverse-normal rating score: finite оба rating →donor_y*exp(clip(beta_facility*Δq,−1.5,1.5)), иначе raw donor_y. Beta finite в[0,2]; group→global fallback. Confidence weight=`.35*exp(−.25*max(0,queryYear−donorYear−1))` без support factor. Output=weight*transported+(1−weight)*H70 baseline; exact/unmatched/ambiguous/skipped rows unchanged.

Cost normalizers, identity assumption, alpha и dummy penalty инженерно фиксированы, не learned confidence. Query statistics используются для joint assignment: **предсказание cohort-dependent**, subset prediction может отличаться; production должен передавать весь годовой submission cohort. Query rows никогда не становятся donors. Идентичный полный cohort после permutation должен давать идентичные keyed outputs.

## Научный протокол до запуска

Regression train SHA256 из provenance `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`. Научный worker здесь CSV не читает. Inner Year<4→4, cachedH70high RMSE30.663255379972725; проверяется incremental gain над H70. Старые baselines не fit. Outer только после отдельного выбора/допуска координатором, viewed outer разведочный; H91best34.21382789041039 остаётся текущим comparator при продвижении. Цель SPEC011≤31.51375506213417 не меняется. Один config, source/data/split/state/config freeze доfit,≤900сек; все исходы отдельноresults, scientific model fits0 при подготовке.

## План реализации и проверки

Только `sprint_assignment.py` и `tests/test_sprint_assignment.py`; синтетические tests-before-code. Проверки: ручное2×2 назначение при local nearest collision; no donor reuse; reserve exact unchanged; latest strictly-past year и block/area/year/cost gates; optional/missing elevation/year penalties; full matrix cap including dummies; query duplicate veto; frozen transport-state copy; row permutation/keyed predictions, pickle и no query memorization. Реальных fit/научного score при реализации не проводить.

## Результат

Source SHA256 при передаче: `3d90369ef582e1d0883890fd402f823c1e43046319181616f64f9af8d689951c`.

`.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_sprint_assignment.py -v`: **7/7 green**,0.042сек. Ручной2×2 fixture: donor areas100/104 иtargets100/300; query areas101/101.5 оба локально предпочитают100, joint optimum назначает100/104 и даёт41.5/111.5 при baseline10. Тест намеренно демонстрирует cohort dependence: одиночный query101.5 даёт41.5.

Дополнительно проверены reserved exact donor, отсутствие повторного использования, duplicatequeryveto, rejected trainingkey fallback, latestglobalyear без возврата к близкому старому году, sameyear/future exclusion, формула cost и pre-assignment rejection cost>1, missing/0year иoptional elevation, dummy-inclusive memory cap, frozen rating transport copy/no estimation, keyed permutation/pickle/query independence и невалидные y/state/baseline.

Полный `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q`: **498 tests, OK**,11.425сек; три известных PyparsingDeprecationWarning от matplotlib. Эта проверка не производит новый scientific score.

Scientific outcome **Не оценивалась**; real-data model fits0. Координатор читает карточку/config/source и допускает отдельно реальный запуск с отдельным results/run.

SPEC011 result 2026-10-24T02:50:50.554109+03:00 HYP-099/inner/default: completed rmse=31.099625876123866 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-099-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
