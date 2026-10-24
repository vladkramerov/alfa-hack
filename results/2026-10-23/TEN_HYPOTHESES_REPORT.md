# Проверка десяти новых гипотез — 2026-10-23

**Регрессия: новый лучший HYP-012, rmse 42.196700027194**; улучшение 0.807938620736; [отчёт](HYP-012-01-regression-log1p_target/report.md).

**Классификация: новый лучший HYP-011, macro_f1 0.591451370919**; улучшение 0.007595411551; [отчёт](HYP-011-02-multiclass-none/report.md).

## Все 11 проверок

Десять гипотез дали 11 validation-результатов: HYP-011 проверена на двух задачах. Δ = новая метрика − прежняя лучшая: для RMSE отрицательная Δ лучше, для Macro F1 положительная. Критерии зафиксированы до fit; строгие пороги: RMSE < 43.0046386479306 / grouped Macro F1 > 0.5838559593680208.

| Гипотеза / механизм | Задача | Метрика | Значение | Δ к прежнему лучшему | Порог пройден | Время, с | Источники |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| HYP-011: XGBoost histogram | regression | rmse | 43.510161481165 | +0.505522833234 | нет | 4.58 | [отчёт](HYP-011-01-regression-none/report.md) · [метрики](HYP-011-01-regression-none/metrics.json) |
| HYP-011: XGBoost histogram | multiclass | macro_f1 | 0.591451370919 | +0.007595411551 | да | 81.23 | [отчёт](HYP-011-02-multiclass-none/report.md) · [метрики](HYP-011-02-multiclass-none/metrics.json) |
| HYP-012: log1p target → expm1 | regression | rmse | 42.196700027194 | -0.807938620736 | да | 26.28 | [отчёт](HYP-012-01-regression-log1p_target/report.md) · [метрики](HYP-012-01-regression-log1p_target/metrics.json) |
| HYP-013: LightGBM Tweedie 1.5 | regression | rmse | 47.188319195788 | +4.183680547857 | нет | 2.41 | [отчёт](HYP-013-01-regression-none/report.md) · [метрики](HYP-013-01-regression-none/metrics.json) |
| HYP-014: Линейные листья + scaling | regression | rmse | 178.084987866166 | +135.080349218236 | нет | 3.69 | [отчёт](HYP-014-01-regression-linear_scaled/report.md) · [метрики](HYP-014-01-regression-linear_scaled/metrics.json) |
| HYP-015: Grouped OOF target encoding | regression | rmse | 44.402896602006 | +1.398257954075 | нет | 13.59 | [отчёт](HYP-015-01-regression-target_encode/report.md) · [метрики](HYP-015-01-regression-target_encode/metrics.json) |
| HYP-016: Категория proto×service×state | multiclass | macro_f1 | 0.580763595709 | -0.003092363659 | нет | 28.59 | [отчёт](HYP-016-01-multiclass-categorical_cross/report.md) · [метрики](HYP-016-01-multiclass-categorical_cross/metrics.json) |
| HYP-017: Четыре train-only частоты | multiclass | macro_f1 | 0.590968449912 | +0.007112490544 | да | 17.34 | [отчёт](HYP-017-01-multiclass-frequency/report.md) · [метрики](HYP-017-01-multiclass-frequency/metrics.json) |
| HYP-018: Multiclass one-vs-rest | multiclass | macro_f1 | 0.578639858805 | -0.005216100563 | нет | 16.14 | [отчёт](HYP-018-01-multiclass-none/report.md) · [метрики](HYP-018-01-multiclass-none/metrics.json) |
| HYP-019: Иерархия Normal / attack / тип | multiclass | macro_f1 | 0.590111042048 | +0.006255082680 | да | 19.05 | [отчёт](HYP-019-01-multiclass-hierarchical/report.md) · [метрики](HYP-019-01-multiclass-hierarchical/metrics.json) |
| HYP-020: Веса 1 / размер raw-X группы | multiclass | macro_f1 | 0.582934907208 | -0.000921052160 | нет | 29.22 | [отчёт](HYP-020-01-multiclass-inverse_group_weights/report.md) · [метрики](HYP-020-01-multiclass-inverse_group_weights/metrics.json) |

Вывод по фиксированному критерию: HYP-011: regression — Не подтверждена по заданному критерию, multiclass — Подтверждена в проверенных условиях; HYP-012: regression — Подтверждена в проверенных условиях; HYP-013: regression — Не подтверждена по заданному критерию; HYP-014: regression — Не подтверждена по заданному критерию; HYP-015: regression — Не подтверждена по заданному критерию; HYP-016: multiclass — Не подтверждена по заданному критерию; HYP-017: multiclass — Подтверждена в проверенных условиях; HYP-018: multiclass — Не подтверждена по заданному критерию; HYP-019: multiclass — Подтверждена в проверенных условиях; HYP-020: multiclass — Не подтверждена по заданному критерию.

Сохранённые прежние ориентиры и их происхождение: [CatBoost H4 и параметры](../history/2026-10-21/historical-experiments.json); [LightGBM HYP-009](HYP-009-02-multiclass-lightgbm/metrics.json). Внутрисемейные ориентиры регрессии: ExtraTrees 43.56756773832011, LightGBM 60.492688256313926; улучшение одного из них отдельно не означает победу над общим RMSE-порогом.

## Классы лучшего нового классификатора: HYP-011

Grouped Macro F1 0.591451370919; [полный per-class отчёт](HYP-011-02-multiclass-none/classification-report.json); [матрица ошибок](HYP-011-02-multiclass-none/confusion-matrix.csv).

| Класс | Precision | Recall | F1 | Число validation-строк |
| --- | ---: | ---: | ---: | ---: |
| Analysis | 0.636364 | 0.132701 | 0.219608 | 211 |
| Backdoor | 0.575758 | 0.120253 | 0.198953 | 158 |
| DoS | 0.496933 | 0.135565 | 0.213018 | 1195 |
| Exploits | 0.630599 | 0.916920 | 0.747272 | 3286 |
| Fuzzers | 0.737775 | 0.746601 | 0.742162 | 1839 |
| Generic | 0.997269 | 0.978034 | 0.987557 | 3733 |
| Normal | 0.933163 | 0.918239 | 0.925641 | 5565 |
| Reconnaissance | 0.913253 | 0.745329 | 0.820790 | 1017 |
| Shellcode | 0.656000 | 0.694915 | 0.674897 | 118 |
| Worms | 0.714286 | 0.263158 | 0.384615 | 19 |

## Проверка и ограничения

Сумма worker runtime: 242.13 с; завершено 11/11 проверок. Всего запланировано 12 новых model fit: HYP-019 включает две свежие модели с отдельной предобработкой binary и attack-only частей; [аудит иерархии](HYP-019-01-multiclass-hierarchical/preprocessing-audit.json). Старые baseline не переобучались (baseline_refits=0); tuning/повторных baseline fit нет.

Все метрики независимо пересчитаны из сохранённых предсказаний и исходных y: [независимая проверка](series-ten-hypotheses-v1/independent-verification.json). [Записи 11 запусков](series-ten-hypotheses-v1/records.json), [итог серии](series-ten-hypotheses-v1/summary.json), [план до fit](series-ten-hypotheses-v1/frozen-plan.json), [SHA исходного кода](series-ten-hypotheses-v1/source-manifest.json). 46 unit tests прошли; [финальный лог тестов](series-ten-hypotheses-v1/final-unit-tests.log).

HYP-018 сохраняет независимые OVA scores и выбирает argmax; они не перенормированы и не обязаны суммироваться к единице. Фактический диапазон сумм OVA scores: [0.09743017472683135, 1.7645939029187696].

Исходные outer indices сохранены: regression 35 000/18 308, multiclass 70 529/17 141; категории, медианы, масштабирование, target/frequency statistics обучены только на fit. Для HYP-015 training encoding исключает всю группу одинакового исходного X; для HYP-020 validation не перевзвешен. Outer уже использовался ранее: выбор разведочный, независимого подтверждения и оценки статистической значимости нет.

Независимый verifier проверяет сохранённые артефакты и заявленные аудиты; они сами по себе не доказывают весь порядок исторического выполнения.

Платформенный агент автоматически не обновлён экспериментальными победителями; локальные метрики не являются скрытым платформенным score. Отправка на соревновательную платформу не выполнялась. Пользователь разрешил публикацию кода в GitHub ivan_branch; разрешение не означает, что push уже выполнен.
