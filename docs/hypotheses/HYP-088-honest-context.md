# HYP-088: contextual residual correction с честным raw OOF teacher

2026-10-24, [SPEC011](../specs/SPEC-011-three-hour-improvement.md). Статус: реализация зафиксирована до научного fit; **не оценивалась**. Реализация не даёт разрешения на запуск: координатор сначала регистрирует H88 и замораживает source/data/splits/config.

## Диагноз и новая процедура

В H74 inner training teacher — предсказания старой raw модели на собственных fit-строках, тогда как query teacher — held-out. Более уверенные training logits могут создавать сдвиг распределения для residual corrector. Это возможный механизм, а не доказанная причина отрицательных результатов. H88 меняет именно процедуру получения training teacher на grouped OOF; CatBoost/context/objective/weights остаются H74. Это не повтор H74 с новым seed и не объявление прежнего in-sample teacher независимым OOF.

## Frozen метод до fit

- Внутренний контекст: исходный cached raw fold0, 47075 fit /23454 query. Три **новых** вложенных SGKF3(seed42) на raw-X groups только внутри47075 fit дают training OOF. Их обучающие подмножества около31тыс строк не совпадают с известными original raw OOF fit около47тыс. Query probabilities берутся напрямую из прежнего `fold-0.npz`; старый teacher не переобучается и его in-sample train predictions не используются.
- Каждый новый teacher — строго `ClassBranch('raw', parameters={}, threads=2)`. Фактический Encoder **native=False, scale=False**, как у cached `ExperimentModel(policy='legacy')`; предположение о scale=True при разборе отвергнуто по исходникам. Median/OHE fit-only; XGB600/depth6/lr.05/lambda1/hist/cpu, random_state42, n_jobs2, multiclass softprob. Классы LabelEncoder выравниваются к общим лексически отсортированным классам; отсутствующий во вложенном train класс получает нулевую колонку, неожиданный класс отклоняется.
- Перед каждым новым teacher fit сохраняются immutable started evidence и split с local/global positions, dataSHA/sourceSHA/config/X/y digests. Обязательный injected hook `(estimator, raw_X, raw_y, configuration)` читает ledger и делает Memory.claim **до** `estimator.fit`; все3 learner fit должны быть отдельно учтены. Fit напрямую из provider не вызывается. Terminal outcome и predictions сохраняются отдельно; interrupted/failed claim без полного валидного cache не переобучается.
- Если точно совместимый historical nested raw cache уже есть, используется он. Завершённые новые fold caches с неизменными config/SHA/positions/classes также только читаются. Повреждённый/неполный cache означает технический отказ, а не скрытый новый fit.
- Исходный outer: training teacher — уже существующий полный raw SGKF3 OOF из `series-twenty-hypotheses-v1/cache`; query teacher — уже существующий H11 RAW outer probabilities. Оба проверяются по original outer positions, классам и data hashes; **0 старых teacher refits**. H48 meta-state вообще не используется: это raw OOF, а не cross-fitted H48 pipeline. Сравнительный outer reference остаётся отдельно H48 `.6495311620074445`, а не raw teacher score.
- Corrector `HonestContextCorrector`: неизменный H74 CatBoost400/depth5/lr.05/l2=10/2CPU, inverse-sqrt class-count weights с mean1, raw X + teacher logp, training logp baseline, query tree residual + logp. Teacher train SHA фиксируется worker в итоговой конфигурации. Wrapper требует корректной caller provenance; сам массив вероятностей не доказывает OOF.

## Критерий и ограничения

Сначала неизменный inner proxy: прямое сравнение с уже полученными H74/H78 и cached raw reference без их fit. Никакого выбора параметров по outer; дополнительный outer перенос только после заранее зафиксированного выбора. Цель20% F1 `>=.7794373944089334` считается только на исходном outer, остающемся разведочным. Общий900сек/H88 включает все3 новых teacher fits, inner contextual fit и возможный outer contextual fit; старые raw/H48 модели не обучаются. Nested training teachers видят меньше строк, чем query teacher; обычный OOF train-size shift остаётся, гарантий выигрыша нет.

Научные исходы и артефакты записывает root launcher в отдельные results/ и общий журнал сразу после проверки. Реализация: `experiments/sprint_crossfit_cache.py`, `sprint_honest.py`; synthetic/mocked tests: `tests/test_sprint_honest.py`, без public-data/real learner fits.

## Проверка реализации до научного запуска

TDD:6 ожидаемых FAIL до отсутствующих модулей; после реализации и двух дополнительных corruption/wrapper checks — **8/8 OK** (`.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_honest -q`). Проверены три hook-вызова после started evidence, raw-X group separation/полное OOF покрытие/class alignment, reuse без fit, selective allow_pickle=False чтение legacy query с лишним object-y, неправильные positions/missing hook, запрет повторения interrupted claims, отказ при изменённых cache bytes и unchanged zero-residual контекстный wrapper через dummy learner. Это технические synthetic/mocked проверки, не научные метрики.

Общий `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q` после всех изменений: **445 tests, OK**,10.550сек. Существующие три `matplotlib` `PyparsingDeprecationWarning`; новых исключений нет. В предыдущей общей проверке до двух дополнительных H88 tests также443/443 OK. Научных/public-data fit в рамках реализации: **0**.

SPEC011 result 2026-10-24T02:33:42.660451+03:00 HYP-088/inner/default: completed macro_f1=0.5978384913324375 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-088-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
