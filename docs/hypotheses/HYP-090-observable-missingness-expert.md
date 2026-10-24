# HYP-090: original-RMSE эксперт для наблюдаемого отсутствия rating поверх H70

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: Запланирована. На competition data не обучалась при реализации.
- До реализации прочитан полный HYPOTHESES.md и H35/H41/H45 proposals.

## Утверждение и новизна

При отсутствующем energy_star_rating условные зависимости и масштаб y могут
отличаться, а прежний log-target fallback может недооценивать высокие значения.
Новый original-scale squared-loss эксперт обучается **только** на training
строках с отсутствующим rating; query gate полностью наблюдаем из X.
Метрика выигрыша относится к новой комбинации с фиксированным cached H70.

H35 обучала log-target экспертов по building_class, H41 выбирала whole-task
pipeline, H45 учила target-tail gate для неизвестного режима y. H60 добавляла
искусственные пропуски training X, H85 учила conditional lognormal mean на всех
строках. H90 не повторяет эти механизмы: явный missing-rating subgroup,
оригинальный RMSE и заранее фиксированный observable query gate; H70 не refit.

Выбор проблемного среза мотивирован уже просмотренными outer ошибками, поэтому
даже frozen последующая outer оценка остаётся адаптивной и разведочной.
Нельзя представлять её как новое независимое доказательство обобщения.

## Frozen метод до realfit

`sprint_missing_expert.MissingRatingExpert(threads=2,fit_hook=None)`:

- Training/query missing mask: `~isfinite(to_numeric(energy_star_rating,
  errors='coerce'))`. NaN/inf/unparseable считаются неизвестным rating;
  конечный0 остаётся наблюдаемым. Отсутствующий столбец отклоняется.
- Минимум300 missing training rows. При меньшем числе learner не обучается,
  все predictions — неизменённый supplied H70.
- CatBoostRegressor800iterations/depth6/learning_rate.05/l2_leaf_reg20,
  loss RMSE на **исходном y**, random_seed42, threads2, verboseFalse,
  allow_writing_filesFalse, без validation early stopping/весов/удаления outlier.
- FeatureComposition29/30/31 fit **только на missing training subset**;
  native tagged categories, numeric inf→NaN. Исходные ID/target удаляет caller
  до fit как в остальных scientific candidates. Known-rating training rows
  не участвуют даже в оценке feature statistics нового эксперта.
- Known-rating query prediction в точности unchanged H70. Missing query:
  `.7*max(expert_prediction,0)+.3*H70`. Вес.7 fixed до первого fit, не
  оптимизируется по inner/outer; эксперта не вызывать для known query rows.
- Supplied `base_predictor` обязателен. Ни его fit, ни training prediction не
  вызываются; допускается read-only cached H70, покрывающий только query rows.
  Base provenance/его predictionsSHA подтверждает coordinator.
- Время≤900сек на гипотезу суммарно inner+возможный outer, до4CPU,
  frozen default2CPU; один новый expert learnerfit на split.

## Учёт фактического learner subset

Сохраняются `fit_mask_`, `fit_positions_`, audit expert_fit_mask,
expert_fit_positions, expert_fit_original_indices, expert_fit_rows/count и
feature fit audit. Эти позиции локальны относительно поданного outer/inner
training X; оригинальный index сохраняется отдельно.

Не смешивать wrapper Memory claim на полном X с фактическим CatBoost learner
fit на subset. Optional hook
`fit_hook(estimator,subset_frame,subset_y,configuration,**fit_kwargs)` позволяет
отдельно зарегистрировать transformed subset и original y; fit_kwargs содержит
cat_features. Hook возвращает fitted estimator и не сохраняется в pickle.
Модуль сам не изменяет общий ledger/harness и не читает старые cache paths.

## Данные и критерий успеха — до запуска

Regression trainSHA256
`096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`.
Inner — trainingYear<4→Year4, original outer-fit; baseline **cached H70inner**
RMSE30.663255379972725, не H69. Wrapper может сохранять также готовый H69
reference42.46005247026025, но inherited H70 gain не выдаётся за вклад H90.
Utility: strict уменьшение overall original RMSE относительно H70, при exact
неизменности known-rating query predictions; missing slice RMSE отдельно.

Outer — trainingYear<5→Year5, fit35000/valid18308, только если H90 выбран
финалистом до score. H70 reference34.318489927675174, original H69
39.39219382766771; пользовательская цель20% **<=31.51375506213417**. Меньший
выигрыш сохраняется честно отдельно от достижения20%; labels не становятся
gate или признаками, validation/test не фильтруются.

## Техническая проверка

TDD:4tests сначала упали из-за отсутствующего модуля, затем4/4green за2.123сек.
Проверены actual subset rows/targets, fit_hook на реальном tinyCatBoost,
feature stats толькоsubset, fixed.7/.3blend, known/zero gate unchanged,
missing/inf semantics, insufficient support nofit fallback, nobasefit,
unknown/missing categories, query batching, pickle и sourceimmutability.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_missing_expert -v`.
Synthetic learnerfit не является scientific competition experiment.

## Запуски

| Дата | Отчёт | Статус выполнения | Вывод |
| --- | --- | --- | --- |
| — | Координатор сохраняет results/ перед fit | Не запускалась | Не оценивалась |

## Следующий шаг

Регистрация H90/source/config/data/split/baseSHA до одного innerfit. Сразу
сохранить итог и фактический subset; при отсутствии incremental H70 gain
outer не запускать и не менять configuration после score.

SPEC011 result 2026-10-24T02:29:25.093548+03:00 HYP-090/inner/default: completed rmse=34.86994782367874 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-090-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
