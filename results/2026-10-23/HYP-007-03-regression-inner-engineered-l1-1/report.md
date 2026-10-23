# HYP-007-03-regression-inner-engineered-l1-1

- Начало: 2026-10-23T16:07:21+03:00
- Окончание: 2026-10-23T16:07:21+03:00
- Статус: completed
- Вывод: Недостаточно данных
- Гипотеза: [описание](../../../docs/hypotheses/HYP-007.md)
- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md)
- Задача: regression; stage: inner.
- Команда: `.venv-ml/Scripts/python.exe -m experiments.run_suite --task regression --threads 2 --tag preprocessing-v1`
- Контроль: None; выигрыш: None.

Критерий до запуска: RMSE меньше контроля / Macro F1 больше контроля; один и тот же split.
Для контроля без парного сравнения или grid-кандидата без reference вывод по гипотезе не выставляется.

[Полные параметры, версии, hashes и метрики](metrics.json).
[Предобработка](preprocessing_audit.json), [validation predictions](validation_predictions.csv).

Inner используется для выбора; outer — панель заранее замороженных конфигураций.
Outer сравнение полного recipe с baseline измеряет совокупный эффект pipeline, а не отдельный вклад feature engineering.
Результат ограничен выбранным split и бюджетом, не является оценкой скрытого test.
