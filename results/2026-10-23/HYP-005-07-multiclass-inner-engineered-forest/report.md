# HYP-005-07-multiclass-inner-engineered-forest

- Начало: 2026-10-23T16:21:55+03:00
- Окончание: 2026-10-23T16:22:45+03:00
- Статус: completed
- Вывод: Не подтверждена по заданному критерию
- Гипотеза: [описание](../../../docs/hypotheses/HYP-005.md)
- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md)
- Задача: multiclass; stage: inner.
- Команда: `.venv-ml/Scripts/python.exe -m experiments.run_suite --task multiclass --threads 2 --tag preprocessing-v1`
- Контроль: HYP-004-04-multiclass-inner-gaussian-forest; выигрыш: 0.0.

Критерий до запуска: RMSE меньше контроля / Macro F1 больше контроля; один и тот же split.
Для контроля без парного сравнения или grid-кандидата без reference вывод по гипотезе не выставляется.

[Полные параметры, версии, hashes и метрики](metrics.json).
[Предобработка](preprocessing_audit.json), [validation predictions](validation_predictions.csv).

Inner используется для выбора; outer — панель заранее замороженных конфигураций.
Outer сравнение полного recipe с baseline измеряет совокупный эффект pipeline, а не отдельный вклад feature engineering.
Результат ограничен выбранным split и бюджетом, не является оценкой скрытого test.
