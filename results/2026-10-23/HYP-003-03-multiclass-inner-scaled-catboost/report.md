# HYP-003-03-multiclass-inner-scaled-catboost

- Начало: 2026-10-23T16:13:27+03:00
- Окончание: 2026-10-23T16:15:36+03:00
- Статус: completed
- Вывод: Не подтверждена по заданному критерию
- Гипотеза: [описание](../../../docs/hypotheses/HYP-003.md)
- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md)
- Задача: multiclass; stage: inner.
- Команда: `.venv-ml/Scripts/python.exe -m experiments.run_suite --task multiclass --threads 2 --tag preprocessing-v1`
- Контроль: HYP-002-04-multiclass-inner-vif-catboost; выигрыш: -0.002467778399352971.

Критерий до запуска: RMSE меньше контроля / Macro F1 больше контроля; один и тот же split.
Для контроля без парного сравнения или grid-кандидата без reference вывод по гипотезе не выставляется.

[Полные параметры, версии, hashes и метрики](metrics.json).
[Предобработка](preprocessing_audit.json), [validation predictions](validation_predictions.csv).

Inner используется для выбора; outer — панель заранее замороженных конфигураций.
Outer сравнение полного recipe с baseline измеряет совокупный эффект pipeline, а не отдельный вклад feature engineering.
Результат ограничен выбранным split и бюджетом, не является оценкой скрытого test.
