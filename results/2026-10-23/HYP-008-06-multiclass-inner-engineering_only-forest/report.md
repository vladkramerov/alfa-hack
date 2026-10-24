# HYP-008-06-multiclass-inner-engineering_only-forest

- Начало: 2026-10-23T16:32:56+03:00; окончание: 2026-10-23T16:34:14+03:00.
- Статус: completed; вывод: Не подтверждена по заданному критерию.
- Гипотеза: [HYP-008](../../../docs/hypotheses/HYP-008.md).
- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md).
- Задача: multiclass; stage: inner.
- Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_engineering_control --task multiclass --threads 2`.
- Контроль: HYP-001-09-multiclass-inner-baseline-forest; выигрыш utility: -0.0077306152032420306.

Единственное семейство изменений относительно исходного baseline — engineer=True; correlation/VIF/scaling/Yeo–Johnson выключены, новые признаки сохранены.
Критерий до запуска: RMSE меньше baseline / Macro F1 больше baseline на тех же строках.

Разведочный анализ после просмотра основной inner-диагностики и части regression outer; splits используются повторно, независимого подтверждения нет. Основной frozen winner не меняется.

[Метаданные и параметры](metrics.json), [аудит признаков](preprocessing_audit.json), [предсказания validation](validation_predictions.csv).
