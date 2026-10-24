# HYP-008-04-regression-outer-engineering_only-forest

- Начало: 2026-10-23T16:27:03+03:00; окончание: 2026-10-23T16:27:56+03:00.
- Статус: completed; вывод: Подтверждена в проверенных условиях.
- Гипотеза: [HYP-008](../../../docs/hypotheses/HYP-008.md).
- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md).
- Задача: regression; stage: outer.
- Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_engineering_control --task regression --threads 2`.
- Контроль: HYP-001-07-regression-outer-baseline-forest; выигрыш utility: 1.8576874202287783.

Единственное семейство изменений относительно исходного baseline — engineer=True; correlation/VIF/scaling/Yeo–Johnson выключены, новые признаки сохранены.
Критерий до запуска: RMSE меньше baseline / Macro F1 больше baseline на тех же строках.

Разведочный анализ после просмотра основной inner-диагностики и части regression outer; splits используются повторно, независимого подтверждения нет. Основной frozen winner не меняется.

[Метаданные и параметры](metrics.json), [аудит признаков](preprocessing_audit.json), [предсказания validation](validation_predictions.csv).
