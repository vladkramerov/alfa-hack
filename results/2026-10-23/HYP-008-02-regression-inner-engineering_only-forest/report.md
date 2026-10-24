# HYP-008-02-regression-inner-engineering_only-forest

- Начало: 2026-10-23T16:24:26+03:00; окончание: 2026-10-23T16:24:50+03:00.
- Статус: completed; вывод: Подтверждена в проверенных условиях.
- Гипотеза: [HYP-008](../../../docs/hypotheses/HYP-008.md).
- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md).
- Задача: regression; stage: inner.
- Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_engineering_control --task regression --threads 2`.
- Контроль: HYP-001-03-regression-inner-baseline-forest; выигрыш utility: 0.04040796189988072.

Единственное семейство изменений относительно исходного baseline — engineer=True; correlation/VIF/scaling/Yeo–Johnson выключены, новые признаки сохранены.
Критерий до запуска: RMSE меньше baseline / Macro F1 больше baseline на тех же строках.

Разведочный анализ после просмотра основной inner-диагностики и части regression outer; splits используются повторно, независимого подтверждения нет. Основной frozen winner не меняется.

[Метаданные и параметры](metrics.json), [аудит признаков](preprocessing_audit.json), [предсказания validation](validation_predictions.csv).
