# HYP-008-08-multiclass-outer-engineering_only-forest

- Начало: 2026-10-23T17:01:13+03:00; окончание: 2026-10-23T17:02:58+03:00.
- Статус: completed; вывод: Подтверждена в проверенных условиях.
- Гипотеза: [HYP-008](../../../docs/hypotheses/HYP-008.md).
- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md).
- Задача: multiclass; stage: outer.
- Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_engineering_control --task multiclass --threads 2`.
- Контроль: HYP-001-12-multiclass-outer-baseline-forest; выигрыш utility: 0.0051823166254236774.

Единственное семейство изменений относительно исходного baseline — engineer=True; correlation/VIF/scaling/Yeo–Johnson выключены, новые признаки сохранены.
Критерий до запуска: RMSE меньше baseline / Macro F1 больше baseline на тех же строках.

Разведочный анализ после просмотра основной inner-диагностики и части regression outer; splits используются повторно, независимого подтверждения нет. Основной frozen winner не меняется.

[Метаданные и параметры](metrics.json), [аудит признаков](preprocessing_audit.json), [предсказания validation](validation_predictions.csv).
