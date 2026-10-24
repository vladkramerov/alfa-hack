# HYP-012-01-regression-log1p_target

Статус: подготовлен до fit. [SPEC-004](../../../docs/specs/SPEC-004-ten-new-hypotheses.md).

Начало:2026-10-23T21:46:17.176118+03:00; завершение:2026-10-23T21:46:45.777872+03:00. Статус:completed; rmse=42.19670002719412; Подтверждена в проверенных условиях.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_novel_hypotheses`.

[Конфигурация](request.json), [SHA данных/split и чтение журнала до fit](evidence.json), [хеши стартового снимка](source-manifest.json), [полный snapshot](../series-ten-hypotheses-v1/source/), [метрики](metrics.json), [лог](worker.log).

Сохранённые контроли RMSE43.0046386479306 / Macro F1 0.5838559593680208 не переобучались. Медианы/статистики fit только на train; validation исходный и неизменённый. Повторно использованный outer: исследовательский вывод без независимого подтверждения.
