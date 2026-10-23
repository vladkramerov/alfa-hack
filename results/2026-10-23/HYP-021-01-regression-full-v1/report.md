# HYP-021 regression

Статус: started до первого fit; исходная конфигурация и SHA записаны. [SPEC-005](../../../docs/specs/SPEC-005-agent-preprocessing.md).

Завершение:2026-10-23T22:06:32.722779+03:00; completed; rmse=43.837201829105254; Не подтверждена по заданному критерию.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.evaluate_full_preprocessing`. Контроль42.19670002719412 переиспользован без обучения. Validation сохраняет все строки; составной эффект без отдельного приписывания нормализации или выбросам. [Данные/код/split](evidence.json), [аудит](preprocessing-audit.json), [снимок кода](source/), [метрики](metrics.json), [лог](worker.log).
