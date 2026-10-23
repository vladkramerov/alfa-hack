# HYP-021 multiclass

Статус: started до первого fit; исходная конфигурация и SHA записаны. [SPEC-005](../../../docs/specs/SPEC-005-agent-preprocessing.md).

Завершение:2026-10-23T22:07:19.676950+03:00; completed; macro_f1=0.5637871673383791; Не подтверждена по заданному критерию.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.evaluate_full_preprocessing`. Контроль0.5914513709191194 переиспользован без обучения. Validation сохраняет все строки; составной эффект без отдельного приписывания нормализации или выбросам. [Данные/код/split](evidence.json), [аудит](preprocessing-audit.json), [снимок кода](source/), [метрики](metrics.json), [лог](worker.log).
