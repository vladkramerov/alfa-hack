# HYP-039 / multiclass

Статус started до первого fit. [SPEC-006](../../../docs/specs/SPEC-006-twenty-hypotheses.md).

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_twenty_hypotheses`.

Данные/split/код: [evidence](evidence.json), [конфигурация](request.json), [снимок кода](../series-twenty-hypotheses-v1/source/).

Итог 2026-10-23T23:07:41.727174+03:00: **completed, macro_f1=0.5795834428470248; Не подтверждена по заданному критерию**.

Прежний лучший 0.5914513709191194 переиспользован безbaselinefit. [Метрики](metrics.json), [аудит](preprocessing-audit.json), [лог](worker.log), [predictions](validation_predictions.npz).

Ограничения: повторно использованный outer, без независимогоtest; локальнаяметрика не scoreплатформы.
