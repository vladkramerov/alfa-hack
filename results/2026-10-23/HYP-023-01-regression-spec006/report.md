# HYP-023 / regression

Статус started до первого fit. [SPEC-006](../../../docs/specs/SPEC-006-twenty-hypotheses.md).

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_twenty_hypotheses`.

Данные/split/код: [evidence](evidence.json), [конфигурация](request.json), [снимок кода](../series-twenty-hypotheses-v1/source/).

Итог 2026-10-23T22:50:44.666196+03:00: **completed, rmse=43.19210947337134; Не подтверждена по заданному критерию**.

Прежний лучший 42.19670002719412 переиспользован безbaselinefit. [Метрики](metrics.json), [аудит](preprocessing-audit.json), [лог](worker.log), [predictions](validation_predictions.npz).

Ограничения: повторно использованный outer, без независимогоtest; локальнаяметрика не scoreплатформы.
