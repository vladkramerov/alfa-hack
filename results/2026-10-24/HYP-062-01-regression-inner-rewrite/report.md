# HYP-062/regression/inner

Started before fit: 2026-10-24T00:56:41.738647+03:00. [SPEC010](../../../docs/specs/SPEC-010-clean-rewrite.md).

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_clean_rewrite`.

Максимум900сек на все стадии гипотезы. Старый outer не переобучается. Source snapshot неизменен.

Итог: **completed, rmse=61.25577980475285**. Новый механизм; параметры не выбирались по confirmation.

[Метрики](metrics.json), [данные/исходники/split](evidence.json), [лог](worker.log).

Датированное заключение2026-10-24 после113/113сверок: HYP-062 не подтверждена по заданному критерию; confirmation RMSE=58.58032538035888 против контроля 42.46005247026025 на тех же позициях. НовыйZIPне создаётся, повтор запрещён. Эта запись не превращает inner-score в независимую оценку.
