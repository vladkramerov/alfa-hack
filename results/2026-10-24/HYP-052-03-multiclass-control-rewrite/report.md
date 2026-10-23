# HYP-052/multiclass/control

Started before fit: 2026-10-24T01:01:14.679905+03:00. [SPEC010](../../../docs/specs/SPEC-010-clean-rewrite.md).

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_clean_rewrite`.

Максимум900сек на все стадии гипотезы. Старый outer не переобучается. Source snapshot неизменен.

Итог: **completed, macro_f1=0.6103340454456273**. Это разрешённая сравнительная проверка старого лидера на новом confirmation, без frozen public meta-state.

[Метрики](metrics.json), [данные/исходники/split](evidence.json), [лог](worker.log).

Датированное заключение2026-10-24 после113/113сверок: HYP-052 не подтверждена по заданному критерию; confirmation Macro-F1=0.4803098632106223 против контроля 0.6103340454456273 на тех же позициях. НовыйZIPне создаётся, повтор запрещён. Эта запись не превращает inner-score в независимую оценку.
