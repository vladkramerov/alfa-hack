# PLAN-009: альтернативные подходы

[SPEC-009](../specs/SPEC-009-alternative-models.md); выполнение прямо поручено пользователем. Baseline H43/H48, текущий ZIP сохранён. Не ждать отдельного согласования рутинных параметров.

- [x] 1. Записать20proposedH49…68 + H69, карточки первойволны играницыновизны.
- [x] 2. Classificationagent: TDD automl_next_class.py/experiments/next_classification.py — H49/51/53/60; безrealfit.
- [x] 3. Regressionagent: TDD automl_next_reg.py/experiments/next_regression.py — H54/55/56/58/60; безrealfit.
- [x] 4. Root: TDD automl_temporal_memory.py, persistent10case runner/worker, frozen sources, oldartifactmapping.
- [x] 5. Проверитьsource/cache/nooutery, numericalobjectives, затем10оценок; каждомуотдельныйreport+ledger.
- [x] 6. Независимыйпересчётметрик, source/fits/lineage; лучшиестрогоотносительно40.972997661684175/.6495311620074445.
- [x] 7. Толькоverifiedwinners portablepromotion, tests/ZIP/fullrefit; отрицательные не повторять. Новыеproposalsнеобъявлятьпроверенными.

Итог:9численныхоценок+1technicalfailure,65сверок,22уникальныхfit; H69перенесена,206tests/5mock/1lexical/2fullrefit passed. H65codecfixed, H68provenanceincomplete. [Приёмка](../../results/2026-10-24/ALTERNATIVE_DELIVERY.json). Tenfoldplatformgoalнеподтверждена.
