# Radical combinations — Implementation Plan

> For agentic workers: parallel isolated regression/classification modules, root owns registrar/runner/report/promotion; execute authorized seven hypotheses without another permission round.

**Goal:** объединить полезные механизмы и добавить в baseline только проверенные улучшения.
**Architecture:** portable root regression/class modules shared by experiments and delivery; one fit registrar and persistent OOFcache; sequential7evaluation runner; independent metrics verifier; explicit knownstrategy promotion.
**Tech Stack:** existing numpy/pandas/sklearn/scipy/ET/XGB/LGB/SMOTENC, providedrequirements unchanged.
**Spec:** [SPEC-008](../specs/SPEC-008-combinations.md).

## Global constraints

7новыхграфовH42…48; frozenouter/dataSHA; incumbent41.480963027907194/.6286047899141216; <=4threads,1800s/case; no oldfitretry; persistentstartedbeforefit; noouteryforselection.

## Review focus

Raw duplicate grouping before synthetic/engineering; nofutureyearsinOOF; oldbaselinecache never regenerated; class/componentalignment; learnedstate reused on refit; only verifiedwins changebaseline.

- [x] 1. TDDregistrar automl_combo_memory.py + tests/test_combo_memory.py; duplicate/seedrename/changedfeaturescases.
- [x] 2. TDDportableautoml_feature_sets.py/automl_reg_combo.py and experiments/reg_combinations.py; expert+features, convexOOFweights,target-tail gate; regressionagentownsfiles.
- [x] 3. TDDautoml_class_combo.py and experiments/class_combinations.py; boundedSMOTEOOF, frequencyblend, frozenmetastack; classificationagentownsfiles.
- [x] 4. Rootrun_combinations.py/combo_worker.py:7frozenjobs+cards+sourcecopy; requestscontaincachedoutercomponentsnotoutery; claim/finishalljournals; resumeuseshistory.
- [x] 5. Preflightunit/review/cacheidentity; execute7cases, saveOOF/weights/state/probas/errorsimmediately.
- [x] 6. verify_combinations.py independenttruth/metric/order/folds/fingerprints; comparisonagainstcurrentbaseline, retainnegativehypotheses.
- [x] 7. Promoteverifiedwinnersinbaseline/knowncanonicalfactory/portfolio,freezelearnedstate; unit+cleanZIPchecks,fullrefitonlyifneeded; finalreportandZIPifimproved.


Итог: 7 метрик,63 независимые сверки,24 уникальных fit,161 тест,5 mock сценариев,2 чистых fullrefit; ZIP57110байт. Результаты: results/2026-10-23/COMBINATIONS_DELIVERY.json.
