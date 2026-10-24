# External repository ideas Implementation Plan

**Goal:** самостоятельно проверить новый causal multi-lag механизм и сравнить
с готовой поставкой Сергея, не обещая20% или platformgain.
**Architecture:** новыйisolated module external_multilag.py, фиксированный
runner experiments/external_ideas_run.py, train-only native transform,
cachedcontrols; текущие productionmodules не менять до победы.
**Tech Stack:** Python3.12, pandas/numpy/LightGBM4.6, unittest.
**Spec:** [SPEC013](../specs/SPEC-013-external-repository-ideas.md).

- [x] T1 записать внешние commits/claims/несопоставимости и карту повторов.
- [x] T2 RED tests: exactlags gaps, no-self/future y, collisions, fit-only
  categories/statistics, featureselection restrictedfit, row/batchinvariance.
- [x] T3 implement H105; run tests GREEN; freeze card/config/source/data/splits.
- [x] T4 inner<=900sec with prefitMemoryclaim and zero oldlearnerrefits;
  twofixedinferencearms; immediate results+ledger evenerror.
- [x] T5 onlyifbestinner beats31.046636 AND30.636364: preregisteronearm,
  outerevaluateonce; independentlyrecompute/verify allpredictions.
- [x] T6 report actual outcome and recommendation; onlypositivequalifying
  winnereligiblefornewportableintegration, otherwise retainSergeyZIP.

Review focus: missing/zero year_built inidentity is ambiguous; futurelabels
neverhistory; unseen categories→missing; exactlags differ fromlastobserved;
input order/duplicatedqueryrows mustnotalterpredictions; foldparametersfixed.

Фактическийитог: H105best32.016724 проиграла31.046636; H10630.963254 дала0.2686%inner, но не прошла порог30.636364. T5условнопропущен, outerнепроводился, T6сохраненSergeyбезновогоZIP. [Отчет](../../results/2026-10-24/external-repositories-review-v1/report.md),562unitPASS и nofitreplayverificationPASS.
