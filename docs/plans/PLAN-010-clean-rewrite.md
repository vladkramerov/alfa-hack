# Clean rewrite Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development. Spec is binding; continue through implementation and verification without new generic approval gates.

**Goal:** независимо реализовать диагностический агент и проверить H52/H62, сохранив прежний лидер при отсутствии подтверждённого выигрыша.
**Architecture:** новые standalone research_*.py: data contracts, два независимых learner, controller/history/LLM; experiments orchestration отдельно. Старая модель используется только как явно разрешённый контроль на дополнительном протоколе.
**Tech stack:** Python3.12, numpy/scipy/pandas/sklearn; existing allowed tree libraries optional.
**Spec:** [SPEC010](../specs/SPEC-010-clean-rewrite.md).

## Global constraints

900sec/new local hypothesis, <=4threads per worker; no old validation repeats; no automl imports in new product files; provided requirements unchanged; every outcome logged. Existing ivan_branch is user-requested. No new archive without promotion and preflight.

## Tasks and interfaces

- [x] T1: `research_neighbors.py` implements `MixedNeighborsClassifier(n_neighbors=25).fit(X:DataFrame,y:Series)` / predict / predict_proba / classes_. Tests `tests/test_research_neighbors.py` first, observe RED then GREEN, exact distance oracle and train-only preprocessing. Model has `.audit_` and JSON serializable `.configuration()`.
- [x] T2: `research_fm.py` implements `FactorizationRegressor(rank=8,epochs=20,batch_size=512,learning_rate=.005,l2=.001,random_state=42)` / fit / predict, `.audit_` and `.configuration()`. `fm_loss_gradient(...)` interface locally documented for numerical-gradient test. Tests `tests/test_research_fm.py` first; finite differences and meaningful interaction-learning check before actual data.
- [x] T3: `research_data.py`: `load_task(path)->Task` dataclass `.X,.y,.X_test,.sample,.task_path,.task_type,.metric,.id_column,.target_column,.hashes`; `split_data(task)->(fit,valid,description)` train-only time/group; `score(metric,y,pred)`, `better(metric,new,old)`, `save_submission(task,pred,path=None)`, `diagnostics(task_type,y,pred,classes=None,probabilities=None)->dict`. Independent renamed/nested/opaque fixture tests first.
- [x] T4: root `research_agent.py`, `rewrite_solution.py`, `research_memory.py`: atomic started/history/outcome, bounded LLMapi, diagnostic context, worker subprocess, typed recipe dispatch to T1/T2 and allowed generic models; valid fallback first, final selected refit once. Tests history/metrics/API/CLI before implementation.
- [x] T5: `experiments/run_clean_rewrite.py`: pre-register H52/H62; source/data/split freeze; inner one-fit per model, 900sec cap; claim before model.fit; persistent log before/after; candidate final comparison and authorized leader comparison only if candidate warranted. `experiments/verify_clean_rewrite.py`: read-only recompute NPZ scores and fit lineage, no training; report/registry updates.
- [x] T6: broad review + meaningful tests, fix only actual issues; optional new flatZIP only on measured promotion; update Codex/docs/results/Git. No rewrite of historical outcomes.

## Review focus

Opaque 001/NA classes and identifiers; all-missing/unknown columns; rare classes and raw duplicates; non-random future test; timeout before final prediction. Owning tests T1/T2/T3/T4 must cover relevant boundaries. Confirmation is additional stability evidence, not private-quality proof.

## Preflight interface scan

| Tasks | Contract / possible conflict | Ruling |
|---|---|---|
| T1/T2/T4 | fit/predict/configuration/DataFrame | learner files disjoint; factory owns dispatch |
| T3/T4/T5 | Task/split/metric/CSV contract | exact interfaces above; no old module imports |
| T1/T5; T2/T5 | each actual fit journaled | experiment controller owns before/after wrapper; synthetic test fits not scientific real-data runs |
| T3/T5 | confirmation vs inner | offline runner freezes confirmation before fitting inner; don't call public outer again |
| T1 | mixed metric and same-source test | brute force oracle; no hyperparameter search |
| T2 | squared loss vs scaled target | scaling affine train-only, same raw squared-loss optimum; finite differences first |
| T4 | general algorithms vs no-repeat public experiments | generic production recipes tested only on synthetic fixtures, no unregistered public baselinefit |
| T5 | leader requires original modules | old imports allowed in offline control only; no frozen public meta reuse on new split |
| T6 | ZIP gate vs request for rewrite | experimental implementation is deliverable even if failed; no misleading improvedZIP |

## Progress

2026-10-24: user confirmed final grill plan. Ruling: follow existing branch and separate new files; no worktree requested, don't relocate user's active checkout.

2026-10-24 completion ledger: T1/T2/T3/T4 implemented and reviewed; T5 all6stages completed, H52/H62 negative on frozen confirmation against authorized controls; T6 verification113/113, unit257/257, cleanCLI4/4, source/fingerprints checked. No promotion/ZIP: metric gate false, old solution/baseline/archive unchanged. Sources/results prepared for user-authorized ivan_branch publication. No independent private-quality claim.

Review fixes: class/probability ordering, total deadlines, fallback/resume and no-repeat history fixed before source freeze. Read-only verifier fixed to bind run splits/actual-fit fingerprints to frozen data and disallow promotion when any check fails. Initial verifier110/113 was a NumPy-vs-pandas numeric parsing difference <=5.684e-14; original failed report/source preserved, canonical CSV conversion correction passed113/113, no fit repeated.
