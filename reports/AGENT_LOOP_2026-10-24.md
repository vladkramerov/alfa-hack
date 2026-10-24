# Full-agent research: another_solution, 24.10.2026

Baseline: commit3567ad073b1ab3fd5bb1aec3423d264b28ee8a02, published ZIP efa97f675a8c80409eedccacbd19d0a1c0d44100180ccabbd974aedf2da64d31. The reported rank69→26 cannot be attributed to a mechanism without submission provenance. No platform submission in this series.

## Protocol and interpretation

Frozen ZIPs, same two masked CSV sets, official dictionaries, Python3.12,8CPU,600s/task,60research iterations,seed42,20kLLM tokens. Real organizer endpoint and qwen/qwen3.8-27b; key only child environment. Parent evaluator alone holds external labels. Child receives no external ground truth. Read-only instrumentation captures each OOF/checkpoint, decisions and actual API usage. Results: results/agent-loop/*.json; full local artifacts /private/tmp/alfa-another-live/{variant}/{regression,classification}, including process/provenance.json, command.log, receipts/exp-*.npz, final.npz, splits.npz and task/submission.csv. No dataset/prediction/model uploaded to Git.

The outer has been repeatedly explored in prior project work: this is a matched research comparison, **not an independent confirmation/private estimate**. A/B are also reused for decisions, not a pristine holdout. API proposals stochastic; one run per recipe cannot establish general causal superiority.600s is a finite evaluation ceiling, not a3-hour platform reproduction. No claim of top10 readiness from local metrics alone.

An initial6-iteration pilot did not reach Qwen (warmup longer than6). Its completed regression and interrupted classification are preserved and excluded; corrected control uses60iterations. This repeat corrected the evaluation protocol rather than re-testing an old model hypothesis.

## Proven process failures and implemented hypotheses

| ID | Evidence / change | Acceptance criterion |
|---|---|---|
| AL1 | Control regression planner prompts13247/13967inputtokens;28688total exceeded20k. Incumbent-relative compact history, prioritized profile preserving schema semantics, explicit organizer configuration, prompt/output/time reserve calibrated from actual usage | Fewer actual tokens, valid/usable proposals; prediction gain assessed separately |
| AL2 | Control exp15 KeyError heating_degree_days_c: inherited drops ran before Qwen code despite raw-column prompt contract. Code now runs before drops, datetime expansion and destructive cleanups | Exact failed control feature code must preprocess real full masked data; tests/no extra fit |
| AL3 | Control classification451.5s,5fallback experiments,0Qwen calls. Preserve baseline+3structural probes; allow planning when fewer than3typical fits remain | Full real-Qwen participation and same-resource final score; more calls alone insufficient |
| AL4 | Classification control exp1 spent94.1s duplicating baseline predictions. Compare complete deterministic train/test matrices before repeat fit, only identical model/supervised settings | Exact equality skips no-op; effective changes retained; actual saved budget |
| AL5 | Regression accepted exp14: pooled39.4998→39.3745, latestfold29.8381→30.3059; exp16 latest30.3752. Under detected forward shift choose incumbent/postprocess/ensemble on latestfold>=1000rows; retain older model training/diagnostic folds | Improved full-agent masked prediction, with explicit changed internal score scope |

AL1–3 tested as one frozen candidate first; AL4–5 plus calibrated AL1 reserve form the second frozen candidate. Combined effects cannot be assigned separately to each mechanism. Original AS001–003 remain closed and were not repeated. Strong senior-agent entity history, residual models, fold-safe supervised features, fallback portfolio, process isolation and guarded AS002 ensemble are retained.

## External and team evidence used

- [MLE-STAR](https://arxiv.org/abs/2506.15692): focused component refinement and ablation motivate fixing the measured failing block instead of replacing the whole agent. No web search at platform runtime introduced.
- [AIDE](https://github.com/AIcrowd/aicrowd-aide): draft/debug/improve branching as a reference for retaining failed hypotheses and recoverable checkpoints. No code/dependency copied.
- [MUCG9 policy](https://github.com/MUCG9/alfa-hack-automl/blob/main/agent_policy.py): bounded validated recipes, canonical signatures, primary-metric/nested-stopping settings. Signature deduplication already exists here; exact matrix no-op detection goes beyond recipe deduplication. Nested stopping is a next direction, not silently claimed as implemented.
- [hilhool review](https://github.com/hilhool/alfa-automl-agent/blob/main/Ревью_и_узкие_места_22-10.md): observed costly/unhelpful LLM loops, missing per-fold evidence, sandbox/deadline failures; useful audit targets, not transferable leaderboard measurements. Our process isolation already exists. No code copied; no ownership relation to leaderboard teams asserted.
- [Token-matched memory research](https://www.servicenow.com/research/publication/sina-hajimiri-are-emnlp2026.html): adjacent-domain motivation to match resources and measure utility rather than count calls; not ML-accuracy proof.

Team references (different budgets/scopes must remain distinct): another published offline OOF29.34208/.6534905 uses original full-public train's internal folds, not the masked outer. Prior maxim whole-policy masked36.30267/.6494125 and fresh Ivan control34.12912/.6276158 are historical longer/different-budget runs, not matched600s real-Qwen controls. Team research figures34.520269/.649531162 and earlier33.979840/.660279593 likewise cannot be presented as this series' delivery scores. Source/checkpoint details remain in maxim docs/RESEARCH_STATE.md and results/neural-transfer/e2e.json; original branches untouched.

Latest remote audit: origin/sergey_branch advanced24e3311→0efccdb (intermediate scientific series); origin/ivan_branch remainsfe674da. New research includes meta/focal/neural/NCA and panel-change evaluations. H84 primaryRMSE34.39312/secondary30.62373 are research estimates, not a fresh live-Qwen delivery; NCA/focal negatives not repeated. Latest origin references fetched without modifying team branches. AL6 transfers raw-X duplicate-group validation from team protocol, with feature-only train/test overlap gate; no model code imported wholesale.

## First measured pair (AL1–3): reject final policy

| Variant | External RMSE | External MacroF1 | Real tokens reg/class | Seconds reg/class |
|---|---:|---:|---:|---:|
| Frozen control |44.02962397|.6397114882|28688 /0|277.97 /452.13|
| AL1–3 |44.02962397|.6004257559|6938 /6257|254.27 /460.01|

Regression submission byte-identical to control; compact context reduced usage but did not improve predictions. Classification accepted0LLM improvements; exp4 log/ratio proposal completed but weakened internal CV. The rejected ensemble{3:.5,4:.5} did not beat original exp3 convincingly, so policy retained postprocessed single exp3. Control had instead trained value_memory and selected{3:.333,4:.667}, which materially improved the external result. Earlier researcher scheduling alone therefore lost a useful deterministic competitor. AL1–3 is not the recommended delivery; checkpoints retained.

16targeted contract tests plus6existing refinement tests pass. Exact control exp15 preprocessing replay now completes35000/18308 rows in.245s without fit. These are correctness/recovery checks, not accuracy proofs. AL1–6 candidate frozen ZIPea969e… now being evaluated. The prepared AL1–5 ZIPad5c8c… was never trained; superseded before first run to include the justified team validation transfer, not counted as an experiment.

## Second candidate, regression checkpoint
ExternalRMSE **44.0296239669→38.7971935984** (11.88% reduction). Actual495.77s versus277.97s under same600s ceiling: improved budget utilization is part of the policy, not compute-matched model superiority. Still below historical best team reg34.12912/own36.30267 (other budgets). ZIPea969e… unchanged source; classification pending, no final recommendation yet.
