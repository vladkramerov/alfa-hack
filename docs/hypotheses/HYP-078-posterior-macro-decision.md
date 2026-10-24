# HYP-078: proper posterior calibration before expected Macro-F1 decisions

2026-10-24, SPEC011. Proposed before any fit. H75 assumes calibrated posterior whereas H48 uses inverse-square-root class weights; proper unweighted CE calibration may improve applicability of its expected confusion objective. This is not H37 multiplicative F1 grid: objective is proper unweighted CE, scalar temperature plus class intercept followed by H75 additive offsets.

Frozen: log probabilities clipped1e-6, scalar log temperature bounded[-2.3,2.3], biases[-5,5], L2=.01/2*norm(theta)^2, L-BFGS-B100iterations/ftol1e-10, zero initialtheta; only training teacher and labels. Keep initial if objective nonfinite/worse. Then25iteration damp.5 expected MacroF1 affine decisions. No query cohort statistics, targets/IDs not features; no validation labels during optimization. Teacher training in-sample limitations inherited and recorded, no claim of independent metaOOF. Full frozen config/source/dataSHA/split written by launcher before fit.

Compare fixed inner raw teacher F1.5873380697; success strict improvement. Outer final winner selected before score, compared against H48.649531;20% target>=.779437 separately. Up to900sec, cached baseline only, all outcomes separate results/ including negatives; no replay.

TDD3/3: calibration finite-difference gradient, named classes/batch invariance/pickle and training CE decrease; invalid teacher rejected. This is implementation evidence only, not competition metric.

SPEC011 result 2026-10-24T02:05:53.156581+03:00 HYP-078/inner/default: completed macro_f1=0.6288481661155734 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-078-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:07:00.046247+03:00 HYP-078/outer/default: completed macro_f1=0.6593084142037399 vscontrol=0.6495311620074445; [report](../../results/2026-10-24/HYP-078-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.
