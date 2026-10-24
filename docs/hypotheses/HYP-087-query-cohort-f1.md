# HYP-087: query-cohort expected MacroF1 without query labels

SPEC011,2026-10-24, proposed before fit. H75 freezes class decisions from training distribution; H87 estimates query class prior from calibrated query probabilities and optimizes the expected confusion objective over the complete unlabeled query cohort. This is explicitly transductive; batch invariance is not promised. Assumes calibration and posterior transport, no guarantee these hold. Applies only to MacroF1 full-table prediction, not probability metrics or streaming rows.

Reuse frozen H78 calibration theta from compatible inner/outer audit, never refit it; normalize teacher, clip1e-6, softmax(logp*exp(-theta0)+bias). Querypi=mean calibratedp. 25 damp.5 affine derivative iterations, select largest expectedMacroF1 with initialargmax included. No query labels/counttruth/ID, no search of iterationparameters or validationF1. H75trainingprior differences and H78propercalibration are separate testedmechanisms; newchange is wholequery unlabeled decisionoptimization. Source/config/cacheSHA/data/splits beforetrial, ≤900sec nooldfits.

Strict incremental inner criterion beats H78.6288481661155734 beforeouter; directteacher .587338 separately recorded. Originalouter≥.779437goal, currentbest.6593084142. Two synthetictests validateexpectedobjective/permutation/nolabelaccess/normalization/state rejection; syntheticlossnotcompetitionmetric.

SPEC011 result 2026-10-24T02:20:21.020777+03:00 HYP-087/inner/default: completed macro_f1=0.6254419869564557 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-087-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
