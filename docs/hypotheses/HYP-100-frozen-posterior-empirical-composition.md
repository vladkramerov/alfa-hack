# HYP-100: frozen H78 decisions plus exact empirical offsets

SPEC011,2026-10-24. Registered beforefit. New combination of provenH78 properposterior/expectedF1 state and H98 empiricalprobability-spaceoffsets. No repeat H78fit: theta/gain/offset copied from matching savedH78inner/outeraudit; postprediction q=softmax(gain*softmax(logclip1e-6(basep)*exp(-theta0)+bias)+offset). Fit NEW empirical probabilityoffsets on q andtraininglabels, fixedH98algorithm3rarefirstpasses/exactbreakpoints/no tiedgroup splitting/strict1e-12. Inference softmax(q+newoffset), no querytargets/cohortstats. RetainH78baseline when trainingoffsets do nothelp; validationmetricnotusedduringfit.

Inner usesoldH78training rawIN-SAMPLE teacher+heldoutquery (a proxy; NOTindependentOOF meta). Outer usesH48OOF-base/frozenmeta in-sampletraining probabilities andH48heldoutquery throughfrozenH78. Trainingoptimization can overfit especiallyminorities; this isexplicitly differentfrom H98strictRAWOOFpipeline, with no independentinnerclaim. Allactualoutervalidationrows untouched. Inputstate/source/cache/data/splitsSHA frozenbeforetrial. ComparatorH48, incremental bestH78.6593084, user20%goal≥.7794374 separate; inner mustbeatH78.628848 beforeits oneouterfit. No oldteachers/oldH78parameterrefits, ≤900sec.

Two synthetic tests manualdouble-softmax/inputstateapplyonce, classalignment/statemutationguard. H98eightexact-breakpointtests inherited; no customlossgradientadded. Everyoutcome separate results including failures.

SPEC011 result 2026-10-24T02:48:40.758830+03:00 HYP-100/inner/default: completed macro_f1=0.550835584943435 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-100-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
