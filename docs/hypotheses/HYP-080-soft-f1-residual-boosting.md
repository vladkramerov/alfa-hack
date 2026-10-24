# HYP-080: exact coupled soft-MacroF1 gradient in residual tree boosting

SPEC011,2026-10-24, registered before realfit. Combine beneficial H72 objective with stronger residual-tree representation from H77. Distinct from H72 MLP40epochs and H77 rank-IoU surrogate: optimize .5meanCE+.5(1-softMacroF1) using exact wholetraining-set gradient and LightGBM trees, positive approximate diagonal curvature max(.5p(1-p),.001). This is a functional-gradient heuristic, not exact Newton nor additive per-row F1 likelihood; no convergence claim.

Frozen300rounds,63leaves,minchild30,lambda5,lr.05,2CPU, teacherlogp init/query added once. Trainonly median/log1p nonnegative/scaling/clip8/OHE, no rowdeletion, no classweights/earlystop/outertuning. Rest same H77 frozenparams. No oldmodelrefits, cached teacher provenance preserved. Customgradient finite-difference establishedH72; newadapter2tests verify exactN scaling/positivehessian/configuration.

Success inner rawteacher MacroF1>.587338; outer only selected byinner beforeits score. Bestexisting innerH78.628848; noouter ifnotbetter unless predeclared newmechanismcombo innerbeatsit. User20% originalouter>=.779437 independently assessed. Separate results and no replay;≤900sec.

SPEC011 result 2026-10-24T02:09:12.796389+03:00 HYP-080/inner/default: completed macro_f1=0.6140917642777984 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-080-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
