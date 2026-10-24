# HYP-096 / HYP-097: honest OOF training probabilities for expected-F1 decisions

SPEC011,2026-10-24; registered before fits. These are new combinations of measured H78/H75 with H88's honest nested raw OOF teacher, replacing the previous in-sample raw training probabilities while preserving the held-out query teacher. They do not repeat the previous calibrator fits: training probability matrices are different and each row's target was excluded from its teacher.

H96: unchanged CalibratedMacroDecision penalty.01/L-BFGS100/scalartemperature[-2.3,2.3]/bias[-5,5], then fixed25/damp.5 expectedF1. H97: unchanged MacroF1Decision25/damp.5/eps1e-8 without CEcalibration. No parameter grid or new learners/teachers. Read-only H88 cache validation exactlyasH92–94; inner47,075/23,454 andoriginalouter70,529/17,141. Outer queryteacherRAW H11; comparator remainsH48. Allsource/config/data/splits/teacherSHA frozen beforefittingparameters, oldteachersnotrefit.

Criterion innerstrict>bestH78.6288481661155734 beforeanyoutertransfer, goaloriginalouter≥.7794373944 separate. Failed/negativeoutcomes preserved; no repeats. Existing analyticgradients/calibration/fixedpath/read-onlyOOF tests apply because algorithms unchanged; newprotocol's teacherSHA andleakage provenance requireindependentverification. ≤900seconds perID.

SPEC011 result 2026-10-24T02:40:18.827293+03:00 HYP-096/inner/default: completed macro_f1=0.6291379699977411 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-096-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:40:57.417483+03:00 HYP-097/inner/default: completed macro_f1=0.6334156298867437 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-097-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:51:04.640764+03:00 HYP-097/outer/default: completed macro_f1=0.6384122468214593 vscontrol=0.6495311620074445; [report](../../results/2026-10-24/HYP-097-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.
