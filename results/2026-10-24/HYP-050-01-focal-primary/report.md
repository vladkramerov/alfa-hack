# H50 / primary

First actual run of previously proposed H50; registered before fit in SPEC012 and HYP050. Class-balanced gamma2 focal boosting on raw median/OHE. Exact gradient checked by finite differences; positive surrogate curvature is not an exact Hessian. Fixed600 trees, no early stopping, no outer tuning, calibration, or SMOTE. H78 was MCC meta learning and is different.

```json
{
  "status": "completed",
  "metric": "macro_f1",
  "score": 0.5812249311112445,
  "control": 0.64472256711304,
  "relative_gain": -0.09848830991929958,
  "stage": "primary",
  "accuracy": 0.8114462400093343,
  "model_fit_count": 1,
  "objective_calls": 600,
  "seconds": 31.825899416115135,
  "total_seconds": 31.84197429101914,
  "max_rss_bytes": 867057664,
  "conclusion": "Не подтверждена по заданному критерию"
}
```
