# H50 / secondary

First actual run of previously proposed H50; registered before fit in SPEC012 and HYP050. Class-balanced gamma2 focal boosting on raw median/OHE. Exact gradient checked by finite differences; positive surrogate curvature is not an exact Hessian. Fixed600 trees, no early stopping, no outer tuning, calibration, or SMOTE. H78 was MCC meta learning and is different.

```json
{
  "status": "completed",
  "metric": "macro_f1",
  "score": 0.608370390426356,
  "control": 0.6075743572072007,
  "relative_gain": 0.0013101823829668735,
  "stage": "secondary",
  "accuracy": 0.8074800045060269,
  "model_fit_count": 1,
  "objective_calls": 600,
  "seconds": 32.22700237506069,
  "total_seconds": 64.09558470803313,
  "max_rss_bytes": 990117888,
  "conclusion": "Подтверждена в проверенных условиях"
}
```
