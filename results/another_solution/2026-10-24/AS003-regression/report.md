# AS003/regression

Only cached original OOF/test predictions, no model refits. A for fitting, B for selection; exploratory local validation.

```json
{
  "status": "not_applicable_unchanged",
  "metric": "rmse",
  "oof_score": 29.373823495698232,
  "holdout_B_score": 28.685722725959646,
  "control_oof_score": 29.373823495698232,
  "control_holdout_B_score": 28.685722725959646,
  "relative_gain_B": -0.0,
  "relative_gain_OOF": -0.0,
  "selected": "ensemble",
  "members": {
    "3": 0.143,
    "5": 0.429,
    "7": 0.143,
    "1": 0.143,
    "0": 0.143
  },
  "post": {},
  "choice_reason": "ансамбль взят: выигрыш на B +1.23%, лучше в 94% бутстрэп-выборок",
  "seconds": 0.048571249935775995,
  "new_tree_fits": 0
}
```
