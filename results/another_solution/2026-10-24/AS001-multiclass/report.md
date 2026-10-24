# AS001/multiclass

Only cached original OOF/test predictions, no model refits. A for fitting, B for selection; exploratory local validation.

```json
{
  "status": "completed",
  "metric": "macro_f1",
  "oof_score": 0.6531702899869024,
  "holdout_B_score": 0.6517756575501228,
  "control_oof_score": 0.6534905279816051,
  "control_holdout_B_score": 0.6481770971418532,
  "relative_gain_B": 0.005551816662664362,
  "relative_gain_OOF": -0.0004900422898122317,
  "selected": "exp0:lgbm",
  "members": {
    "0": 1.0
  },
  "post": {
    "weights": [
      0.3333333333333333,
      0.5,
      1.0,
      0.5,
      0.5,
      0.5,
      0.5,
      0.3333333333333333,
      0.3333333333333333,
      0.6666666666666666
    ]
  },
  "choice_reason": "ансамбль не собран: ни одна модель не улучшила лучшую одиночную на половине A",
  "seconds": 1.8765549999661744,
  "new_tree_fits": 0,
  "win_rate_vs_original_on_B": 0.945
}
```
