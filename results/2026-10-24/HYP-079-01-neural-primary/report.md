# HYP-079 / primary

SPEC012/PLAN012. Started before fit.

Original small PLE neural implementation; BatchEnsemble-inspired for H80, not full TabM. No pretrained weights, external data, LLM calls or outer-label model selection. Both protocols share one 900-second budget.

## Result

```json
{
  "status": "completed",
  "metric": "macro_f1",
  "score": 0.593065996862908,
  "control": 0.64472256711304,
  "relative_gain": -0.08012216864292732,
  "accuracy": 0.8021702351088035,
  "weighted_f1": 0.8101506963200782,
  "fit_rows": 70529,
  "valid_rows": 17141,
  "frozen_epochs": 16,
  "selection": {
    "best_epoch": 16,
    "inner_score": 0.5855994363818925,
    "epochs_run": 22,
    "parameters": 74818,
    "normalized_class_weights": [
      3.583463191986084,
      3.7835376262664795,
      1.4393707513809204,
      0.8755284547805786,
      1.1920108795166016,
      0.7887730002403259,
      0.6773996353149414,
      1.5547959804534912,
      4.69133186340332,
      14.345046997070312
    ]
  },
  "refit": {
    "best_epoch": 16,
    "inner_score": null,
    "epochs_run": 16,
    "parameters": 74946,
    "normalized_class_weights": [
      3.6056506633758545,
      3.7876436710357666,
      1.441420078277588,
      0.8745965361595154,
      1.189223051071167,
      0.7940875887870789,
      0.6761749982833862,
      1.5575953722000122,
      4.77968168258667,
      14.932868003845215
    ]
  },
  "seconds": 14.966782708885148,
  "total_seconds": 31.560619834112003,
  "max_rss_bytes": 870924288,
  "conclusion": "Не подтверждена по заданному критерию"
}
```
