# HYP-081 neural calibration / secondary

SPEC012 extension. Started before any fit. New grouped neural OOF predictions augment the three existing tree families. H79 full-fit outer probabilities are reused unchanged; this runner never computes an outer score or trains the meta head. Inner-only epoch selection and frozen OOF-fit refits. Total ID budget900s, 60s reserved for the parent-owned meta head.

## Calibration outcome

```json
{
  "status": "calibration_ready",
  "stage": "secondary",
  "rows": 69916,
  "classes": 10,
  "folds": 3,
  "training_model_fits": 6,
  "outer_model_fits": 0,
  "outer_score_evaluations": 0,
  "audits": [
    {
      "status": "completed",
      "fold": 0,
      "selection": {
        "best_epoch": 18,
        "inner_score": 0.6077915612918509,
        "epochs_run": 24,
        "parameters": 74690,
        "normalized_class_weights": [
          3.5658369064331055,
          3.7180233001708984,
          1.4661575555801392,
          0.8769087791442871,
          1.176194667816162,
          0.8040580749511719,
          0.6670598387718201,
          1.5943148136138916,
          4.8841633796691895,
          12.781664848327637
        ]
      },
      "refit": {
        "best_epoch": 18,
        "inner_score": null,
        "epochs_run": 18,
        "parameters": 74818,
        "normalized_class_weights": [
          3.59061598777771,
          3.7852389812469482,
          1.468409538269043,
          0.8801001906394958,
          1.1830376386642456,
          0.7935650944709778,
          0.6726174354553223,
          1.5548741817474365,
          4.857892036437988,
          13.22404956817627
        ]
      },
      "total_elapsed": 68.27448458294384,
      "outer_score_evaluations": 0
    },
    {
      "status": "completed",
      "fold": 1,
      "selection": {
        "best_epoch": 7,
        "inner_score": 0.5866017767017435,
        "epochs_run": 13,
        "parameters": 74818,
        "normalized_class_weights": [
          3.502429723739624,
          3.7873430252075195,
          1.432835578918457,
          0.8777198791503906,
          1.1727954149246216,
          0.7987707853317261,
          0.6740146279335022,
          1.499000906944275,
          4.675786972045898,
          13.915587425231934
        ]
      },
      "refit": {
        "best_epoch": 7,
        "inner_score": null,
        "epochs_run": 7,
        "parameters": 74818,
        "normalized_class_weights": [
          3.5081965923309326,
          3.823620319366455,
          1.4396508932113647,
          0.8796519637107849,
          1.169280767440796,
          0.7940799593925476,
          0.6748120188713074,
          1.5275357961654663,
          4.650429725646973,
          13.52584457397461
        ]
      },
      "total_elapsed": 74.4563932500314,
      "outer_score_evaluations": 0
    },
    {
      "status": "completed",
      "fold": 2,
      "selection": {
        "best_epoch": 19,
        "inner_score": 0.5787628152848694,
        "epochs_run": 25,
        "parameters": 74434,
        "normalized_class_weights": [
          3.6104907989501953,
          3.8042566776275635,
          1.4639675617218018,
          0.8751119375228882,
          1.180077075958252,
          0.7995754480361938,
          0.671039879322052,
          1.5499595403671265,
          4.631316184997559,
          13.396169662475586
        ]
      },
      "refit": {
        "best_epoch": 19,
        "inner_score": null,
        "epochs_run": 19,
        "parameters": 74434,
        "normalized_class_weights": [
          3.6180214881896973,
          3.7765493392944336,
          1.4446663856506348,
          0.8694297671318054,
          1.179994821548462,
          0.8097672462463379,
          0.6687207818031311,
          1.5584741830825806,
          4.586359024047852,
          14.026268005371094
        ]
      },
      "total_elapsed": 83.64907845808193,
      "outer_score_evaluations": 0
    }
  ],
  "seconds": 25.086780542042106,
  "total_seconds": 83.77231399994344,
  "remaining_total_budget_seconds": 816.2276855839882,
  "conclusion": "Не оценивалась; только missing OOF calibration, head/outer score у parent"
}
```

## Head / final scientific assessment

```json
{
  "status": "completed",
  "metric": "macro_f1",
  "score": 0.6176160023719055,
  "control": 0.6075743572072007,
  "relative_gain": 0.01652743412487423,
  "stage": "secondary",
  "meta_fits": 1,
  "tree_fits": 0,
  "head_seconds": 7.040339458035305,
  "selection": "fixed preregistered four-family normalized head; no outer tuning",
  "warnings": []
}
```
