# HYP-081 neural calibration / primary

SPEC012 extension. Started before any fit. New grouped neural OOF predictions augment the three existing tree families. H79 full-fit outer probabilities are reused unchanged; this runner never computes an outer score or trains the meta head. Inner-only epoch selection and frozen OOF-fit refits. Total ID budget900s, 60s reserved for the parent-owned meta head.

## Calibration outcome

```json
{
  "status": "calibration_ready",
  "stage": "primary",
  "rows": 70529,
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
        "best_epoch": 23,
        "inner_score": 0.5835574073498101,
        "epochs_run": 29,
        "parameters": 74818,
        "normalized_class_weights": [
          3.6119327545166016,
          3.807816505432129,
          1.4239237308502197,
          0.8828659057617188,
          1.1738580465316772,
          0.7993961572647095,
          0.6740277409553528,
          1.5336582660675049,
          4.843605041503906,
          15.091800689697266
        ]
      },
      "refit": {
        "best_epoch": 23,
        "inner_score": null,
        "epochs_run": 23,
        "parameters": 74818,
        "normalized_class_weights": [
          3.6232686042785645,
          3.7827308177948,
          1.4305932521820068,
          0.8784246444702148,
          1.1829156875610352,
          0.7929847240447998,
          0.6768907308578491,
          1.5469927787780762,
          4.898881912231445,
          14.183385848999023
        ]
      },
      "total_elapsed": 11.359326791018248,
      "outer_score_evaluations": 0
    },
    {
      "status": "completed",
      "fold": 1,
      "selection": {
        "best_epoch": 15,
        "inner_score": 0.5985276218581793,
        "epochs_run": 21,
        "parameters": 74818,
        "normalized_class_weights": [
          3.5174448490142822,
          3.683311939239502,
          1.4041281938552856,
          0.861847460269928,
          1.1911510229110718,
          0.7982326149940491,
          0.6783234477043152,
          1.572694182395935,
          4.701141834259033,
          15.112055778503418
        ]
      },
      "refit": {
        "best_epoch": 15,
        "inner_score": null,
        "epochs_run": 15,
        "parameters": 74818,
        "normalized_class_weights": [
          3.5144238471984863,
          3.7343931198120117,
          1.4204362630844116,
          0.8670769929885864,
          1.1864014863967896,
          0.7988959550857544,
          0.6761921048164368,
          1.58352530002594,
          4.72367525100708,
          15.59017276763916
        ]
      },
      "total_elapsed": 19.5302572499495,
      "outer_score_evaluations": 0
    },
    {
      "status": "completed",
      "fold": 2,
      "selection": {
        "best_epoch": 15,
        "inner_score": 0.5799450173511452,
        "epochs_run": 21,
        "parameters": 74562,
        "normalized_class_weights": [
          3.681891679763794,
          3.8573172092437744,
          1.4852038621902466,
          0.8817281126976013,
          1.1996408700942993,
          0.7896934151649475,
          0.675226092338562,
          1.5194716453552246,
          4.69568395614624,
          15.450218200683594
        ]
      },
      "refit": {
        "best_epoch": 15,
        "inner_score": null,
        "epochs_run": 15,
        "parameters": 74818,
        "normalized_class_weights": [
          3.685225248336792,
          3.8484010696411133,
          1.47490656375885,
          0.878510057926178,
          1.1986409425735474,
          0.7906084060668945,
          0.6755471229553223,
          1.5436245203018188,
          4.7235002517700195,
          15.134869575500488
        ]
      },
      "total_elapsed": 27.991399666061625,
      "outer_score_evaluations": 0
    }
  ],
  "seconds": 28.116337249986827,
  "total_seconds": 28.11657404107973,
  "remaining_total_budget_seconds": 871.8834255000111,
  "conclusion": "Не оценивалась; только missing OOF calibration, head/outer score у parent"
}
```

## Head / final scientific assessment

```json
{
  "status": "completed",
  "metric": "macro_f1",
  "score": 0.6351657518194138,
  "control": 0.64472256711304,
  "relative_gain": -0.014823143753785462,
  "stage": "primary",
  "meta_fits": 1,
  "tree_fits": 0,
  "head_seconds": 7.511741624912247,
  "selection": "fixed preregistered four-family normalized head; no outer tuning",
  "warnings": []
}
```
