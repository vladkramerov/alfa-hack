# refit

{
  "candidate": {
    "family": "xgboost",
    "parameters": {},
    "strategy": "macro_f1_calibrated",
    "calibration_folds": 3,
    "hypothesis": "Grouped OOF class decision multipliers for Macro-F1",
    "preprocessing": "macro-f1-v1",
    "scale": false,
    "max_outlier_fraction": 0,
    "target_transform": "none",
    "class_multipliers": {
      "001": 1.0,
      "010": 1.0
    }
  },
  "result": {
    "status": "completed",
    "mode": "refit",
    "train_rows": 80,
    "test_rows": 3,
    "fit_retained_rows": 80,
    "calibration_multipliers": {
      "001": 1.0,
      "010": 1.0
    },
    "seconds": 0.10999999998603016
  }
}