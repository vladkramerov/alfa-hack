# evaluate

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
    "target_transform": "none"
  },
  "result": {
    "status": "completed",
    "metric": "macro_f1",
    "score": 1.0,
    "fit_rows": 64,
    "valid_rows": 16,
    "fit_retained_rows": 64,
    "calibration_multipliers": {
      "001": 1.0,
      "010": 1.0
    },
    "seconds": 0.26600000000325963
  }
}