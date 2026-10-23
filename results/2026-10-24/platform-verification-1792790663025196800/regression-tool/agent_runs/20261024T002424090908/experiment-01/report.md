# evaluate

{
  "candidate": {
    "family": "extra_trees",
    "parameters": {
      "n_estimators": 12,
      "max_depth": 6
    },
    "hypothesis": "Synthetic interaction protocol check",
    "preprocessing": "full-v1",
    "scale": true,
    "max_outlier_fraction": 0.05,
    "feature_code": "def engineer(X):\n    X = X.copy()\n    X[\"signal_square\"] = X[\"signal\"] ** 2\n    return X\n"
  },
  "result": {
    "status": "completed",
    "metric": "rmse",
    "score": 0.7126799108403095,
    "fit_rows": 80,
    "valid_rows": 20,
    "fit_retained_rows": 76,
    "seconds": 0.28200000000651926
  }
}