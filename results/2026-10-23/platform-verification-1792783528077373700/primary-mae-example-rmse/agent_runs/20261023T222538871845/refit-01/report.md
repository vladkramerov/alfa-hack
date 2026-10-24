# refit

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
    "mode": "refit",
    "train_rows": 100,
    "test_rows": 3,
    "fit_retained_rows": 95,
    "seconds": 0.264999999984866
  }
}