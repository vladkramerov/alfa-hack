# evaluate

{
  "candidate": {
    "family": "extra_trees",
    "parameters": {
      "n_estimators": 12,
      "max_depth": 6
    },
    "hypothesis": "Synthetic interaction protocol check",
    "feature_code": "def engineer(X):\n    X = X.copy()\n    X[\"signal_square\"] = X[\"signal\"] ** 2\n    return X\n"
  },
  "result": {
    "status": "completed",
    "metric": "rmse",
    "score": 0.29579904484172626,
    "fit_rows": 80,
    "valid_rows": 20,
    "seconds": 0.09399999998277053
  }
}