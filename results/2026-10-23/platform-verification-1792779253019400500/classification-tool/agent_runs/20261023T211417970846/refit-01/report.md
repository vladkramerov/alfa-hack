# refit

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
    "mode": "refit",
    "train_rows": 100,
    "test_rows": 3,
    "seconds": 0.07800000000861473
  }
}