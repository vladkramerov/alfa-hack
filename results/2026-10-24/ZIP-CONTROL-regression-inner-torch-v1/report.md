# Previous ZIP control regression/inner

Started before fit: 2026-10-24T14:26:39.427844+03:00

SPEC016 REQ001: explicitly authorized actual ZIP comparison; not a replay of cached H107/H108.

```json
{
  "status": "completed",
  "kind": "regression",
  "stage": "inner",
  "role": "explicit_previous_zip_control",
  "metric": "rmse",
  "score": 30.656905391248326,
  "fit_rows": 22054,
  "valid_rows": 12946,
  "seconds": 146.59399999998277,
  "zip_sha256": "9a3924a53f6bf6191d27dd0717c44afc4137757557e469efe77c3eac28d1756a",
  "actual_archived_factory": true,
  "previous_quality_trials_replayed": false
}
```

Command: `.venv-ml/Scripts/python.exe -X utf8 -m experiments.torch_zip_control --kind regression --stage inner`.
