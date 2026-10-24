# H82 / primary

Registered before fit in HYP-082.md and SPEC012. ModernNCA-inspired learned metric, not an exact paper replication and not fixed-distance H52. Inner grouped validation selects epochs; fresh outer-fit refit uses the frozen epoch count. Outer labels are used once for scoring only. Total900s across both protocols.

```json
{
  "status": "completed",
  "metric": "macro_f1",
  "score": 0.4600746394936919,
  "stage": "primary",
  "control": 0.64472256711304,
  "relative_gain": -0.28639904516786296,
  "accuracy": 0.7996032903564553,
  "fit_rows": 70529,
  "valid_rows": 17141,
  "frozen_epochs": 6,
  "model_fit_count": 2,
  "seconds": 78.43368487502448,
  "total_seconds": 78.45256229187362,
  "max_rss_bytes": 1308753920,
  "conclusion": "Не подтверждена по заданному критерию"
}
```
