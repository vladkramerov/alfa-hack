"""EDA без LLM: компактный профиль данных, который помещается в промпт."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ag_schemas import MULTILABEL, TaskSpec


def profile(train: pd.DataFrame, test: pd.DataFrame, task: TaskSpec, feature_cols: list[str],
            datetime_cols: list[str]) -> dict:
    y = train[task.target]
    cols = []
    for c in feature_cols:
        s = train[c]
        info = {
            "name": c,
            "dtype": "numeric" if pd.api.types.is_numeric_dtype(s) else ("datetime" if c in datetime_cols else "categorical"),
            "nunique": int(s.nunique()),
            "null": round(float(s.isna().mean()), 3),
        }
        if info["dtype"] == "numeric" and s.notna().any():
            q = s.quantile([0.01, 0.5, 0.99])
            info["p01_p50_p99"] = [round(float(v), 4) for v in q]
            if s.nunique() > 2 and s.std() > 0:
                info["skew"] = round(float(s.skew()), 2)
        elif info["dtype"] == "categorical":
            info["top"] = [str(v) for v in s.value_counts().head(3).index]
            unseen = ~test[c].isin(s.unique()) if c in test.columns else pd.Series(dtype=bool)
            if len(unseen):
                info["test_unseen_share"] = round(float(unseen.mean()), 3)
        if c in task.feature_descriptions:
            info["desc"] = task.feature_descriptions[c][:120]
        cols.append(info)

    prof = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(feature_cols),
        "n_categorical": sum(1 for c in cols if c["dtype"] == "categorical"),
        "n_datetime": len(datetime_cols),
        "duplicated_rows_share": round(float(train[feature_cols].duplicated().mean()), 4),
        "columns": cols,
    }
    # полные дубли строк с разными таргетами: их различает только «память по дублям» (row_memory)
    if prof["duplicated_rows_share"] > 0:
        key = pd.util.hash_pandas_object(train[feature_cols], index=False)
        nun = y.groupby(key.to_numpy()).transform("nunique")
        prof["conflicting_duplicates_share"] = round(float((nun > 1).mean()), 4)
    if task.task_type == MULTILABEL:
        prev = {c: round(float(pd.to_numeric(train[c], errors="coerce").mean()), 4) for c in task.targets}
        prof["label_prevalence"] = prev
        vals = [v for v in prev.values() if v > 0]
        prof["imbalance_ratio"] = round(float(max((1 - v) / v for v in vals)), 1) if vals else 1.0
    elif task.is_classification:
        vc = y.value_counts()
        prof["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        prof["imbalance_ratio"] = round(float(vc.max() / max(vc.min(), 1)), 1)
    else:
        prof["target"] = {
            "mean": round(float(y.mean()), 4), "std": round(float(y.std()), 4),
            "min": round(float(y.min()), 4), "max": round(float(y.max()), 4),
            "skew": round(float(y.skew()), 2), "nonnegative": bool((y >= 0).all()),
            "zero_share": round(float((y == 0).mean()), 4),
        }
    # признаки, подозрительно сильно связанные с таргетом (кандидаты в утечку)
    leaks = []
    if task.task_type == MULTILABEL:
        prof["possible_leaks"] = leaks
        return prof
    yn = pd.factorize(y)[0] if task.is_classification else y.to_numpy(dtype=float)
    for c in feature_cols:
        s = train[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            corr = np.corrcoef(np.nan_to_num(s.to_numpy(dtype=float)), yn)[0, 1]
            if np.isfinite(corr) and abs(corr) > 0.98:
                leaks.append(c)
    prof["possible_leaks"] = leaks
    return prof


def compact(prof: dict, max_cols: int = 80) -> dict:
    """Урезанная версия для промпта."""
    p = dict(prof)
    p["columns"] = prof["columns"][:max_cols]
    if len(prof["columns"]) > max_cols:
        p["columns_truncated"] = len(prof["columns"]) - max_cols
    return p
