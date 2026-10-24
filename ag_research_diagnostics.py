"""Разбор ошибок лучшего решения: где именно теряется метрика. Это то, с чего опытный ML-инженер
начинает следующую гипотезу — не «попробуем ещё модель», а «модель ошибается вот тут».

* классификация — F1 и полнота по классам, самые частые пары «истинный -> предсказанный»;
* регрессия — доля суммарной квадратичной ошибки на 1% худших строк (выбросы), ошибка у известных
  и новых объектов (если найден ключ объекта), сегменты категориальных признаков, где ошибка
  непропорционально велика.
Сводка идёт в промпт LLM и в отчёт."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from ag_ml_shift import category_columns


def classification(y: np.ndarray, proba: np.ndarray, idx: np.ndarray, classes: list, post: dict | None = None) -> dict:
    p = np.asarray(proba)[idx]
    if post and post.get("weights") is not None:
        p = p * np.asarray(post["weights"])
    pred = p.argmax(1) if p.ndim == 2 else (p > 0.5).astype(int)
    yt = y[idx]
    k = len(classes)
    f1 = skm.f1_score(yt, pred, average=None, labels=list(range(k)), zero_division=0)
    rec = skm.recall_score(yt, pred, average=None, labels=list(range(k)), zero_division=0)
    cm = skm.confusion_matrix(yt, pred, labels=list(range(k)))
    np.fill_diagonal(cm, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:4]
    worst = np.argsort(f1)[:4]
    return {
        "worst_classes": {str(classes[i]): {"f1": round(float(f1[i]), 3), "recall": round(float(rec[i]), 3),
                                            "n": int((yt == i).sum())} for i in worst},
        "top_confusions": [f"{classes[a]} -> {classes[b]}: {int(cm[a, b])}" for a, b in pairs if cm[a, b] > 0],
    }


def regression(y: np.ndarray, pred: np.ndarray, idx: np.ndarray, X: pd.DataFrame, seen: np.ndarray | None = None) -> dict:
    e = (y[idx] - np.asarray(pred)[idx]) ** 2
    tot = e.sum() or 1.0
    srt = np.sort(e)[::-1]
    out = {"sse_share_top1pct_rows": round(float(srt[: max(1, len(srt) // 100)].sum() / tot), 3),
           "rmse_without_top1pct": round(float(np.sqrt(srt[len(srt) // 100:].mean())), 4)}
    if seen is not None:
        s = seen[idx]
        if s.any() and (~s).any():
            out["rmse_known_objects"] = round(float(np.sqrt(e[s].mean())), 4)
            out["rmse_new_objects"] = round(float(np.sqrt(e[~s].mean())), 4)
            out["new_objects_share"] = round(float((~s).mean()), 3)
    segs = []
    Xi = X.iloc[idx].reset_index(drop=True)
    for c in category_columns(Xi)[:10]:
        g = pd.DataFrame({"v": Xi[c].astype(str), "e": e}).groupby("v")["e"].agg(["sum", "size"])
        g["sse_share"] = g["sum"] / tot
        g["row_share"] = g["size"] / len(e)
        g["lift"] = g["sse_share"] / g["row_share"]
        g = g[(g["size"] >= 30) & (g["sse_share"] >= 0.05)].sort_values("lift", ascending=False).head(2)
        for v, r in g.iterrows():
            segs.append(f"{c}={v}: {r['sse_share']:.0%} ошибки при {r['row_share']:.0%} строк")
    if segs:
        out["heavy_segments"] = segs[:6]
    return out
