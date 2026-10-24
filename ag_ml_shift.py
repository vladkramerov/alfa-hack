"""Валидация, похожая на тест по составу.

Сдвиг между фолдами валидации и тестом портит и выбор гипотез, и оценку качества. Типичный случай —
валидация по времени: в каком-то прошлом периоде появилась целая новая категория (новый регион,
новый тип объекта), и фолд этого периода почти целиком состоит из строк, которых модель «не видела».
В тесте таких строк может не быть вовсе: тогда ошибки на них раздувают CV и тянут выбор гипотез
в сторону того, что тесту не нужно.

* test_like_folds — убирает из проверочной части фолдов строки с категориями, которых нет в
  обучающей части фолда, если в тесте таких строк практически нет (доля < max_test_share).
  Эти строки остаются в обучении следующих фолдов и в финальном обучении — меняется только то,
  на чём измеряем качество и делаем раннюю остановку.
* composition — доля «известных» объектов (ключ объекта уже встречался раньше) в каждом фолде и в
  тесте; adjusted_estimate пересчитывает ошибку под долю известных объектов в тесте
  (для rmse / mse / mae). Это оценка качества на тесте, а не метрика для выбора гипотез."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_CARD = 100   # категориальные колонки с большим числом значений (id, текст) не учитываем


def category_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c])
            and X[c].nunique(dropna=True) <= MAX_CARD]


def new_category_rows(X_fit: pd.DataFrame, X_q: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Строки X_q, у которых хотя бы в одной колонке значение, которого нет в X_fit (пропуск — не новое)."""
    out = np.zeros(len(X_q), dtype=bool)
    for c in cols:
        known = set(X_fit[c].dropna().unique())
        q = X_q[c]
        out |= (q.notna() & ~q.isin(known)).to_numpy()
    return out


def test_like_folds(X_train: pd.DataFrame, X_test: pd.DataFrame, folds: list, max_test_share: float = 0.005,
                    min_fold_share: float = 0.01) -> tuple[list, dict]:
    cols = category_columns(X_train)
    info = {"columns": cols, "test_new_share": 0.0, "removed": []}
    if not cols:
        return folds, info
    test_new = new_category_rows(X_train, X_test, cols)
    info["test_new_share"] = round(float(test_new.mean()), 4)
    if test_new.mean() >= max_test_share:
        return folds, info   # в тесте новые категории есть — фолды с ними честно похожи на тест
    out = []
    for tr, va in folds:
        new = new_category_rows(X_train.iloc[tr], X_train.iloc[va], cols)
        if new.mean() >= min_fold_share and (~new).sum() >= max(30, len(va) // 10):
            info["removed"].append(int(new.sum()))
            out.append((tr, va[~new]))
        else:
            info["removed"].append(0)
            out.append((tr, va))
    if not any(info["removed"]):
        return folds, info
    return out, info


def composition(entity: np.ndarray | None, folds: list, n_train: int) -> dict | None:
    """Доля строк с известным объектом (ключ встречался в обучающей части) в проверочных строках и в тесте."""
    if entity is None:
        return None
    tr_ent, te_ent = entity[:n_train], entity[n_train:]
    seen = np.zeros(n_train, dtype=bool)
    for tr, va in folds:
        seen[va] = np.isin(tr_ent[va], tr_ent[tr])
    covered = np.concatenate([va for _, va in folds])
    return {"seen": seen, "valid_seen_share": float(seen[covered].mean()),
            "test_seen_share": float(np.isin(te_ent, tr_ent).mean())}


def adjusted_estimate(metric_name: str, y: np.ndarray, oof: np.ndarray, idx: np.ndarray, seen: np.ndarray,
                      test_seen_share: float) -> float | None:
    """Ошибка на OOF, пересчитанная под долю известных объектов в тесте (только rmse / mse / mae)."""
    if metric_name not in ("rmse", "mse", "mae"):
        return None
    s = seen[idx]
    if s.all() or (~s).all():
        return None
    err = np.abs(y[idx] - oof[idx]) if metric_name == "mae" else (y[idx] - oof[idx]) ** 2
    v = test_seen_share * err[s].mean() + (1 - test_seen_share) * err[~s].mean()
    return float(np.sqrt(v)) if metric_name == "rmse" else float(v)
