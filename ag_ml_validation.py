"""Схема валидации. Фолды фиксируются один раз на весь запуск, чтобы OOF разных
экспериментов были сравнимы и их можно было ансамблировать."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold

from ag_schemas import MULTILABEL, RANKING, TaskSpec


def make_folds(task: TaskSpec, y: np.ndarray, n_folds: int, seed: int,
               groups: np.ndarray | pd.Series | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """groups — строки одной группы целиком в одном фолде (ранжирование с новыми группами в тесте)."""
    n = len(y)
    idx = np.arange(n)
    if groups is not None and len(np.unique(np.asarray(groups))) >= n_folds:
        g = np.asarray(groups)
        # GroupKFold без перемешивания раскладывает группы по размеру; перемешиваем коды групп
        perm = np.random.default_rng(seed).permutation(np.unique(g))
        g = perm[np.searchsorted(np.unique(g), g)]
        return [(tr, va) for tr, va in GroupKFold(n_splits=n_folds).split(idx, y, g)]
    strat = None
    if task.task_type == MULTILABEL:
        # стратификация по самой редкой из присутствующих меток: она сильнее всего страдает от неравных фолдов
        Y = np.asarray(y)
        pos = Y.sum(0)
        ok = np.flatnonzero((pos >= n_folds) & (pos <= n - n_folds))
        if len(ok):
            strat = Y[:, ok[np.argmin(pos[ok])]]
    elif task.is_classification:
        strat = np.asarray(y, dtype=int)
    if strat is not None:
        counts = np.bincount(strat)
        k = int(min(n_folds, max(2, counts[counts > 0].min())))
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in splitter.split(idx, strat)]
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in splitter.split(idx)]


def split_halves(n: int, y: np.ndarray, classification: bool, seed: int,
                 groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Две половины OOF: на A подбираются веса ансамбля и постобработка, на B проверяются.
    groups — группа целиком в одной половине (ранжирование: метрика считается внутри группы)."""
    rng = np.random.default_rng(seed + 1)
    if groups is not None:
        g = np.asarray(groups)
        u = rng.permutation(np.unique(g))
        in_a = np.isin(g, u[: len(u) // 2])
        return np.flatnonzero(in_a), np.flatnonzero(~in_a)
    if classification:
        a, b = [], []
        for c in np.unique(y):
            ids = np.flatnonzero(y == c)
            rng.shuffle(ids)
            h = len(ids) // 2
            a.append(ids[:h])
            b.append(ids[h:])
        return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))
    perm = rng.permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def fold_groups(task: TaskSpec, ds) -> np.ndarray | None:
    """Группы для фолдов: при ранжировании, если группы теста в train почти не встречаются, валидация
    тоже должна быть на новых группах; иначе строки одной группы делятся между фолдами, как в тесте."""
    if task.task_type != RANKING or ds.groups is None:
        return None
    n = len(ds.y)
    tr, te = ds.groups[:n], ds.groups[n:]
    if len(te) and np.isin(te, tr).mean() >= 0.5:
        return None
    return tr


def unseen_duplicate_folds(task: TaskSpec, train: pd.DataFrame, test: pd.DataFrame,
                           y: np.ndarray, n_folds: int, seed: int):
    """Match unseen duplicate groups only when feature-only test overlap supports it.

    Transfer of team's raw-X duplicate grouping; do not group arbitrary repeated
    IDs or force grouped CV when deployment contains known duplicate rows.
    """
    if not task.is_classification or train.empty or test.empty:
        return None
    both = pd.concat([train, test], ignore_index=True)
    keys = pd.util.hash_pandas_object(both, index=False).to_numpy()
    groups = keys[:len(train)]
    duplicate_share = 1 - len(np.unique(groups)) / len(groups)
    overlap = np.isin(keys[len(train):], groups).mean()
    if duplicate_share < 0.05 or overlap >= 0.01:
        return None
    class_groups = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    k = min(n_folds, min(class_groups))
    if k < 2:
        return None
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [(a,b) for a,b in splitter.split(train, y, groups)]
    return folds, {"duplicate_share":round(duplicate_share,4), "test_exact_overlap":round(float(overlap),4)}
