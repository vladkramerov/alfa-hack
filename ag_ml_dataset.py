"""Данные задачи в форме, которую понимают эксперименты: таргет под тип задачи, признаки с учётом
типов из data_dictionary, текст (TF-IDF), «сетки» пикселей и сигналов, группы ранжирования.
Общая часть агента и выгруженного кода решения (agent/export.py).

Решения принимаются по данным и по ролям/типам колонок из data_dictionary, а не по именам задач:
* текстовая колонка (тип text или почти уникальные строки из нескольких слов) уходит из таблицы:
  деревья получают её простые статистики (длина, слова, цифры...), а линейная модель на TF-IDF —
  сам текст (models.text_linear, блок text_oof);
* колонка с типом category из data_dictionary — категория, даже если записана числом (id фильма);
* «сетка» (pixel_r_c, acc_x_0..acc_x_127) — массив для свёрточной сети, если соседние колонки
  действительно похожи (картинка или сигнал гладкие); к многоканальным сигналам добавляются статистики
  каналов (среднее, разброс, квантили, энергия по частотам);
* группа ранжирования — коды групп; как признак остаётся, только если группы теста встречаются в train."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ag_ml_preprocess import detect_datetime_cols, detect_grid, detect_text_cols, text_stats
from ag_schemas import MULTILABEL, RANKING, TaskSpec

log = logging.getLogger("agent.dataset")


@dataclass
class Dataset:
    X_train: pd.DataFrame          # признаки train (без id и таргета; текст заменён статистиками)
    X_test: pd.DataFrame
    y: np.ndarray                  # форма по типу задачи: см. agent/ml/metrics.py
    test_ids: pd.Series | None
    datetime_cols: list[str]
    period: np.ndarray | None = None    # код периода по строкам train+test (agent/ml/structure.py)
    entity: np.ndarray | None = None    # код объекта по строкам train+test
    refit: bool = False                 # тест предсказывает модель, дообученная на всём train
    text: Any = None                    # разреженная матрица TF-IDF по строкам train+test
    grid: np.ndarray | None = None      # (n, C, H, W) или (n, C, L) по строкам train+test
    groups: np.ndarray | None = None    # коды групп ранжирования по строкам train+test
    info: dict = field(default_factory=dict)   # что найдено: текст, сетка, группы (для профиля и отчёта)


def _to01(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.astype(float).round().astype(int).to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes", "y", "t", "да"]).astype(int).to_numpy()


def group_codes(train: pd.DataFrame, test: pd.DataFrame, col: str | None) -> np.ndarray:
    if not col:
        return np.zeros(len(train) + len(test), dtype=np.int64)
    both = pd.concat([train[col], test[col]], ignore_index=True).astype(str)
    return pd.factorize(both)[0].astype(np.int64)


def encode_target(task: TaskSpec, train: pd.DataFrame, groups: np.ndarray | None = None) -> np.ndarray:
    if task.task_type == MULTILABEL:
        return np.column_stack([_to01(train[c]) for c in task.targets])
    if task.is_classification:
        mapping = {c: i for i, c in enumerate(task.classes)}
        y = train[task.target].map(mapping)
        if y.isna().any():
            raise ValueError("в train есть классы, которых нет в описании задачи")
        return y.to_numpy(dtype=int)
    y = train[task.target].to_numpy(dtype=float)
    if task.task_type == RANKING:
        g = groups[:len(train)] if groups is not None else np.zeros(len(train))
        return np.column_stack([y, g.astype(float)])
    return y


def feature_columns(task: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    skip = {task.id_col, task.target, *task.targets}
    return [c for c in test.columns if c in train.columns and c not in skip]


def _adjacent_similarity(V: np.ndarray, shape: tuple) -> tuple[float, float]:
    """Средняя корреляция соседних колонок сетки и случайных пар: у картинки и сигнала соседние значения
    похожи, у безымянных признаков f1..f300 — нет."""
    n = len(V)
    A = V.reshape(n, shape[0], -1) if len(shape) == 2 else V.reshape(n, shape[0], shape[1], shape[2])
    if A.ndim == 3:
        a, b = A[:, :, :-1].reshape(n, -1), A[:, :, 1:].reshape(n, -1)
    else:
        a, b = A[:, :, :, :-1].reshape(n, -1), A[:, :, :, 1:].reshape(n, -1)
    rng = np.random.default_rng(0)
    j = rng.choice(a.shape[1], size=min(400, a.shape[1]), replace=False)
    flat = V.reshape(n, -1)
    r1, r2 = rng.choice(flat.shape[1], size=400), rng.choice(flat.shape[1], size=400)

    def corr(x, z):
        x, z = x - x.mean(0), z - z.mean(0)
        den = np.sqrt((x ** 2).sum(0) * (z ** 2).sum(0))
        c = np.where(den > 0, (x * z).sum(0) / np.where(den > 0, den, 1), np.nan)
        return float(np.nanmean(c)) if np.isfinite(c).any() else 0.0

    return corr(a[:, j], b[:, j]), corr(flat[:, r1], flat[:, r2])


def signal_stats(G: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Статистики каналов сигнала (n, C, L): уровень, разброс, форма распределения, «дрожание»,
    энергия в четырёх полосах частот и доминирующая частота."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in range(G.shape[1]):
            x = G[:, c, :].astype(np.float64)
            p = names[c]
            q = np.nanpercentile(x, [10, 50, 90], axis=1)
            out.update({f"{p}__mean": np.nanmean(x, 1), f"{p}__std": np.nanstd(x, 1), f"{p}__min": np.nanmin(x, 1),
                        f"{p}__max": np.nanmax(x, 1), f"{p}__p10": q[0], f"{p}__p50": q[1], f"{p}__p90": q[2],
                        f"{p}__absdiff": np.nanmean(np.abs(np.diff(x, axis=1)), 1)})
            xc = np.nan_to_num(x - np.nanmean(x, 1, keepdims=True))
            out[f"{p}__zcr"] = (np.diff(np.sign(xc), axis=1) != 0).mean(1)
            spec = np.abs(np.fft.rfft(xc, axis=1)) ** 2
            tot = spec[:, 1:].sum(1) + 1e-12
            bands = np.array_split(np.arange(1, spec.shape[1]), 4)
            for i, b in enumerate(bands):
                out[f"{p}__band{i}"] = spec[:, b].sum(1) / tot
            out[f"{p}__domfreq"] = spec[:, 1:].argmax(1).astype(float)
        if G.shape[1] >= 2:   # связь каналов (оси одного датчика): корреляция пар и модуль вектора
            mag = np.sqrt(np.nansum(G.astype(np.float64) ** 2, axis=1))
            out["__sig_mag_mean"], out["__sig_mag_std"] = np.nanmean(mag, 1), np.nanstd(mag, 1)
    return pd.DataFrame(out).astype("float32")


def text_matrix(texts: dict[str, pd.Series], max_features: int = 200_000, max_chars: int = 60_000_000):
    """TF-IDF по каждой текстовой колонке: слова и пары слов + символьные n-граммы (устойчивы к
    морфологии и опечаткам). Словарь строится по train+test без таргета."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    parts = []
    per_col = max(20_000, max_features // max(1, len(texts)))
    for name, s in texts.items():
        t = s.fillna("").astype(str)
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=per_col // 2, sublinear_tf=True,
                               token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        try:
            parts.append(word.fit_transform(t))
        except ValueError:   # пустой словарь
            continue
        if t.str.len().sum() <= max_chars:
            char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=per_col // 2,
                                   sublinear_tf=True, dtype=np.float32)
            try:
                parts.append(char.fit_transform(t))
            except ValueError:
                pass
    if not parts:
        return None
    return sparse.hstack(parts).tocsr()


def build(task: TaskSpec, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str] | None = None,
          with_text: bool = True, with_grid: bool = True) -> Dataset:
    feature_cols = feature_columns(task, train, test) if feature_cols is None else list(feature_cols)
    types = task.column_types or {}
    info: dict = {}
    groups = group_codes(train, test, task.group_col) if task.task_type == RANKING else None
    y = encode_target(task, train, groups)
    cols = list(feature_cols)
    if task.group_col and task.group_col in cols:
        seen = test[task.group_col].isin(set(train[task.group_col])).mean()
        info["test_groups_seen_share"] = round(float(seen), 3)
        if seen < 0.5:   # группы теста новые: id группы как признак ничего не даёт
            cols.remove(task.group_col)
    X_tr, X_te = train[cols].copy(), test[cols].copy()
    # категории из data_dictionary, записанные числами (id товара, код региона)
    for c in cols:
        if types.get(c) == "category" and pd.api.types.is_numeric_dtype(X_tr[c]) and X_tr[c].nunique() > 2:
            for X in (X_tr, X_te):
                X[c] = X[c].map(lambda v: None if pd.isna(v) else str(int(v)) if float(v).is_integer() else str(v))
    dt_cols = detect_datetime_cols(X_tr, types)
    text_cols = detect_text_cols(X_tr, types, exclude=set(dt_cols))
    text = None
    if text_cols:
        info["text_columns"] = text_cols
        both = pd.concat([X_tr[text_cols], X_te[text_cols]], ignore_index=True)
        if with_text:
            text = text_matrix({c: both[c] for c in text_cols})
            if text is not None:
                info["text_vocabulary"] = int(text.shape[1])
        stats = pd.concat([text_stats(both[c], c) for c in text_cols], axis=1)
        X_tr = pd.concat([X_tr.drop(columns=text_cols).reset_index(drop=True),
                          stats.iloc[:len(X_tr)].reset_index(drop=True)], axis=1)
        X_te = pd.concat([X_te.drop(columns=text_cols).reset_index(drop=True),
                          stats.iloc[len(X_tr):].reset_index(drop=True)], axis=1)
    grid = None
    if with_grid:
        num = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c]) and c not in dt_cols]
        g = detect_grid(num, X_tr) if len(num) >= 64 else None
        if g is not None:
            V = X_tr[g["columns"]].head(3000).to_numpy(dtype=np.float32)
            V = np.nan_to_num(V, nan=0.0)
            adj, rnd = _adjacent_similarity(V, g["shape"])
            if adj >= 0.5 and adj - rnd >= 0.2:
                grid = np.nan_to_num(np.vstack([X_tr[g["columns"]].to_numpy(np.float32),
                                                X_te[g["columns"]].to_numpy(np.float32)]), nan=0.0)
                grid = grid.reshape((len(grid),) + tuple(g["shape"]))
                info["grid"] = {"kind": g["kind"], "shape": list(g["shape"]),
                                "adjacent_corr": round(adj, 2), "random_corr": round(rnd, 2)}
                if g["kind"] == "sequence" and g["shape"][0] >= 2:
                    prefixes = _channel_names(g["columns"], g["shape"])
                    st = signal_stats(grid, prefixes)
                    X_tr = pd.concat([X_tr.reset_index(drop=True), st.iloc[:len(X_tr)].reset_index(drop=True)], axis=1)
                    X_te = pd.concat([X_te.reset_index(drop=True), st.iloc[len(X_tr):].reset_index(drop=True)], axis=1)
                    info["grid"]["channel_stats"] = st.shape[1]
            else:
                info["grid_rejected"] = {"kind": g["kind"], "adjacent_corr": round(adj, 2), "random_corr": round(rnd, 2)}
    if groups is not None:
        info["ranking_groups"] = int(len(np.unique(groups[:len(train)])))
    ids = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c]) and c not in dt_cols
           and 20 <= X_tr[c].nunique() <= 0.5 * len(X_tr)]
    if ids:
        info["id_columns"] = ids[:10]   # идентификаторы с повторами: кандидаты для id_factors и target encoding
    return Dataset(X_train=X_tr.reset_index(drop=True), X_test=X_te.reset_index(drop=True), y=y,
                   test_ids=test[task.id_col] if task.id_col and task.id_col in test.columns else None,
                   datetime_cols=dt_cols, text=text, grid=grid, groups=groups, info=info)


def _channel_names(columns: list[str], shape) -> list[str]:
    C, L = shape[0], shape[-1]
    out = []
    for i in range(C):
        name = str(columns[i * L])
        out.append(name.rstrip("0123456789").rstrip("_-. ") or f"ch{i}")
    return out
