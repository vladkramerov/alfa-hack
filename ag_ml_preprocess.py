"""Предобработка, не зависящая от конкретного датасета: типы колонок, даты, текст, «сетки» пикселей
и сигналов, высококардинальные категории."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

_DATE_HINT = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
HIGH_CARD = 256          # категория с большим числом значений идёт в деревья кодом по частоте, а не как категория


def detect_datetime_cols(df: pd.DataFrame, types: dict | None = None) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        sample = s.dropna().astype(str).head(300)
        hinted = (types or {}).get(c) == "datetime"
        if len(sample) < 20 or (not hinted and sample.str.match(_DATE_HINT).mean() < 0.9):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.9:
            out.append(c)
    return out


def detect_text_cols(df: pd.DataFrame, types: dict | None = None, exclude: set | None = None) -> list[str]:
    """Свободный текст: строковая колонка, где значения почти уникальны и состоят из нескольких слов
    (отзыв, описание, название), или колонка, помеченная как текст в data_dictionary."""
    out = []
    for c in df.columns:
        if exclude and c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        v = s.dropna().astype(str)
        if len(v) < 20:
            continue
        hinted = (types or {}).get(c) == "text"
        smp = v.sample(min(len(v), 3000), random_state=0)
        words = smp.str.count(r"\S+")
        uniq = v.nunique() / len(v)
        if hinted or (uniq >= 0.3 and words.mean() >= 2.0 and smp.str.len().mean() >= 12) or \
                (words.mean() >= 5 and uniq >= 0.05):
            out.append(c)
    return out


def text_stats(s: pd.Series, prefix: str) -> pd.DataFrame:
    """Дешёвые признаки текста для деревьев: длина, слова, доля заглавных, цифры, знаки."""
    t = s.fillna("").astype(str)
    n = t.str.len().astype("float32")
    words = t.str.count(r"\S+").astype("float32")
    return pd.DataFrame({
        f"{prefix}__len": n,
        f"{prefix}__words": words,
        f"{prefix}__upper": (t.str.count(r"[A-ZА-ЯЁ]") / n.clip(lower=1)).astype("float32"),
        f"{prefix}__digits": (t.str.count(r"\d") / n.clip(lower=1)).astype("float32"),
        f"{prefix}__excl": t.str.count(r"[!?]").astype("float32"),
        f"{prefix}__wordlen": (n / words.clip(lower=1)).astype("float32"),
        f"{prefix}__empty": (s.isna() | (t.str.strip() == "")).astype("float32"),
    }, index=s.index)


# ---------------------------------------------------------------- сетки: изображения и сигналы
_GRID = re.compile(r"^(.*?)[_\-.\s]?(\d+)(?:[_\-.x](\d+))?$")


def detect_grid(columns: list[str], df: pd.DataFrame | None = None, min_cols: int = 64) -> dict | None:
    """Колонки-«сетка»: пиксели изображения (pixel_r_c, pixel0..pixel783 при квадратном числе) или
    несколько сигналов одинаковой длины (acc_x_0..acc_x_127, acc_y_0..). Возвращает
    {"kind": "image"|"sequence", "columns": [...в порядке (канал, позиция)], "shape": (C, H, W) | (C, L)}.
    Решение по именам и числу колонок, а не по конкретному датасету."""
    groups: dict[str, dict] = {}
    for c in columns:
        m = _GRID.match(str(c))
        if not m:
            continue
        pre, a, b = m.group(1), int(m.group(2)), m.group(3)
        g = groups.setdefault(pre, {})
        g[(a, int(b)) if b is not None else (a,)] = c
    groups = {p: g for p, g in groups.items() if len(g) >= 16}
    if not groups or sum(len(g) for g in groups.values()) < min_cols:
        return None
    pixel_like = True
    if df is not None:
        cols = [c for g in groups.values() for c in g.values()]
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in cols[:2000]):
            return None
        v = df[cols[:: max(1, len(cols) // 200)]].head(500).to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        pixel_like = len(v) > 0 and v.min() >= 0 and (v.max() <= 1.0 or (v.max() <= 255 and np.allclose(v % 1, 0)))
    pixel_like = pixel_like or any(re.search(r"pix|px|img|image", p, re.I) for p in groups)
    # одна группа с двумя индексами — двумерная картинка
    if len(groups) == 1:
        (pre, g), = groups.items()
        keys = list(g)
        if len(keys[0]) == 2:
            rows = sorted({k[0] for k in keys})
            cols_ = sorted({k[1] for k in keys})
            if len(rows) * len(cols_) == len(keys) and len(rows) >= 8 and len(cols_) >= 8 and pixel_like:
                order = [g[(r, c)] for r in rows for c in cols_]
                return {"kind": "image", "columns": order, "shape": (1, len(rows), len(cols_))}
            return None
        idx = sorted(k[0] for k in keys)
        n = len(idx)
        if idx != list(range(idx[0], idx[0] + n)):
            return None
        order = [g[(i,)] for i in idx]
        for ch in (1, 3):
            side = int(round((n / ch) ** 0.5))
            if side >= 8 and side * side * ch == n and pixel_like:
                return {"kind": "image", "columns": order, "shape": (ch, side, side)}
        if n >= 32:
            return {"kind": "sequence", "columns": order, "shape": (1, n)}
        return None
    # несколько групп одинаковой длины с одним индексом — многоканальный сигнал
    lens = {len(g) for g in groups.values()}
    if len(lens) != 1 or any(len(next(iter(g))) != 1 for g in groups.values()):
        return None
    L = lens.pop()
    if L < 16:
        return None
    order = []
    for pre in sorted(groups):
        g = groups[pre]
        idx = sorted(k[0] for k in g)
        if idx != list(range(idx[0], idx[0] + L)):
            return None
        order += [g[(i,)] for i in idx]
    return {"kind": "sequence", "columns": order, "shape": (len(groups), L)}


# ---------------------------------------------------------------- даты
def expand_datetimes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    new = {}
    for c in cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = pd.to_datetime(df[c], errors="coerce")
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_convert(None)
        new[f"{c}__year"] = d.dt.year.astype("float32")
        new[f"{c}__month"] = d.dt.month.astype("float32")
        new[f"{c}__day"] = d.dt.day.astype("float32")
        new[f"{c}__dow"] = d.dt.dayofweek.astype("float32")
        new[f"{c}__doy"] = d.dt.dayofyear.astype("float32")
        new[f"{c}__week"] = d.dt.isocalendar().week.astype("float32").to_numpy()
        new[f"{c}__ts"] = ((d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0).astype("float32")
        secs = d.dt.hour * 3600 + d.dt.minute * 60 + d.dt.second
        if secs.fillna(0).nunique() > 1:      # есть время суток
            new[f"{c}__hour"] = d.dt.hour.astype("float32")
            new[f"{c}__minute_of_day"] = (secs / 60.0).astype("float32")
            new[f"{c}__hour_of_week"] = (d.dt.dayofweek * 24 + d.dt.hour).astype("float32")
        new[f"{c}__weekend"] = (d.dt.dayofweek >= 5).astype("float32")
        new[f"{c}__month_end"] = d.dt.is_month_end.astype("float32")
    if cols:
        df = pd.concat([df.drop(columns=cols), pd.DataFrame(new, index=df.index)], axis=1)
    return df


def finalize_types(df: pd.DataFrame, n_fit: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Числа -> float32, остальное -> category. Категории с > HIGH_CARD значениями упорядочиваются по
    частоте в train: их код = ранг частоты, и деревья могут использовать его как число."""
    cat_cols = []
    n_fit = len(df) if n_fit is None else n_fit
    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            df[c] = s.astype("float32")
        else:
            st = s.astype(str).where(s.notna(), None)
            nun = st.nunique()
            if nun > HIGH_CARD:
                order = st.iloc[:n_fit].value_counts().index.tolist()
                rest = [v for v in pd.unique(st.dropna()) if v not in set(order)]
                df[c] = pd.Categorical(st, categories=order + rest)
            else:
                df[c] = st.astype("category")
            cat_cols.append(c)
    # константы и полностью пустые признаки ничего не дают
    nun = df.nunique(dropna=False)
    drop = [c for c in df.columns if nun[c] <= 1]
    if drop and len(drop) < df.shape[1]:
        df = df.drop(columns=drop)
        cat_cols = [c for c in cat_cols if c not in drop]
    df.columns = [_safe_name(c) for c in df.columns]
    cat_cols = [_safe_name(c) for c in cat_cols]
    return df, cat_cols


def _safe_name(c: str) -> str:
    # LightGBM не принимает спецсимволы JSON в именах признаков
    return re.sub(r'[\[\]{}":,]', "_", str(c))


def native_cats(df: pd.DataFrame, cat_cols: list[str], limit: int = HIGH_CARD) -> list[str]:
    """Категории, которые модель берёт как категории; остальные (много значений) — кодом по частоте."""
    return [c for c in cat_cols if c in df.columns and len(df[c].cat.categories) <= limit]


def codes_frame(X: pd.DataFrame, cat_cols: list[str], keep: list[str]) -> pd.DataFrame:
    """Высококардинальные категории (не из keep) -> числовой код (ранг частоты), пропуск -> NaN."""
    conv = [c for c in cat_cols if c in X.columns and c not in keep]
    if not conv:
        return X
    X = X.copy()
    for c in conv:
        codes = X[c].cat.codes.astype("float32")
        X[c] = codes.where(codes >= 0)
    return X


def to_numeric_matrix(df: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    """Для моделей без нативных категорий: коды категорий, NaN -> -1 у категорий."""
    out = df.copy()
    for c in cat_cols:
        if c in out.columns and isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].cat.codes.astype("float32")
    return out.to_numpy(dtype="float32")
