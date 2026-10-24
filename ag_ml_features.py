"""Блоки чистки и генерации признаков и безопасное исполнение кода признаков от LLM.

Блоки работают на объединённом train+test без таргета, а свои статистики (квантили, корреляции,
средние по группам) считают только по строкам train, поэтому утечки нет.
Признаки, которым нужен таргет (target_encoding, entity_history), считаются внутри фолда
в agent/ml/supervised.py; их описания тоже лежат в FEATURE_BLOCK_DOCS, чтобы LLM их видела.
"""
from __future__ import annotations

import ast
import math
from typing import Callable

import numpy as np
import pandas as pd


def _cat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]


def _num_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def freq_encoding(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    for c in _cat_cols(df):
        df[f"{c}__freq"] = df[c].map(df[c].value_counts(dropna=False)).astype("float32")
    return df


def duplicate_count(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    """Сколько раз точно такая же строка признаков встречается в train+test (таргет не нужен).
    Повторяющиеся строки обычно означают типовое событие (шаблонный запрос, массовая атака, типовой объект)."""
    key = pd.Series(pd.util.hash_pandas_object(df, index=False).to_numpy())
    df["__dup_count"] = key.map(key.value_counts()).astype("float32").to_numpy()
    return df


def missing_count(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    df["__n_missing"] = df.isna().sum(axis=1).astype("float32")
    return df


def row_stats(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    nums = _num_cols(df)
    if len(nums) >= 3:
        block = df[nums].astype("float32")
        df["__row_mean"] = block.mean(axis=1)
        df["__row_std"] = block.std(axis=1)
        df["__row_max"] = block.max(axis=1)
        df["__row_min"] = block.min(axis=1)
    return df


def cat_pairs(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    cats = [c for c in _cat_cols(df) if 2 < df[c].nunique() <= 200][:4]
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a, b = cats[i], cats[j]
            df[f"{a}__x__{b}"] = df[a].astype(str) + "|" + df[b].astype(str)
    return df


def numeric_as_cat(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    for c in _num_cols(df):
        nun = df[c].nunique()
        if 2 < nun <= 30 and np.allclose(df[c].dropna() % 1, 0):
            df[f"{c}__cat"] = df[c].astype("Int64").astype(str)
    return df


_SENTINELS = (-1.0, -9.0, -99.0, -999.0, -9999.0, -99999.0, 99.0, 999.0, 9999.0, 99999.0, 999999.0)


def placeholders_as_missing(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    """Чистка: значения-заглушки -> пропуск. Решение по строкам train (n_fit), применяется ко всем.
    * 0, если ненулевые значения лежат далеко от нуля (год постройки 1900–2020, а встречается 0);
    * «круглые» заглушки (-1, -999, 9999, ...), если стоят на краю распределения отдельным пиком,
      далеко от остальных значений."""
    fit = df.iloc[:n_fit]
    for c in _num_cols(df):
        s = fit[c]
        nz = s[(s != 0) & s.notna()]
        zero = (s == 0).mean()
        if 0 < zero < 0.2 and len(nz) > 20 and nz.min() > 0 and nz.min() / max(nz.median(), 1e-9) > 0.5:
            df[c] = df[c].where(df[c] != 0)
            continue
        vals = s.dropna()
        if len(vals) < 50:
            continue
        for v in (vals.min(), vals.max()):
            share = (vals == v).mean()
            if v not in _SENTINELS or not 0.002 <= share < 0.5:
                continue
            rest = vals[vals != v]
            q1, q3 = rest.quantile([0.25, 0.75])
            if len(rest) > 20 and abs(v - rest.median()) > 5 * max(q3 - q1, 1e-9) \
                    and (v < rest.min() or v > rest.max()):
                df[c] = df[c].where(df[c] != v)
    return df


def clip_outliers(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    """Чистка выбросов: числовые признаки обрезаются по квантилям 0.5% / 99.5% строк train.
    Деревьям почти безразлично, линейной модели и нейросети помогает."""
    fit = df.iloc[:n_fit]
    for c in _num_cols(df):
        if fit[c].nunique() <= 20:
            continue
        lo, hi = fit[c].quantile([0.005, 0.995])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            df[c] = df[c].clip(lo, hi)
    return df


def log_skewed(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    """Нормализация формы: log1p для неотрицательных признаков с сильной правой асимметрией
    (площадь, деньги, счётчики). Порядок значений не меняется, деревьям это безразлично."""
    fit = df.iloc[:n_fit]
    for c in _num_cols(df):
        s = fit[c].dropna()
        if s.nunique() > 20 and s.min() >= 0 and s.skew() > 2:
            df[c] = np.log1p(df[c].clip(lower=0))
    return df


def drop_redundant(df: pd.DataFrame, n_fit: int, corr_threshold: float = 0.98) -> pd.DataFrame:
    """Удаление лишних признаков без таргета (по строкам train):
    * почти константы (одно значение у >= 99.5% строк);
    * дубликаты: числовые с одинаковыми значениями, категориальные с взаимно однозначным соответствием;
    * сильно коррелирующие числовые (|Спирмен| >= corr_threshold) — из пары остаётся признак с меньшим
      числом пропусков. Это убирает шум и ускоряет обучение, а информации почти не теряет."""
    fit = df.iloc[:n_fit]
    drop: list[str] = []
    for c in df.columns:
        top = fit[c].value_counts(dropna=False, normalize=True)
        if len(top) and top.iloc[0] >= 0.995:
            drop.append(c)
    cats = [c for c in _cat_cols(df) if c not in drop]
    codes = {c: pd.factorize(fit[c])[0] for c in cats}
    for i, a in enumerate(cats):
        if a in drop:
            continue
        for b in cats[i + 1:]:
            if b in drop:
                continue
            na, nb = codes[a].max() + 1, codes[b].max() + 1
            if na == nb and pd.Series(codes[a] * (nb + 1) + codes[b]).nunique() == na:
                drop.append(b)
    nums = [c for c in _num_cols(df) if c not in drop]
    if len(nums) >= 2:
        sample = fit[nums]
        if len(sample) > 50_000:
            sample = sample.sample(50_000, random_state=0)
        corr = sample.rank().corr().abs().to_numpy()
        nulls = sample.isna().mean().to_numpy()
        removed = set()
        for i in range(len(nums)):
            if i in removed:
                continue
            for j in range(i + 1, len(nums)):
                if j in removed or not corr[i, j] >= corr_threshold:
                    continue
                # оставляем признак с меньшим числом пропусков (при равенстве — первый)
                if nulls[j] < nulls[i]:
                    removed.add(i)
                    break
                removed.add(j)
        drop += [nums[i] for i in sorted(removed)]
    if drop and len(drop) < df.shape[1]:
        df = df.drop(columns=drop)
    return df


def drop_high_null(df: pd.DataFrame, n_fit: int) -> pd.DataFrame:
    keep = [c for c in df.columns if df[c].iloc[:n_fit].isna().mean() <= 0.95]
    return df[keep]


FEATURE_BLOCKS: dict[str, Callable[[pd.DataFrame, int], pd.DataFrame]] = {
    # порядок важен: сначала чистка, потом удаление лишнего, потом генерация
    "placeholders_as_missing": placeholders_as_missing,
    "zero_as_missing": placeholders_as_missing,   # старое имя блока
    "drop_high_null": drop_high_null,
    "clip_outliers": clip_outliers,
    "log_skewed": log_skewed,
    "drop_redundant": drop_redundant,
    "duplicate_count": duplicate_count,
    "freq_encoding": freq_encoding,
    "missing_count": missing_count,
    "row_stats": row_stats,
    "cat_pairs": cat_pairs,
    "numeric_as_cat": numeric_as_cat,
}

FEATURE_BLOCK_DOCS = {
    "placeholders_as_missing": "чистка: значения-заглушки (0 у года/площади, -1, -999, 9999 отдельным пиком на краю) считать пропуском",
    "drop_high_null": "удалить признаки с >95% пропусков",
    "clip_outliers": "обрезать выбросы числовых признаков по квантилям 0.5%/99.5% train (важно для linear и mlp)",
    "log_skewed": "log1p для сильно скошенных неотрицательных признаков (важно для linear и mlp)",
    "drop_redundant": "удалить почти константы, дубликаты и сильно коррелирующие числовые признаки (|corr| >= 0.98)",
    "freq_encoding": "частота каждого значения категориальных признаков",
    "duplicate_count": "сколько раз точно такая же строка признаков встречается в train+test",
    "missing_count": "число пропусков в строке",
    "row_stats": "mean/std/min/max по числовым признакам строки",
    "cat_pairs": "склейка пар категориальных признаков (до 4 признаков)",
    "numeric_as_cat": "низкокардинальные целые числа как категории",
    "target_encoding": "среднее таргета по значению каждой категории (считается внутри фолда, без утечки)",
    "row_memory": "память по дублям: распределение таргета у точно таких же строк train (внутри фолда, без утечки); "
                  "полезно, когда много полных дублей строк",
    "value_memory": "память по точным значениям: распределение таргета у строк с тем же точным значением часто "
                    "повторяющегося числового признака (размеры, счётчики), внутри фолда, без утечки",
    "text_oof": "прогноз линейной модели на TF-IDF текста как признак для деревьев (внутри фолда, без утечки); "
                "только если в задаче есть текстовые колонки",
    "id_factors": "скрытые факторы пар идентификаторов (пользователь × товар): матричная факторизация таргета, как "
                  "в рекомендациях; прогноз для новых пар по похожим (внутри фолда, без утечки)",
    "group_relative": "ранжирование: признаки относительно своей группы (отклонение от среднего по группе, ранг "
                      "внутри группы)",
    "entity_history": "история объекта: таргет того же объекта в прошлых периодах и изменение меняющихся признаков с прошлого наблюдения (только если найдены период и ключ объекта)",
}


# ---------- сгенерированные признаки (автопоиск, agent/ml/feature_search.py) ----------
# Описание признака — строка "op|a|b": ratio, prod, diff — пара числовых признаков;
# grp_mean, grp_dev — числовой признак a относительно среднего по категории b; grp_count — размер группы a.
# Статистики групп считаются по строкам train (n_fit), без таргета.

def generated_name(d: str) -> str:
    op, *cols = d.split("|")
    return "__g_" + op + "__" + "__".join(cols)


def _gen_values(df: pd.DataFrame, d: str, n_fit: int) -> pd.Series | None:
    op, *cols = d.split("|")
    if any(c not in df.columns for c in cols):
        return None
    if op in ("ratio", "prod", "diff"):
        a, b = (df[c].astype("float64") for c in cols)
        if op == "ratio":
            out = a / b.where(b != 0)
        elif op == "prod":
            out = a * b
        else:
            out = a - b
        return out.replace([np.inf, -np.inf], np.nan)
    if op in ("grp_mean", "grp_dev"):
        num, cat = cols
        key = df[cat].astype(str)
        mean = df[num].iloc[:n_fit].astype("float64").groupby(key.iloc[:n_fit]).mean()
        m = key.map(mean).astype("float64")
        return m if op == "grp_mean" else df[num].astype("float64") - m
    if op == "grp_count":
        key = df[cols[0]].astype(str)
        return key.map(key.iloc[:n_fit].value_counts()).astype("float64").fillna(0.0)
    return None


def add_generated(df: pd.DataFrame, defs: list[str], n_fit: int) -> pd.DataFrame:
    new = {}
    for d in defs:
        name = generated_name(d)
        if name in new or name in df.columns:   # тот же признак из двух проходов автопоиска
            continue
        v = _gen_values(df, d, n_fit)
        if v is not None:
            new[name] = v.astype("float32").to_numpy()
    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


# ---------- код признаков от LLM ----------

_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input",
    "breakpoint", "os", "sys", "subprocess", "shutil", "pathlib", "socket", "getattr", "setattr",
    "delattr", "exit", "quit", "help", "memoryview",
}


class UnsafeCode(ValueError):
    pass


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"SyntaxError: {e}") from e
    has_fn = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            if any(n not in ("numpy", "pandas", "math") for n in names):
                raise UnsafeCode(f"запрещённый import: {names}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeCode(f"запрещённое имя: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCode(f"запрещённый атрибут: {node.attr}")
        if isinstance(node, ast.FunctionDef) and node.name == "make_features":
            has_fn = True
    if not has_fn:
        raise UnsafeCode("нет функции make_features(df)")


def compile_feature_code(code: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    check_code(code)
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                     for k in ("abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "list", "dict",
                               "set", "tuple", "float", "int", "str", "bool", "round", "sorted", "any", "all",
                               "isinstance", "map", "filter", "ValueError", "KeyError", "Exception", "print")}

    def _import(name, *args, **kwargs):
        if name.split(".")[0] not in ("numpy", "pandas", "math"):
            raise ImportError(name)
        return __import__(name, *args, **kwargs)

    safe_builtins["__import__"] = _import
    ns: dict = {"__builtins__": safe_builtins, "np": np, "pd": pd, "math": math}
    exec(compile(code, "<make_features>", "exec"), ns)  # noqa: S102 — код прошёл check_code
    fn = ns["make_features"]

    def run(df: pd.DataFrame) -> pd.DataFrame:
        out = fn(df.copy())
        if not isinstance(out, pd.DataFrame) or len(out) != len(df):
            raise ValueError("make_features должна вернуть DataFrame той же длины")
        return out

    return run


def apply_features(df: pd.DataFrame, blocks: list[str], code: str | None, n_fit: int | None = None) -> pd.DataFrame:
    """Блоки применяются в порядке FEATURE_BLOCKS (чистка -> удаление лишнего -> генерация).
    n_fit — число первых строк df, относящихся к train: статистики блоков считаются только по ним."""
    n_fit = len(df) if n_fit is None else n_fit
    # The planner contract promises original inputs. Destructive blocks such as
    # drop_redundant must not remove the operands of an accepted hypothesis.
    if code:
        fn = compile_feature_code(code)
        fn(df.head(200))
        df = fn(df)
    done = set()
    for name, fn in FEATURE_BLOCKS.items():
        if name in blocks and fn not in done:
            df = fn(df, n_fit)
            done.add(fn)
    return df
