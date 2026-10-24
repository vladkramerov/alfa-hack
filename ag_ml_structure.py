"""Структура данных, которую агент находит сам, без знаний о конкретном датасете.

* Период: признак времени, по которому разделены train и test (весь test «позже» train).
  Тогда случайная CV врёт: модель видит будущее. Валидируемся вперёд по времени.
* Объект: набор признаков, который идентифицирует одну сущность (здание, клиента, устройство)
  в разные периоды. Тогда прошлые значения таргета этой сущности — сильнейший законный признак
  (они известны на момент прогноза: train целиком раньше test)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("agent.structure")


def _as_order(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s - pd.Timestamp("1970-01-01")).dt.days.astype(float)
    return None


def detect_period(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], dt_cols: list[str]) -> str | None:
    """Колонка, у которой >=95% значений test строго больше максимума train, а в train >= 3 разных значения.
    Из нескольких кандидатов — с наименьшим числом уникальных значений (код периода, а не точная дата)."""
    cands = []
    for c in cols:
        a = train[c] if c not in dt_cols else pd.to_datetime(train[c], errors="coerce")
        b = test[c] if c not in dt_cols else pd.to_datetime(test[c], errors="coerce")
        a, b = _as_order(a), _as_order(b)
        if a is None or a.notna().mean() < 0.95 or b.notna().mean() < 0.95 or a.nunique() < 3:
            continue
        if (b > a.max()).mean() >= 0.95:
            cands.append((a.nunique(), c))
    if not cands:
        return None
    return sorted(cands)[0][1]


def period_values(df: pd.DataFrame, col: str, is_dt: bool) -> np.ndarray:
    s = pd.to_datetime(df[col], errors="coerce") if is_dt else df[col]
    return _as_order(s).to_numpy()


def rare_class_mask(y: np.ndarray, is_classification: bool) -> np.ndarray | None:
    """Строки самого редкого класса — чтобы в каждом проверочном блоке их было достаточно."""
    if not is_classification or y.ndim != 1 or len(y) == 0:
        return None
    vals, cnt = np.unique(y, return_counts=True)
    return y == vals[np.argmin(cnt)]


def time_folds(period: np.ndarray, n_folds: int, min_share: float = 0.05,
               min_train_share: float = 0.35, test_share: float | None = None,
               rare: np.ndarray | None = None, min_rare: int = 50) -> list[tuple[np.ndarray, np.ndarray]]:
    """Валидация вперёд: для каждого из последних периодов обучаемся на всём, что раньше.
    Мелкие хвостовые периоды объединяются, чтобы на валидации было >= min_share строк.
    Период может быть и непрерывным (секунды, точная дата): блоки собираются по накопленному числу строк.

    test_share — передаётся, когда признаки зависят от горизонта (история объекта): тогда проверочный блок
    как горизонт теста (но не больше 15% train) — прогноз на 4 недели вперёд проверяем на блоках по 4 недели,
    иначе история в проверке свежее, чем будет в тесте. Без истории блоки мельче: больше фолдов и обучение
    ближе к полному train. rare — строки редкого класса: в блоке их должно быть хотя бы min_rare (или 15%
    всех), иначе метрика блока — шум (card_fraud: 17 мошенничеств на блок 5%)."""
    ok = ~np.isnan(period)
    uniq, inv, counts = np.unique(period[ok], return_inverse=True, return_counts=True)
    n = len(period)
    if test_share is not None:
        min_share = max(min_share, min(0.15, test_share))
    rare_cnt = np.bincount(inv, weights=rare[ok].astype(float), minlength=len(uniq)) if rare is not None \
        else np.zeros(len(uniq))
    need_rare = min(min_rare, 0.15 * rare_cnt.sum())
    blocks: list[tuple[float, float]] = []   # [начало, конец] блока по значению периода
    acc, acc_rare, hi = 0, 0.0, None
    for v, c, r in zip(uniq[::-1], counts[::-1], rare_cnt[::-1]):          # с конца: последние периоды
        hi = v if hi is None else hi
        acc += c
        acc_rare += r
        if acc >= min_share * n and acc_rare >= need_rare:
            blocks.append((v, hi))
            acc, acc_rare, hi = 0, 0.0, None
        if len(blocks) >= n_folds:
            break
    folds = []
    for lo, hi in blocks[::-1]:
        va = np.flatnonzero(ok & (period >= lo) & (period <= hi))
        tr = np.flatnonzero(ok & (period < lo))
        if len(tr) >= 0.2 * n:    # слишком мало истории — фолд бесполезен
            folds.append((tr, va))
    folds = folds[-min(n_folds, 3):]
    # финальная модель учится на всём train: фолды, где истории меньше min_train_share, плохо похожи на тест
    # (на задаче 1 фолд с двумя годами истории давал RMSE 45 против 30 у фолда с четырьмя) — отбрасываем,
    # если после этого остаётся хотя бы два фолда
    good = [f for f in folds if len(f[0]) >= min_train_share * n]
    return good if len(good) >= 2 else folds


def detect_group_time(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], dt_cols: list[str],
                      group: str | None = None) -> dict | None:
    """Тест — последние наблюдения каждой группы, а не общий хвост по времени («последние 20% оценок
    каждого пользователя», «следующие покупки каждого клиента»). Признак: есть колонка времени, у которой
    почти у всех строк test значение не меньше максимума train той же группы, хотя по всему train test
    позже не целиком. Группа — колонка группы из условия или повторяющийся идентификатор, который есть
    и в test. Возвращает {"group", "time", "ratio"}: ratio — сколько строк test приходится на строку
    train в типичной группе (размер проверочного хвоста)."""
    times = {}
    for c in cols:
        if c not in train.columns or c not in test.columns:
            continue
        a = _as_order(pd.to_datetime(train[c], errors="coerce") if c in dt_cols else train[c])
        b = _as_order(pd.to_datetime(test[c], errors="coerce") if c in dt_cols else test[c])
        if a is None or b is None or a.notna().mean() < 0.95 or b.notna().mean() < 0.95 or a.nunique() < 10:
            continue
        if (b > a.max()).mean() >= 0.95:     # это обычный период, его ловит detect_period
            continue
        times[c] = (a.to_numpy(), b.to_numpy())
    if not times:
        return None
    n = len(train)
    groups = [group] if group and group in train.columns and group in test.columns else []
    for c in cols:
        if c in groups or c in times or c not in train.columns or c not in test.columns:
            continue
        k = train[c].nunique()
        if 20 <= k <= 0.5 * n:
            groups.append(c)
    for g in groups:
        codes, _ = pd.factorize(pd.concat([train[g], test[g]], ignore_index=True))
        g_tr, g_te = codes[:n], codes[n:]
        seen = np.isin(g_te, g_tr) & (g_te >= 0)
        if seen.mean() < 0.5 or n / max(1, len(np.unique(g_tr))) < 3:
            continue
        for c, (a, b) in times.items():
            if c == g or pd.Series(a).groupby(g_tr).nunique().gt(1).mean() < 0.5:   # время меняется внутри группы
                continue
            mx = pd.Series(a).groupby(g_tr).max()
            later = b[seen] >= mx.reindex(g_te[seen]).to_numpy()
            if later.mean() < 0.95:
                continue
            n_tr = pd.Series(g_tr).value_counts()
            n_te = pd.Series(g_te[seen]).value_counts()
            ratio = float(np.median((n_te / n_tr.reindex(n_te.index)).to_numpy()))
            return {"group": g, "time": c, "ratio": float(np.clip(ratio, 0.05, 0.5)),
                    "share_later": round(float(later.mean()), 3)}
    return None


def group_time_folds(time_tr: np.ndarray, group_tr: np.ndarray, ratio: float, n_folds: int,
                     min_train_share: float = 0.35) -> list[tuple[np.ndarray, np.ndarray]]:
    """Фолды как тест: в каждой группе проверка — хвост её наблюдений по времени, обучение — всё, что
    у этой группы раньше (у других групп — тоже только их прошлое). Фолд j берёт j-й с конца хвост
    размера ratio от строк группы."""
    n = len(time_tr)
    order = np.lexsort((-np.nan_to_num(time_tr, nan=-np.inf), group_tr))   # внутри группы — с конца
    g_sorted = group_tr[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n) - np.repeat(starts, sizes)           # 0 — самое позднее наблюдение группы
    pos = np.empty(n)
    pos[order] = rank / np.repeat(sizes, sizes)
    folds = []
    for j in range(1, n_folds + 1):
        va = np.flatnonzero((pos >= (j - 1) * ratio) & (pos < j * ratio))
        tr = np.flatnonzero(pos >= j * ratio)
        if len(tr) < min_train_share * n or len(va) == 0:
            break
        folds.append((tr, va))
    return folds[::-1]


def detect_entity(train: pd.DataFrame, test: pd.DataFrame, period: str | None, cols: list[str],
                  max_cols: int = 6) -> list[str] | None:
    """Жадно собираем ключ объекта: внутри одного периода ключ почти уникален, а между соседними
    периодами повторяется. Признаки, меняющиеся год к году (погода, рейтинги), повторяемость рушат
    и в ключ не попадают."""
    if period is None:
        return None
    cols = [c for c in cols if c != period]
    df = pd.concat([train[cols + [period]], test[cols + [period]]], ignore_index=True)
    raw = df[period]
    if not pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_datetime64_any_dtype(raw):
        raw = pd.to_datetime(raw, errors="coerce")
    per = _as_order(raw).to_numpy(dtype=float)
    ok = ~np.isnan(per)
    pers = np.unique(per[ok])
    if len(pers) < 2:
        return None
    # номер периода по порядку; строки без периода — в отдельный «период» -1, в повторяемость не идут
    pidx = np.where(ok, np.searchsorted(pers, np.where(ok, per, pers[0])), -1).astype(np.int64)
    codes = {c: pd.factorize(df[c])[0].astype(np.int64) for c in cols if df[c].nunique() > 1}
    cand = list(codes)
    P = len(pers) + 1

    def combine(key: list[str]) -> np.ndarray:
        k = codes[key[0]]
        for c in key[1:]:
            k = pd.factorize(k * (codes[c].max() + 2) + codes[c] + 1)[0].astype(np.int64)
        return k

    def stats(key: list[str]) -> tuple[float, float]:
        """Уникальность ключа внутри периода и средняя по парам соседних периодов доля строк периода, чей ключ
        был в предыдущем периоде. Векторно: при непрерывном времени периодов десятки тысяч."""
        k = combine(key)
        both = k * P + pidx + 1
        uniq = 1.0 - float(pd.Series(both).duplicated().mean())
        sel = ok & (pidx > 0)
        prev_present = np.isin(k[sel] * P + pidx[sel], both)      # (ключ, период - 1) встречался
        rec = pd.Series(prev_present).groupby(pidx[sel]).mean()
        return uniq, float(rec.mean()) if len(rec) else 0.0

    # стартуем с самого «уникального» признака, который при этом повторяется между периодами
    scored = []
    for c in cand:
        if df[c].nunique() < 0.01 * len(df):
            continue
        u, r = stats([c])
        if r >= 0.2:
            scored.append((u, r, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    key = [scored[0][2]]
    u, r = scored[0][0], scored[0][1]
    for _ in range(max_cols - 1):
        if u >= 0.99:
            break
        best = None
        for c in cand:
            if c in key:
                continue
            u2, r2 = stats(key + [c])
            # берём признак, который добавляет уникальности и почти не рушит повторяемость
            if u2 > u + 0.005 and r2 >= 0.85 * r and (best is None or u2 > best[0]):
                best = (u2, r2, c)
        if best is None:
            break
        u, r = best[0], best[1]
        key.append(best[2])
    if u < 0.9 or r < 0.2:
        return None
    log.info("ключ объекта: %s (уникальность в периоде %.3f, повторяемость между периодами %.2f)", key, u, r)
    return key
