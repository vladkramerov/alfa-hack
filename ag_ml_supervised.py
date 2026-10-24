"""Признаки, которым нужен таргет. Считаются ВНУТРИ фолда: только по строкам обучающей части,
поэтому валидация остаётся честной (никакого подглядывания в таргет валидации и теста).

* target_encoding — сглаженное среднее таргета по значению категории; для самих обучающих строк
  считается по вложенным фолдам (иначе модель видит свой же таргет).
* row_memory — «память по дублям»: распределение таргета у точно таких же строк train (сглаженное
  к общему, smoothing=1). Когда в данных много полных дублей, а у части из них разные таргеты, это
  прямо даёт модели долю каждого класса в группе одинаковых строк; для обучающих строк — по вложенным
  фолдам, как target_encoding.
* value_memory — «память по точным значениям»: то же сглаженное распределение таргета, но по точному
  значению отдельных числовых признаков, которые часто повторяются (размеры пакетов, счётчики, площади
  типовых объектов). Дерево может выделить такое значение только серией разбиений, а здесь оно
  получает его «репутацию» сразу. Задача 2: +0.02 macro F1 поверх row_memory.
* entity_history — история объекта: таргет того же объекта в прошлых периодах (последнее значение,
  среднее, число наблюдений, сколько периодов прошло). Нужны найденные период и ключ объекта
  (agent/ml/structure.py).
* text_oof — прогноз линейной модели на TF-IDF текста как признак для деревьев (стекинг): для обучающих
  строк — по вложенным фолдам, для остальных — модель на всех обучающих строках. Так бустинг соединяет
  текст с таблицей, не глядя на десятки тысяч разреженных колонок.
* id_factors — скрытые факторы пар идентификаторов (пользователь × товар, магазин × товар): матричная
  факторизация таргета по парам, как в рекомендательных системах. Даёт прогноз для пары, которой не было
  в train, по похожим пользователям и товарам; для обучающих строк — по вложенным фолдам."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SUPERVISED_BLOCKS = ("target_encoding", "row_memory", "value_memory", "entity_history", "text_oof", "id_factors")


def row_codes(df: pd.DataFrame, discrete_share: float = 0.02) -> dict[str, np.ndarray]:
    """Коды групп одинаковых строк: по всем признакам и по одним дискретным (категории, счётчики,
    флаги — до discrete_share*строк уникальных значений). Второй ключ объединяет строки, которые
    отличаются только «шумными» непрерывными измерениями (длительности, скорости, случайные номера)."""
    def code(cols):
        return pd.factorize(pd.util.hash_pandas_object(df[cols], index=False).to_numpy())[0]
    out = {"__row": code(list(df.columns))}
    disc = [c for c in df.columns if df[c].nunique(dropna=False) <= discrete_share * len(df)]
    if 0 < len(disc) < df.shape[1]:
        out["__row_d"] = code(disc)
    return out


def value_codes(df: pd.DataFrame, max_cols: int = 16, min_unique: int = 20, max_unique_share: float = 0.1,
                min_repeat_share: float = 0.5) -> dict[str, np.ndarray]:
    """Коды точных значений числовых колонок, где значения повторяются: уникальных значений от min_unique
    до max_unique_share строк, и хотя бы min_repeat_share строк приходится на значения, встречающиеся
    5+ раз. Колонки с самой высокой повторяемостью, не больше max_cols."""
    cand = []
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        nu = s.nunique()
        if nu < min_unique or nu > max_unique_share * len(s):
            continue
        vc = s.map(s.value_counts())
        rep = float((vc >= 5).mean())
        if rep >= min_repeat_share:
            cand.append((rep, c))
    cand = sorted(cand, key=lambda t: -t[0])[:max_cols]
    return {f"__val_{c}": pd.factorize(df[c])[0] for _, c in cand}


def _te_stats(codes: np.ndarray, y: np.ndarray, smoothing: float) -> tuple[dict, float]:
    prior = float(np.mean(y))
    df = pd.DataFrame({"c": codes, "y": y})
    g = df.groupby("c")["y"].agg(["sum", "count"])
    enc = (g["sum"] + prior * smoothing) / (g["count"] + smoothing)
    return enc.to_dict(), prior


def target_encoding(cat_frames: dict[str, np.ndarray], y_tr: np.ndarray, tr_idx: np.ndarray,
                    other_idx: list[np.ndarray], seed: int, smoothing: float = 20.0) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """cat_frames: имя -> коды категорий по всем строкам (train+test). y_tr — таргет строк tr_idx
    (для классификации — матрица one-hot, кодируется каждый класс, кроме первого)."""
    Y = y_tr.reshape(len(y_tr), -1)
    out_tr = {}
    out_other = [dict() for _ in other_idx]
    kf = KFold(5, shuffle=True, random_state=seed)
    for name, codes in cat_frames.items():
        c_tr = codes[tr_idx]
        for j in range(Y.shape[1]):
            col = f"{name}__te" + (f"{j}" if Y.shape[1] > 1 else "")
            vals = np.empty(len(tr_idx))
            for a, b in kf.split(c_tr):
                enc, prior = _te_stats(c_tr[a], Y[a, j], smoothing)
                vals[b] = pd.Series(c_tr[b]).map(enc).fillna(prior).to_numpy()
            out_tr[col] = vals
            enc, prior = _te_stats(c_tr, Y[:, j], smoothing)
            for k, idx in enumerate(other_idx):
                out_other[k][col] = pd.Series(codes[idx]).map(enc).fillna(prior).to_numpy()
    return pd.DataFrame(out_tr), [pd.DataFrame(o) for o in out_other]


def entity_history(entity: np.ndarray, period: np.ndarray, y_known: np.ndarray, known_idx: np.ndarray,
                   query_idx: np.ndarray) -> pd.DataFrame:
    """Для каждой строки query: статистика таргета того же объекта по строкам known строго
    из более ранних периодов. entity/period — по всем строкам (train+test)."""
    yk = y_known.reshape(len(y_known), -1)[:, 0] if y_known.ndim > 1 else y_known
    known = pd.DataFrame({"e": entity[known_idx], "p": period[known_idx], "y": yk})
    per_p = known.groupby(["e", "p"], sort=True)["y"].agg(["mean", "count"]).reset_index()
    per_p = per_p.sort_values(["e", "p"])
    per_p["csum"] = (per_p["mean"] * per_p["count"]).groupby(per_p["e"]).cumsum()
    per_p["ccnt"] = per_p["count"].groupby(per_p["e"]).cumsum()
    per_p["hist_mean"] = per_p["csum"] / per_p["ccnt"]
    # без lambda по группам: при десятках тысяч объектов она в сотни раз медленнее сдвигов и сумм
    g = per_p.groupby("e")["mean"]
    last = np.column_stack([per_p["mean"], g.shift(1), g.shift(2)])
    per_p["last3"] = np.nanmean(last, axis=1)                                    # уровень последних периодов
    k = g.cumcount().to_numpy() + 1.0
    s1 = g.cumsum().to_numpy()
    s2 = (per_p["mean"] ** 2).groupby(per_p["e"]).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(k > 1, (s2 - s1 ** 2 / k) / (k - 1), np.nan)
    per_p["hist_std"] = np.sqrt(np.clip(var, 0, None))                           # насколько ряд шумный
    per_p["prev_p"] = per_p["p"]
    q = pd.DataFrame({"e": entity[query_idx], "p": period[query_idx], "_row": np.arange(len(query_idx))})
    q = q.sort_values("p")
    right = per_p[["e", "p", "mean", "hist_mean", "ccnt", "prev_p", "last3", "hist_std"]].sort_values("p")
    m = pd.merge_asof(q, right, on="p", by="e", allow_exact_matches=False, direction="backward")
    m = m.sort_values("_row")
    return pd.DataFrame({
        "__hist_last": m["mean"].to_numpy(),
        "__hist_mean": m["hist_mean"].to_numpy(),
        "__hist_n": m["ccnt"].fillna(0).to_numpy(),
        "__hist_gap": (m["p"] - m["prev_p"]).to_numpy(),
        "__hist_last3": m["last3"].to_numpy(),
        "__hist_std": m["hist_std"].to_numpy(),
    })


def entity_last_index(entity: np.ndarray, period: np.ndarray, known_idx: np.ndarray,
                      query_idx: np.ndarray) -> np.ndarray:
    """Для каждой строки query — номер строки (среди всех строк train+test) последнего наблюдения того же
    объекта строго в более раннем периоде среди known; NaN, если такого нет."""
    known = pd.DataFrame({"e": entity[known_idx], "p": period[known_idx], "r": known_idx.astype(float)})
    known = known.sort_values(["p", "r"])
    q = pd.DataFrame({"e": entity[query_idx], "p": period[query_idx], "_row": np.arange(len(query_idx))})
    m = pd.merge_asof(q.sort_values("p"), known, on="p", by="e", allow_exact_matches=False, direction="backward")
    return m.sort_values("_row")["r"].to_numpy()


def varying_columns(X: pd.DataFrame, entity: np.ndarray, max_cols: int = 30) -> list[str]:
    """Числовые признаки, которые у одного объекта меняются между периодами (погода, рейтинги):
    их изменение с прошлого наблюдения объясняет изменение таргета."""
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not str(c).startswith("__")]
    if not num:
        return []
    df = X[num].copy()
    df["__e"] = entity
    g = df.groupby("__e")
    multi = g.size() > 1
    if not multi.any():
        return []
    share = (g.nunique()[multi] > 1).mean()
    share = share[share > 0.05].sort_values(ascending=False)
    return list(share.index[:max_cols])


def _text_model(kind: str, n: int, seed: int):
    from sklearn.linear_model import Ridge, SGDClassifier
    if kind == "reg":
        return Ridge(alpha=3.0, solver="sparse_cg" if n > 5000 else "auto")
    # логистическая регрессия стохастическим градиентом: на 40 000 текстах в 15 раз быстрее LBFGS при том же
    # качестве (ru_reviews: macro F1 0.7366 против 0.7378)
    return SGDClassifier(loss="log_loss", alpha=max(2e-6, 0.8 / n), max_iter=30, tol=1e-4, random_state=seed)


# Кэш прогнозов по тексту: они зависят только от строк обучения, таргета и строк запроса, а не от эксперимента.
# Оркестратор заполняет его для фиксированных фолдов до запуска экспериментов (fork наследует кэш).
_TEXT_CACHE: dict = {}


def _key(*arrays) -> int:
    return hash(tuple(np.ascontiguousarray(a).tobytes() for a in arrays))


def text_oof(T, y_known: np.ndarray, known_idx: np.ndarray, query_idx: list[np.ndarray], kind: str, seed: int,
             n_inner: int = 3) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """kind: reg — y число; cls — y коды классов (вероятности всех классов, кроме первого; для бинарной — одна
    колонка); ml — y матрица меток (по колонке на метку). T — TF-IDF по всем строкам train+test."""
    base = (_key(known_idx, np.asarray(y_known)), kind, seed, id(T))
    tr = _TEXT_CACHE.get(base)
    qs = [_TEXT_CACHE.get(base + (_key(q),)) for q in query_idx]
    if tr is not None and all(q is not None for q in qs):
        return tr, qs
    tr, qs = _text_oof(T, y_known, known_idx, query_idx, kind, seed, n_inner)
    _TEXT_CACHE[base] = tr
    for q, out in zip(query_idx, qs):
        _TEXT_CACHE[base + (_key(q),)] = out
    return tr, qs


def _text_oof(T, y_known, known_idx, query_idx, kind, seed, n_inner):
    import warnings
    Y = y_known if kind == "ml" else y_known.reshape(-1, 1)
    targets = [Y[:, j] for j in range(Y.shape[1])]
    k = int(Y.max()) + 1 if kind == "cls" else 2

    def fit_predict(rows_fit, y_fit, rows_pred):
        if kind == "reg":
            m = _text_model(kind, len(rows_fit), seed)
            mu = float(np.mean(y_fit))
            m.fit(T[rows_fit], y_fit - mu)
            return (m.predict(T[rows_pred]) + mu).reshape(-1, 1)
        y_fit = y_fit.astype(int)
        cls = np.unique(y_fit)
        if len(cls) < 2:
            out = np.zeros((len(rows_pred), k))
            out[:, cls[0]] = 1.0
        else:
            m = _text_model(kind, len(rows_fit), seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(T[rows_fit], y_fit)
            out = np.zeros((len(rows_pred), k))
            out[:, m.classes_.astype(int)] = m.predict_proba(T[rows_pred])
        return out[:, 1:]   # первый класс лишний: вероятности в сумме дают 1

    names, tr_parts, q_parts = [], [], [[] for _ in query_idx]
    kf = KFold(n_inner, shuffle=True, random_state=seed)
    for j, yt in enumerate(targets):
        vals = None
        for a, b in kf.split(known_idx):
            p = fit_predict(known_idx[a], yt[a], known_idx[b])
            if vals is None:
                vals = np.zeros((len(known_idx), p.shape[1]))
            vals[b] = p
        tr_parts.append(vals)
        for i, q in enumerate(query_idx):
            q_parts[i].append(fit_predict(known_idx, yt, q))
        width = vals.shape[1]
        names += [f"__text_p{j}_{c}" if kind == "ml" else f"__text_p{c}" for c in range(width)]
    tr = pd.DataFrame(np.hstack(tr_parts), columns=names)
    return tr, [pd.DataFrame(np.hstack(p), columns=names) for p in q_parts]


def id_pairs(codes: dict[str, np.ndarray], n_train: int, max_pairs: int = 3) -> list[tuple[str, str]]:
    """Пары высококардинальных идентификаторов, у которых сочетания почти не повторяются (матрица
    «пользователь × товар» разрежена): для них target encoding пары бесполезен, а факторизация — нет."""
    cand = []
    for name, c in codes.items():
        tr = c[:n_train]
        nu = len(np.unique(tr[tr >= 0]))
        if 20 <= nu <= 0.5 * n_train:
            cand.append((nu, name))
    cand = [nm for _, nm in sorted(cand, reverse=True)[:4]]
    out = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = codes[cand[i]][:n_train], codes[cand[j]][:n_train]
            if pd.Series(a.astype(np.int64) * (b.max() + 2) + b).nunique() > 0.5 * n_train:
                out.append((cand[i], cand[j]))
    return out[:max_pairs]


def _als(a: np.ndarray, b: np.ndarray, r: np.ndarray, na: int, nb: int, k: int = 8, lam: float = 5.0,
         iters: int = 8, seed: int = 0):
    """Смещённая матричная факторизация r ≈ ba[a] + bb[b] + P[a]·Q[b] чередующимися наименьшими квадратами
    (все строки одной стороны решаются разом, батчем систем (k+1)x(k+1))."""
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(scale=0.1, size=(na, k)), rng.normal(scale=0.1, size=(nb, k))
    ba, bb = np.zeros(na), np.zeros(nb)
    reg = lam * np.eye(k + 1)
    reg[-1, -1] = lam * 0.2   # смещения регуляризуем слабее факторов

    def grouping(idx):
        order = np.argsort(idx, kind="stable")
        sidx = idx[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sidx)) + 1]
        return order, starts, sidx[starts]

    ga, gb = grouping(a), grouping(b)

    def solve(grp, n_self, F, b_other, t):
        order, starts, uniq = grp
        X = np.hstack([F, np.ones((len(t), 1))])[order]
        G = np.zeros((n_self, k + 1, k + 1))
        G[uniq] = np.add.reduceat(X[:, :, None] * X[:, None, :], starts, axis=0)
        h = np.zeros((n_self, k + 1))
        h[uniq] = np.add.reduceat(X * (t - b_other)[order][:, None], starts, axis=0)
        sol = np.linalg.solve(G + reg, h[..., None])[..., 0]
        return sol[:, :k], sol[:, k]

    for _ in range(iters):
        P, ba = solve(ga, na, Q[b], bb[b], r)
        Q, bb = solve(gb, nb, P[a], ba[a], r)
    return P, Q, ba, bb


def id_factors(codes: dict[str, np.ndarray], pairs: list[tuple[str, str]], y_known: np.ndarray,
               known_idx: np.ndarray, query_idx: list[np.ndarray], seed: int,
               n_inner: int = 5) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Прогноз факторизации для каждой пары (codes — коды по всем строкам train+test; -1 — пропуск)."""
    y = np.asarray(y_known, dtype=float).reshape(len(known_idx), -1)[:, 0]

    def fit_predict(rows_fit, y_fit, rows_pred, ca, cb):
        mu = float(np.mean(y_fit))
        na, nb = int(ca.max()) + 2, int(cb.max()) + 2
        a, b = ca[rows_fit] + 1, cb[rows_fit] + 1          # -1 (пропуск) -> 0: своя строка факторов
        P, Q, ba, bb = _als(a, b, y_fit - mu, na, nb, seed=seed)
        qa, qb = ca[rows_pred] + 1, cb[rows_pred] + 1
        return mu + ba[qa] + bb[qb] + np.einsum("ij,ij->i", P[qa], Q[qb])

    kf = KFold(n_inner, shuffle=True, random_state=seed)
    tr_out, q_out = {}, [dict() for _ in query_idx]
    for pa, pb in pairs:
        ca, cb = codes[pa], codes[pb]
        name = f"__mf_{pa}__{pb}"
        vals = np.zeros(len(known_idx))
        for i_a, i_b in kf.split(known_idx):
            vals[i_b] = fit_predict(known_idx[i_a], y[i_a], known_idx[i_b], ca, cb)
        tr_out[name] = vals
        for i, q in enumerate(query_idx):
            q_out[i][name] = fit_predict(known_idx, y, q, ca, cb)
    return pd.DataFrame(tr_out), [pd.DataFrame(o) for o in q_out]
