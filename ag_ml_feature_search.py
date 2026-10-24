"""Автопоиск признаков: генерация кандидатов пачками и отбор, а не по одному признаку за эксперимент.

Работает внутри одного эксперимента (agent/ml/pipeline.py) на обучающей части последнего фолда:
она делится ещё раз на два внутренних разбиения (по времени — два последних периода, если найден
период, иначе два случайных), и все решения принимаются на них. Валидация последнего фолда в поиске
не участвует, а итоговое качество найденного набора агент потом проверяет полной CV, как любую гипотезу.

Важность признака — permutation importance на валидации внутреннего разбиения: насколько ухудшается
метрика быстрой модели, если перемешать значения признака. К данным добавляются шумовые колонки;
признак полезен, только если он важнее шума.
1. Отбор: признаки не важнее шума — кандидаты на удаление; удаление принимается, если качество
   не ухудшилось ни на одном разбиении.
2. Генерация (режим "full"): из самых важных числовых признаков строятся отношения, произведения и
   разности пар, из важных категорий — среднее числового признака по группе, отклонение от него и
   размер группы. Кандидаты проверяются пачками; остаются важнее шума, и только если вместе они
   улучшают качество в среднем и не ухудшают ни на одном разбиении.
Таргет в сами признаки не попадает: он нужен только чтобы решить, какие признаки оставить."""
from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import pandas as pd

import ag_ml_metrics as M
from ag_ml_features import _gen_values, generated_name
from ag_ml_models import _pad_proba

log = logging.getLogger("agent.feature_search")

NOISE = tuple(f"__noise_{kind}{i}" for kind in ("gauss", "unif", "perm") for i in range(2))


def _inner_splits(idx: np.ndarray, period: np.ndarray | None, rng, max_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """До двух внутренних разбиений: по времени (валидация — последний и предпоследний период) или случайных."""
    out = []
    if period is not None:
        p = period[idx]
        for last in np.unique(p[~np.isnan(p)])[::-1][:2]:
            a, b = idx[p < last], idx[p == last]
            if len(b) >= 0.1 * len(idx) and len(a) >= 0.25 * len(idx):
                out.append((a, b))
    if not out:
        for s in range(2):
            perm = np.random.default_rng(int(rng.integers(1 << 30)) + s).permutation(idx)
            cut = int(len(perm) * 0.75)
            out.append((np.sort(perm[:cut]), np.sort(perm[cut:])))
    return [(np.sort(rng.choice(a, max_rows, replace=False)) if len(a) > max_rows else a, b) for a, b in out]


def _proxy(task, n_jobs: int, seed: int):
    """Быстрая модель для поиска: LightGBM, без него — HistGradientBoosting из sklearn."""
    clf = task.is_classification and not task.learns_as_regression
    try:
        import lightgbm as lgb
        params = dict(n_estimators=600, learning_rate=0.1, num_leaves=31, min_child_samples=20, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, n_jobs=n_jobs, random_state=seed, verbose=-1)

        def fit(Xa, ya, Xb, yb):
            m = (lgb.LGBMClassifier if clf else lgb.LGBMRegressor)(**params)
            m.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(40, verbose=False)])
            return m
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        def fit(Xa, ya, Xb, yb):
            m = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(
                max_iter=300, learning_rate=0.1, categorical_features="from_dtype", random_state=seed)
            return m.fit(Xa, ya)
    return fit


def search_features(X_tr: pd.DataFrame, y: np.ndarray, y_fit: np.ndarray, inverse, task, metric: M.Metric, fold,
                    period: np.ndarray | None, n_jobs: int, seed: int, seconds: float, mode: str = "full",
                    augment=None, top_num: int = 8, top_cat: int = 3, batch: int = 40, max_keep: int = 30,
                    max_rows: int = 60_000) -> dict:
    t_end = time.time() + max(10.0, seconds)
    rng = np.random.default_rng(seed)
    splits = _inner_splits(np.asarray(fold[0]), period, rng, max_rows)
    fit = _proxy(task, n_jobs, seed)
    clf = task.is_classification and not task.learns_as_regression
    k = len(task.classes) if clf else 1
    sign = 1 if metric.greater_is_better else -1
    tol = 2e-3   # улучшение на внутреннем разбиении должно быть заметным, иначе на CV оно не переносится
    base_cols = list(X_tr.columns)
    cats = [c for c in base_cols if isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
    # уже сгенерированные признаки в новые не комбинируем (описание ссылается только на базовые колонки)
    nums = [c for c in base_cols if c not in cats and not c.startswith("__g_") and "|" not in c]
    sup = []   # признаки по таргету (target encoding, история объекта) — внутри каждого разбиения
    for a, b in splits:
        if augment is None:
            sup.append((None, None))
            continue
        fa, (fb,) = augment(a, [b])
        extra = [c for c in fa.columns if c not in X_tr.columns]
        sup.append((fa[extra].reset_index(drop=True), fb[extra].reset_index(drop=True)))
    log_lines: list[str] = []

    def loss(b: np.ndarray, pred: np.ndarray) -> float:
        """Потеря для permutation importance (меньше — лучше): для классификации logloss (гладкая),
        иначе метрика задачи."""
        if clf:
            return -float(np.mean(np.log(np.clip(pred[np.arange(len(b)), y_fit[b].astype(int)], 1e-15, 1))))
        return -sign * M.score(metric, y[b], pred)

    def predict(m, X):
        if clf:
            return _pad_proba(m.predict_proba(X), k, np.asarray(m.classes_).astype(int))
        p = m.predict(X)
        return inverse(p) if inverse is not None else p

    def run(cols: list[str], extra: dict[str, np.ndarray] | None = None, noise: bool = False, perm: bool = False):
        """Качество (метрика задачи) на каждом разбиении и средняя permutation importance признаков."""
        scores, imps = [], {}
        for (a, b), (sa, sb) in zip(splits, sup):
            Xa = X_tr[cols].iloc[a].reset_index(drop=True)
            Xb = X_tr[cols].iloc[b].reset_index(drop=True)
            add_a, add_b = [], []
            if extra:
                e = pd.DataFrame(extra).astype("float32")
                add_a.append(e.iloc[a].reset_index(drop=True))
                add_b.append(e.iloc[b].reset_index(drop=True))
            if sa is not None:
                add_a.append(sa)
                add_b.append(sb)
            if noise:
                top = X_tr[nums[0]].to_numpy() if nums else rng.normal(size=len(X_tr))
                for part, idx in ((add_a, a), (add_b, b)):
                    part.append(pd.DataFrame({
                        **{f"__noise_gauss{i}": rng.normal(size=len(idx)) for i in range(2)},
                        **{f"__noise_unif{i}": rng.uniform(size=len(idx)) for i in range(2)},
                        **{f"__noise_perm{i}": rng.permutation(top[idx]) for i in range(2)}}).astype("float32"))
            Xa, Xb = pd.concat([Xa] + add_a, axis=1), pd.concat([Xb] + add_b, axis=1)
            m = fit(Xa, y_fit[a], Xb, y_fit[b])
            pred = predict(m, Xb)
            scores.append(M.score(metric, y[b], pred))
            if perm:
                ref = loss(b, pred)
                for c in Xb.columns:
                    if not (c in cols or c in NOISE or (extra and c in extra)):
                        continue
                    saved = Xb[c].copy()
                    # перестановка с сохранением типа (у категорий — того же набора категорий)
                    Xb[c] = pd.Series(saved.to_numpy()[rng.permutation(len(saved))], index=Xb.index).astype(saved.dtype)
                    p2 = predict(m, Xb)
                    Xb[c] = saved
                    imps[c] = imps.get(c, 0.0) + (loss(b, p2) - ref) / len(splits)
        return scores, imps

    mean = lambda s: float(np.mean(s))  # noqa: E731

    def rel_gain(new, old) -> np.ndarray:
        n, o = np.asarray(new), np.asarray(old)
        return sign * (n - o) / np.maximum(np.abs(o), 1e-12)

    def improves(new, old, margin):
        """Добавление: в среднем лучше на margin и ни на одном разбиении не хуже на margin."""
        rel = rel_gain(new, old)
        return rel.mean() > margin and rel.min() > -margin

    def not_worse(new, old, margin):
        """Удаление: в среднем не хуже больше чем на margin (меньше признаков при равном качестве — лучше)."""
        return rel_gain(new, old).mean() >= -margin

    # 1. базовое качество и важности рядом с шумом
    s_base, imp = run(base_cols, noise=True, perm=True)
    thr = max(0.0, max(imp.get(n, 0.0) for n in NOISE))
    s_base, _ = run(base_cols)
    log_lines.append(f"база {metric.name}={mean(s_base):.5f} на {len(splits)} внутренних разбиениях "
                     f"({', '.join(f'{len(a)}/{len(b)}' for a, b in splits)} строк), порог шума {thr:.3g}")

    # 2. отбор: признаки не важнее шума
    keep = list(base_cols)
    dropped: list[str] = []
    weak = [c for c in base_cols if imp.get(c, 0.0) <= thr]
    if weak and len(base_cols) - len(weak) >= 3 and time.time() < t_end:
        cand = [c for c in base_cols if c not in weak]
        s_sel, _ = run(cand)
        if not_worse(s_sel, s_base, tol / 4):   # без слабых признаков не хуже (с небольшим допуском)
            keep, dropped = cand, weak
            log_lines.append(f"удалено {len(weak)} признаков не важнее шума: {metric.name} {mean(s_base):.5f} -> "
                             f"{mean(s_sel):.5f}")
            s_base = s_sel
        else:
            log_lines.append(f"удаление {len(weak)} слабых признаков ухудшает качество ({mean(s_sel):.5f}), оставлены")
    elif weak:
        log_lines.append(f"{len(weak)} из {len(base_cols)} признаков не важнее шума: остальных слишком мало, чтобы удалять")

    gen: list[str] = []
    if mode == "full" and time.time() < t_end:
        rank = lambda cs: sorted(cs, key=lambda c: -imp.get(c, 0.0))  # noqa: E731
        good_nums = [c for c in rank([c for c in nums if c in keep]) if imp.get(c, 0.0) > thr][:top_num]
        good_cats = [c for c in rank([c for c in cats if c in keep and "|" not in c])
                     if imp.get(c, 0.0) > thr and 2 <= X_tr[c].nunique() <= 5000][:top_cat]
        defs = []
        for x, z in itertools.combinations(good_nums, 2):
            defs += [f"ratio|{x}|{z}", f"prod|{x}|{z}", f"diff|{x}|{z}"]
        for c in good_cats:
            defs.append(f"grp_count|{c}")
            for x in good_nums[:5]:
                defs += [f"grp_mean|{x}|{c}", f"grp_dev|{x}|{c}"]
        values = {}
        for d in defs:
            if generated_name(d) in X_tr.columns:   # уже добавлен прошлым проходом
                continue
            v = _gen_values(X_tr, d, len(X_tr))
            if v is not None and v.notna().mean() > 0.5 and v.nunique() > 1:
                values[d] = v.to_numpy(dtype="float32")
        log_lines.append(f"кандидатов: {len(values)} (из {len(good_nums)} числовых и {len(good_cats)} категориальных)")
        passed: dict[str, float] = {}
        names = list(values)
        for i in range(0, len(names), batch):
            if time.time() > t_end:
                log_lines.append("время поиска вышло, проверены не все кандидаты")
                break
            chunk = names[i:i + batch]
            _, imp_b = run(keep, {generated_name(d): values[d] for d in chunk}, noise=True, perm=True)
            thr_b = max(0.0, max(imp_b.get(n, 0.0) for n in NOISE))
            for d in chunk:
                if imp_b.get(generated_name(d), 0.0) > thr_b:
                    passed[d] = imp_b[generated_name(d)]
        ranked = sorted(passed, key=lambda d: -passed[d])[:max_keep]
        log_lines.append(f"важнее шума: {len(ranked)}")
        # проверяем самые важные и все прошедшие; из двух примерно равных наборов берём меньший
        best_set, best_s = None, s_base
        for subset in (ranked[:max(1, len(ranked) // 3)], ranked):
            if not subset or time.time() > t_end or (best_set is not None and len(subset) == len(best_set)):
                continue
            s_gen, _ = run(keep, {generated_name(d): values[d] for d in subset})
            if improves(s_gen, s_base, tol) and (best_set is None or improves(s_gen, best_s, tol / 2)):
                best_set, best_s = list(subset), s_gen
        if best_set:
            gen = best_set
            log_lines.append(f"добавлено {len(gen)} признаков: {metric.name} {mean(s_base):.5f} -> {mean(best_s):.5f}")
            # второй проход отбора на итоговом наборе: с сильными новыми признаками часть старых
            # и новых перестаёт быть нужной
            if time.time() < t_end:
                ext = {generated_name(d): values[d] for d in gen}
                _, imp2 = run(keep, ext, noise=True, perm=True)
                thr2 = max(0.0, max(imp2.get(n, 0.0) for n in NOISE))
                weak2 = [c for c in keep + list(ext) if imp2.get(c, 0.0) <= thr2]
                if weak2 and len(keep) + len(ext) - len(weak2) >= 3:
                    keep2 = [c for c in keep if c not in weak2]
                    # производные от удалённых шумовых колонок тоже убираем
                    gen2 = [d for d in gen if generated_name(d) not in weak2
                            and not set(d.split("|")[1:]) & set(weak2)]
                    s2, _ = run(keep2, {generated_name(d): values[d] for d in gen2})
                    if not_worse(s2, best_s, tol / 4):
                        log_lines.append(f"второй проход отбора: убрано ещё {len(weak2)}: {mean(best_s):.5f} -> {mean(s2):.5f}")
                        dropped += [c for c in keep if c in weak2]
                        keep, gen = keep2, gen2
                    else:
                        log_lines.append(f"второй проход отбора: удаление {len(weak2)} ухудшает ({mean(s2):.5f}), оставлены")
        elif ranked:
            log_lines.append("новые признаки не дали заметного улучшения")
    for line in log_lines:
        log.info("автопоиск признаков: %s", line)
    return {"gen_features": gen, "drop_columns": dropped, "log": log_lines}
