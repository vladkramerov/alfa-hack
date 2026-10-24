"""Один эксперимент: признаки -> предобработка -> модель на каждом фолде -> OOF + test.

Форма таргета и предсказаний зависит от типа задачи (agent/ml/metrics.py): регрессия и ранжирование —
одно число на строку, классификация — вероятности классов, порядковая — число на шкале кодов классов,
multilabel — вероятность каждой метки (отдельная модель на метку)."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

import ag_ml_metrics as M
from ag_ml_dataset import Dataset
from ag_ml_features import add_generated, apply_features
from ag_ml_models import EXTRA, MODELS, NEEDS, FitContext, effective_params, refit_params, search_space
from ag_ml_preprocess import expand_datetimes, finalize_types
from ag_ml_supervised import (SUPERVISED_BLOCKS, entity_history, entity_last_index, id_factors, id_pairs,
                                 row_codes, target_encoding, text_oof, value_codes, varying_columns)
from ag_schemas import BINARY, MULTILABEL, RANKING, REGRESSION, PipelineSpec, TaskSpec

def build_matrices(ds: Dataset, spec: PipelineSpec) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Признаки без таргета: исходные -> код LLM -> даты -> блоки чистки/генерации -> типы ->
    сгенерированные автопоиском -> удаление отобранных. Статистики считаются по строкам train."""
    n_tr = len(ds.X_train)
    full = pd.concat([ds.X_train, ds.X_test], axis=0, ignore_index=True)
    drop = [c for c in spec.drop_columns if c in full.columns]
    if drop and not spec.feature_code:
        full = full.drop(columns=drop)
    if spec.feature_code:
        full = apply_features(full, [], spec.feature_code, n_tr)
    full = expand_datetimes(full, [c for c in ds.datetime_cols if c in full.columns])
    full = apply_features(full, [f for f in spec.features if f not in SUPERVISED_BLOCKS], None, n_tr)
    if "group_relative" in spec.features and ds.groups is not None:
        full = group_relative(full, ds.groups)
    full, cat_cols = finalize_types(full, n_tr)
    if spec.gen_features:
        full = add_generated(full, spec.gen_features, n_tr)
    drop = [c for c in spec.drop_columns if c in full.columns]
    if drop and len(drop) < full.shape[1]:
        full = full.drop(columns=drop)
        cat_cols = [c for c in cat_cols if c not in drop]
    return full.iloc[:n_tr].reset_index(drop=True), full.iloc[n_tr:].reset_index(drop=True), cat_cols


def group_relative(df: pd.DataFrame, groups: np.ndarray, max_cols: int = 40) -> pd.DataFrame:
    """Ранжирование: признак относительно своей группы (запроса, пользователя) — отклонение от среднего по
    группе и ранг внутри группы. Модель сравнивает объекты одной выдачи, и масштаб признака между группами
    ей только мешает. Таргет не нужен."""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2][:max_cols]
    if not num:
        return df
    g = pd.Series(groups[:len(df)], index=df.index)
    block = df[num].astype("float32")
    grp = block.groupby(g)
    dev = (block - grp.transform("mean")).add_suffix("__grp_dev")
    rank = grp.rank(pct=True).astype("float32").add_suffix("__grp_rank")
    return pd.concat([df, dev, rank], axis=1)


def target_setup(task: TaskSpec, spec: PipelineSpec, metric: M.Metric, y: np.ndarray):
    """Что учит модель: (y_fit, обратное преобразование, тип для моделей, функция потерь, её параметры).
    Функция потерь регрессии — под метрику (Metric.objective), если эксперимент не задал другую;
    log_* — обучение на log1p(y) (RMSLE — это RMSE в логарифмах)."""
    if task.task_type == MULTILABEL:
        return y, None, BINARY, "l2", {}
    if task.task_type == RANKING:
        return y[:, 0].astype(float), None, RANKING, "l2", {}
    if task.ordinal:
        return y.astype(float), None, REGRESSION, "l2", {}
    if task.is_classification:
        return y, None, task.task_type, "l2", {}
    obj = spec.objective or metric.objective
    args = dict(metric.args)
    y = np.asarray(y, dtype=float)
    ymin = float(np.nanmin(y))
    if spec.target_transform == "log1p":
        if ymin < 0:
            raise ValueError("log1p применим только к неотрицательному таргету регрессии")
        obj = obj[4:] if obj.startswith("log_") else obj
        obj = obj if obj in ("l1", "huber", "quantile") else "l2"
        return np.log1p(y), np.expm1, REGRESSION, obj, args
    if obj.startswith("log_"):
        if ymin >= 0:
            return np.log1p(y), np.expm1, REGRESSION, obj[4:], args
        obj = obj[4:]
    if obj in ("poisson", "tweedie") and ymin < 0 or obj == "gamma" and ymin <= 0:
        obj = "l2"
    return y, None, REGRESSION, obj, args


def _rows(a: np.ndarray, b: np.ndarray | None, t) -> dict:
    """Номера строк (в train+test) для частей вызова модели: a — обучение, b — проверка, t — предсказание."""
    return {"a": a, "b": b, "t": None if t is None else np.asarray(t)}


def run_pipeline(ds: Dataset, task: TaskSpec, spec: PipelineSpec, folds, n_jobs: int, seed: int,
                 timeout: float | None = None) -> dict:
    """Возвращает dict (пиклится через Pipe): cv_mean, cv_std, fold_scores, oof, test_pred, importance.
    timeout — сколько секунд у эксперимента: модели с долгим обучением (нейросеть, бустинг на широких
    данных) останавливаются раньше, чтобы эксперимент закончился с результатом, а не по таймауту."""
    t0 = time.time()
    deadline = t0 + timeout if timeout else None
    warnings.filterwarnings("ignore")  # процесс эксперимента: предупреждения библиотек только засоряют лог
    if spec.model not in MODELS:
        raise ValueError(f"модель {spec.model} недоступна; есть: {list(MODELS)}")
    EXTRA.clear()
    EXTRA.update(text=ds.text, grid=ds.grid, groups=ds.groups)
    need = NEEDS.get(spec.model)
    if need and EXTRA.get(need) is None:
        raise ValueError(f"модели {spec.model} нужны данные «{need}», в этой задаче их нет")
    metric = M.METRICS[task.metric]
    X_tr, X_te, cat_cols = build_matrices(ds, spec)
    n_tr = len(X_tr)
    y = ds.y
    k = task.n_outputs
    y_fit, inverse, fit_type, objective, obj_args = target_setup(task, spec, metric, y)
    n_cls = len(task.classes) if task.is_classification and not task.ordinal else (2 if fit_type == BINARY else 1)

    def new_ctx(cats):
        return FitContext(fit_type, n_cls, cats, n_jobs, seed, spec.class_weight, objective=objective,
                          objective_args=obj_args)

    ctx = new_ctx(cat_cols)
    base_fit = MODELS[spec.model]
    params = dict(spec.params)
    sup = [b for b in spec.features if b in SUPERVISED_BLOCKS]
    if "entity_history" in sup and ds.entity is None:
        sup.remove("entity_history")
    if "text_oof" in sup and ds.text is None:
        sup.remove("text_oof")
    augment = _make_augmenter(sup, ds, task, X_tr, X_te, y_fit, seed) if sup else None
    refit = ds.refit or bool(sup)   # признаки по таргету для теста считаются по всему train
    found = None
    if spec.feature_search and task.task_type != MULTILABEL:
        from ag_ml_feature_search import search_features
        found = search_features(X_tr, y, y_fit, inverse, task, metric, folds[-1], ds.period, n_jobs, seed,
                                spec.search_seconds, spec.feature_search, augment)
        if found["gen_features"] or found["drop_columns"]:
            spec = PipelineSpec.from_dict({**spec.to_dict(), "gen_features": spec.gen_features + found["gen_features"],
                                           "drop_columns": spec.drop_columns + found["drop_columns"]})
            X_tr, X_te, cat_cols = build_matrices(ds, spec)
            ctx = new_ctx(cat_cols)
            augment = _make_augmenter(sup, ds, task, X_tr, X_te, y_fit, seed) if sup else None
    split = spec.entity_mode == "split" and "entity_history" in sup and task.task_type == REGRESSION
    log_ratio = inverse is None and float(np.nanmin(y_fit)) >= 0 if split else False

    def wrap(c: FitContext):
        """Модель с обёртками: раздельная (объекты с историей / новые) и по метке на multilabel."""
        f = base_fit
        if split:
            f = _split_fit(f, spec.model, c, log_ratio)
        if task.task_type == MULTILABEL:
            f = _multilabel_fit(f, spec.model, c, y.shape[1])
        return f

    tuned = None
    if spec.tune_trials > 0:
        # тюнинг той же модели, что пойдёт в CV (с обёртками), со своим контекстом
        tuned = tune(wrap, spec, new_ctx(cat_cols), metric, X_tr, y, y_fit, inverse, folds, X_te, seed, augment)
        params = {**params, **tuned}
    fit = wrap(ctx)
    shape = (lambda n: (n, k)) if k > 1 else (lambda n: (n,))
    oof = np.zeros(shape(n_tr))
    test_pred = np.zeros(shape(len(X_te)))
    fold_scores, importance = [], {}
    covered = np.concatenate([va for _, va in folds])
    n_fits = len(folds) + (1 if refit else 0)
    for i, (tr_idx, va_idx) in enumerate(folds):
        Xa, Xb, Xt = X_tr.iloc[tr_idx], X_tr.iloc[va_idx], (X_te.iloc[:1] if refit else X_te)
        t_rows = n_tr + (np.arange(1) if refit else np.arange(len(X_te)))
        if augment:
            Xa, (Xb, Xt) = augment(tr_idx, [va_idx, None if not refit else "one"])
        ctx.rows = _rows(tr_idx, va_idx, t_rows)
        if deadline:
            ctx.time_budget = max(3.0, 0.9 * (deadline - time.time()) / (n_fits - i))
        pv, pt, imp = fit(dict(params), ctx, Xa, y_fit[tr_idx], Xb, y_fit[va_idx], Xt)
        if inverse is not None:
            pv, pt = inverse(pv), inverse(pt)
        oof[va_idx] = pv
        if not refit:
            test_pred += np.asarray(pt) / len(folds)
        fold_scores.append(M.score(metric, y[va_idx], pv))
        for f, v in imp.items():
            importance[f] = importance.get(f, 0.0) + float(v)
    if refit:
        all_idx = np.arange(n_tr)
        Xa, Xt = X_tr, X_te
        if augment:
            Xa, (Xt,) = augment(all_idx, [None])
        ctx.rows = _rows(all_idx, None, n_tr + np.arange(len(X_te)))
        if deadline:
            ctx.time_budget = max(3.0, 0.9 * (deadline - time.time()))
        _, pt, _ = fit(refit_params(spec.model, params, ctx.best_iters), ctx, Xa, y_fit, None, None, Xt)
        test_pred = inverse(pt) if inverse is not None else np.asarray(pt)
    total = sum(importance.values()) or 1.0
    importance = dict(sorted(((f, round(v / total, 4)) for f, v in importance.items()), key=lambda x: -x[1]))
    return {
        "cv_mean": M.score(metric, y[covered], oof[covered]),  # на всём OOF (стабильнее среднего по фолдам)
        "cv_std": float(np.std(fold_scores)),
        "fold_scores": [float(s) for s in fold_scores],
        "oof": oof,
        "test_pred": np.asarray(test_pred, dtype=float),
        "importance": importance,
        "n_features": X_tr.shape[1],
        "seconds": time.time() - t0,
        "tuned_params": tuned,
        "found_features": found,
    }


def _make_augmenter(sup: list[str], ds: Dataset, task: TaskSpec, X_tr: pd.DataFrame, X_te: pd.DataFrame,
                    y_fit: np.ndarray, seed: int):
    """Функция, которая добавляет признаки по таргету: known_idx — строки train, по таргету которых
    считаем; queries — индексы строк train или None (= весь test) или "one" (одна строка теста-заглушки)."""
    n = len(X_tr)
    cat_codes = {}
    if "target_encoding" in sup:
        for c in X_tr.columns:
            if isinstance(X_tr[c].dtype, pd.CategoricalDtype) and X_tr[c].nunique() > 2:
                cat_codes[c] = np.concatenate([X_tr[c].cat.codes.to_numpy(), X_te[c].cat.codes.to_numpy()])
        if ds.entity is not None:
            cat_codes["__entity"] = ds.entity
    # ключи дублей — по исходным признакам: производные (счётчик дублей и т.п.) дробили бы группы
    raw = pd.concat([ds.X_train, ds.X_test], ignore_index=True) if {"row_memory", "value_memory"} & set(sup) else None
    rows = row_codes(raw) if "row_memory" in sup else {}
    values = value_codes(raw) if "value_memory" in sup else {}
    if task.task_type == MULTILABEL:
        Y = y_fit
    elif task.is_classification and not task.ordinal:
        Y = np.eye(len(task.classes))[y_fit.astype(int)][:, 1:]
    else:
        Y = y_fit
    text_kind = "ml" if task.task_type == MULTILABEL else \
        "cls" if task.is_classification and not task.ordinal else "reg"
    mf_codes, mf_pairs = {}, []
    if "id_factors" in sup and (y_fit.ndim == 1 and (text_kind == "reg" or len(task.classes) == 2)):
        for c in X_tr.columns:
            if isinstance(X_tr[c].dtype, pd.CategoricalDtype):
                mf_codes[c] = np.concatenate([X_tr[c].cat.codes.to_numpy(), X_te[c].cat.codes.to_numpy()])
        if ds.groups is not None:
            mf_codes["__group"] = ds.groups
        mf_pairs = id_pairs(mf_codes, n)

    def to_all(q):
        if q is None:
            return np.arange(n, n + len(X_te))
        if isinstance(q, str):
            return np.array([n])
        return q

    def frame(idx):
        return (X_tr.iloc[idx] if idx.max() < n else X_te.iloc[idx - n]).reset_index(drop=True)

    if "entity_history" in sup:
        n_all = n + len(X_te)
        delta_cols = varying_columns(X_tr, ds.entity[:n])
        vals = np.vstack([X_tr[delta_cols].to_numpy(dtype=float), X_te[delta_cols].to_numpy(dtype=float)]) \
            if delta_cols else np.empty((n_all, 0))

    def deltas(known_idx, q):
        """Изменение меняющихся признаков объекта с его прошлого наблюдения."""
        last = entity_last_index(ds.entity, ds.period, known_idx, q)
        ok = ~np.isnan(last)
        prev = np.full((len(q), vals.shape[1]), np.nan)
        prev[ok] = vals[last[ok].astype(int)]
        return pd.DataFrame(vals[q] - prev, columns=[f"__hist_d_{c}" for c in delta_cols])

    def augment(known_idx: np.ndarray, queries: list):
        q_all = [to_all(q) for q in queries]
        base_k = X_tr.iloc[known_idx].reset_index(drop=True)
        base_q = [frame(q) for q in q_all]
        add_k, add_q = [], [[] for _ in q_all]

        def put(tk, tq):
            add_k.append(tk)
            for i, t in enumerate(tq):
                add_q[i].append(t)

        if cat_codes:
            put(*target_encoding(cat_codes, Y[known_idx], known_idx, q_all, seed))
        if rows:   # память по дублям: слабое сглаживание, группа из нескольких строк уже информативна
            put(*target_encoding(rows, Y[known_idx], known_idx, q_all, seed, smoothing=1.0))
        if values:   # память по точным значениям признаков
            put(*target_encoding(values, Y[known_idx], known_idx, q_all, seed, smoothing=10.0))
        if "text_oof" in sup:
            put(*text_oof(ds.text, y_fit[known_idx], known_idx, q_all, text_kind, seed))
        if mf_pairs:
            put(*id_factors(mf_codes, mf_pairs, y_fit[known_idx], known_idx, q_all, seed))
        if "entity_history" in sup:
            add_k.append(entity_history(ds.entity, ds.period, y_fit[known_idx], known_idx, known_idx))
            for i, q in enumerate(q_all):
                add_q[i].append(entity_history(ds.entity, ds.period, y_fit[known_idx], known_idx, q))
            if delta_cols:
                add_k.append(deltas(known_idx, known_idx))
                for i, q in enumerate(q_all):
                    add_q[i].append(deltas(known_idx, q))
        out_k = pd.concat([base_k] + add_k, axis=1)
        out_q = [pd.concat([b] + a, axis=1) for b, a in zip(base_q, add_q)]
        return out_k, out_q

    return augment


def _sub_rows(ctx: FitContext, ma: np.ndarray | None, mb: np.ndarray | None, mt: np.ndarray | None) -> dict | None:
    """Номера строк для подмножества частей (раздельная модель учит каждую подмодель на своих строках)."""
    r = ctx.rows
    if not r:
        return None
    pick = lambda idx, m: None if idx is None else (idx if m is None else np.asarray(idx)[m])  # noqa: E731
    return {"a": pick(r.get("a"), ma), "b": pick(r.get("b"), mb), "t": pick(r.get("t"), mt)}


def _multilabel_fit(fit, model: str, ctx: FitContext, n_labels: int):
    """Multilabel: отдельная бинарная модель на каждую метку (у каждой свои лучшие итерации для дообучения)."""
    subs = [ctx.child(task_type=BINARY, n_classes=2) for _ in range(n_labels)]

    def ml_fit(params, _ctx, Xa, Ya, Xb, Yb, Xt):
        refit = Xb is None
        pv = None if refit else np.zeros((len(Xb), n_labels))
        pt = np.zeros((len(Xt), n_labels))
        imp: dict = {}
        for j, c in enumerate(subs):
            c.rows, c.time_budget = _ctx.rows, (_ctx.time_budget / n_labels if _ctx.time_budget else None)
            p = refit_params(model, params, c.best_iters) if refit else dict(params)
            a, b, i = fit(p, c, Xa, np.asarray(Ya)[:, j], Xb, None if refit else np.asarray(Yb)[:, j], Xt)
            if not refit:
                pv[:, j] = np.asarray(a)[:, 1]
            pt[:, j] = np.asarray(b)[:, 1]
            for f, v in i.items():
                imp[f] = imp.get(f, 0.0) + float(v)
        return pv, pt, imp

    return ml_fit


def _split_fit(fit, model: str, ctx: FitContext, log_ratio: bool = False):
    """Раздельная модель для задач с историей объектов (регрессия):
    * объекты с прошлым значением таргета — отдельная модель на изменение таргета относительно прошлого
      значения: для неотрицательного таргета — на логарифм отношения log1p(y) - log1p(прошлое), то есть
      на относительное изменение (у больших и малых объектов оно сравнимо), иначе на разность;
    * новые объекты — общая модель на всех строках (признаки истории у новых пустые, и модель учится
      на таких же строках из прошлого), ранняя остановка — по новым объектам.
    Проверено на задаче 1 (валидация год 5): известные здания 23.0 -> 20.9 RMSE против одной общей модели;
    модель для новых без признаков истории была хуже общей (58.8 против 54.6)."""
    ctx_seen, ctx_new = ctx.child(), ctx.child()
    if log_ratio:
        fwd = lambda y, last: np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(last, 0, None))  # noqa: E731
        back = lambda r, last: np.expm1(r + np.log1p(np.clip(last, 0, None)))  # noqa: E731
    else:
        fwd = lambda y, last: y - last  # noqa: E731
        back = lambda r, last: r + last  # noqa: E731

    def split_fit(params, _ctx, Xa, ya, Xb, yb, Xt):
        refit = Xb is None
        last = lambda X: X["__hist_last"].to_numpy(dtype=float)  # noqa: E731
        sa, st = ~np.isnan(last(Xa)), ~np.isnan(last(Xt))
        sb = None if refit else ~np.isnan(last(Xb))
        p_new = refit_params(model, params, ctx_new.best_iters) if refit else dict(params)
        nb = None if refit else (~sb if (~sb).any() else np.ones(len(Xb), bool))
        ctx_new.rows = _sub_rows(_ctx, None, nb, None)
        ctx_new.time_budget = _ctx.time_budget
        pv_n, pt_n, imp = fit(p_new, ctx_new, Xa, ya, None if refit else Xb[nb], None if refit else yb[nb], Xt)
        pv = None if refit else np.zeros(len(Xb))
        pt = np.asarray(pt_n, dtype=float).copy()
        if not refit:
            pv[nb] = pv_n
        if sa.sum() >= 30 and (refit or sb.any()):
            iters = ctx_seen.best_iters or ctx_new.best_iters
            p_seen = refit_params(model, params, iters) if refit else dict(params)
            ctx_seen.rows = _sub_rows(_ctx, sa, sb, st if st.any() else np.arange(1))
            ctx_seen.time_budget = _ctx.time_budget
            pv_s, pt_s, imp_s = fit(p_seen, ctx_seen, Xa[sa], fwd(ya[sa], last(Xa)[sa]),
                                    None if refit else Xb[sb], None if refit else fwd(yb[sb], last(Xb)[sb]),
                                    Xt[st] if st.any() else Xt.iloc[:1])
            if not refit:
                pv[sb] = back(np.asarray(pv_s), last(Xb)[sb])
            if st.any():
                pt[st] = back(np.asarray(pt_s), last(Xt)[st])
            for f, v in imp_s.items():
                imp[f] = imp.get(f, 0.0) + float(v)
        if not refit:
            _ctx.best_iters.append(ctx_new.best_iters[-1] if ctx_new.best_iters else 0)
        return pv, pt, imp

    return split_fit


def tune(wrap, spec: PipelineSpec, ctx: FitContext, metric, X_tr, y, y_fit, inverse, folds, X_te, seed,
         augment=None) -> dict:
    """Optuna (TPE) по гиперпараметрам модели в пределах spec.tune_seconds.
    При 2–3 фолдах (валидация по времени) и на малых данных (до 30 тыс. строк) — по всем фолдам сразу: на
    задаче 1 тюнинг по одному последнему фолду улучшал его (28.78 -> 28.56) и портил другой (28.30 -> 28.90).
    На больших данных при 5 фолдах — по последнему, иначе проб слишком мало. wrap(ctx) — модель с обёртками для этого контекста."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_tr = len(X_tr)
    parts = []
    # на малых данных один фолд — шум (bank_churn: 1 400 строк проверки, F1 ±0.02 между фолдами), и Optuna
    # подбирает параметры под него; обучение там дешёвое — берём все фолды
    for tr_idx, va_idx in (folds if len(folds) <= 3 or n_tr <= 30_000 else folds[-1:]):
        X_a, X_b = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        if augment:
            X_a, (X_b,) = augment(tr_idx, [va_idx])
        parts.append((tr_idx, va_idx, X_a, X_b))
    va_all = np.concatenate([p[1] for p in parts])
    t_end = time.time() + max(5.0, spec.tune_seconds)

    def objective(trial):
        p = {**spec.params, **search_space(spec.model, trial)}
        c = ctx.child()
        fit = wrap(c)
        preds = []
        for tr_idx, va_idx, X_a, X_b in parts:
            c.rows = _rows(tr_idx, va_idx, va_idx[:1])
            c.time_budget = max(3.0, (t_end - time.time()) / len(parts))
            pv, _, _ = fit(p, c, X_a, y_fit[tr_idx], X_b, y_fit[va_idx], X_b.iloc[:1])  # тест не нужен
            preds.append(inverse(pv) if inverse is not None else np.asarray(pv))
        return M.score(metric, y[va_all], np.concatenate(preds))

    study = optuna.create_study(direction="maximize" if metric.greater_is_better else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8))
    # первая проба — текущие параметры, чтобы тюнинг не мог стартовать хуже исходной модели
    # (лишние ключи Optuna игнорирует: берутся только те, что запрашивает search_space)
    study.enqueue_trial(effective_params(spec.model, len(parts[-1][2]), spec.params, X_tr.shape[1], ctx.n_classes))
    study.optimize(objective, n_trials=spec.tune_trials, timeout=max(5.0, spec.tune_seconds), catch=(Exception,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return {}
    best = dict(study.best_params)
    if spec.model == "catboost":
        best["iterations"] = 4000
    return best


def warm_text_cache(ds: Dataset, task: TaskSpec, folds, seed: int) -> None:
    """Прогнозы линейной модели по тексту для фиксированных фолдов — один раз в родительском процессе:
    иначе каждый эксперимент с блоком text_oof заново учит ~25 моделей на TF-IDF."""
    if ds.text is None:
        return
    metric = M.METRICS[task.metric]
    y_fit = target_setup(task, PipelineSpec(), metric, ds.y)[0]
    kind = "ml" if task.task_type == MULTILABEL else "cls" if task.is_classification and not task.ordinal else "reg"
    n_tr, n_te = len(ds.X_train), len(ds.X_test)
    one = np.array([n_tr])
    for tr_idx, va_idx in folds:
        text_oof(ds.text, y_fit[tr_idx], tr_idx, [va_idx, one], kind, seed)
    all_idx = np.arange(n_tr)
    text_oof(ds.text, y_fit, all_idx, [np.arange(n_tr, n_tr + n_te)], kind, seed)
