"""Реестр моделей. Необязательные библиотеки подключаются, только если установлены.

Каждая модель: fit(params, ctx, X_tr, y_tr, X_va, y_va, X_te) -> (pred_va, pred_te, importance).
Классификация возвращает вероятности (n, K) для классов 0..K-1, регрессия и ранжирование — вектор.

Функция потерь регрессии подбирается под метрику (ctx.objective, см. Metric.objective): MAE и WAPE —
L1 (оптимум — медиана), MAPE — относительная ошибка, Poisson/Gamma/Tweedie — log-link, квантиль —
pinball. Учить RMSE-моделью задачу с MAE — частая ошибка: модель предсказывает среднее, а не медиану.

Модели, которым нужны данные не из таблицы (текст, пиксели, группы ранжирования), берут их из EXTRA
(заполняет pipeline.run_pipeline в процессе эксперимента) по номерам строк ctx.rows."""
from __future__ import annotations

import importlib.util
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ag_ml_preprocess import codes_frame, native_cats, to_numeric_matrix
from ag_schemas import BINARY, MULTICLASS, RANKING, REGRESSION


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


HAS_LGBM = _has("lightgbm")
HAS_CATBOOST = _has("catboost")
HAS_XGB = _has("xgboost")
HAS_TORCH = _has("torch")

# Данные не из таблицы, по всем строкам train+test: "text" — разреженная матрица TF-IDF,
# "grid" — массив (n, C, H, W) или (n, C, L), "groups" — коды групп ранжирования (только train).
EXTRA: dict[str, Any] = {}


@dataclass
class FitContext:
    task_type: str
    n_classes: int
    cat_cols: list[str]
    n_jobs: int
    seed: int
    class_weight: str | None
    time_budget: float | None = None   # сек на один фолд (мягкое ограничение)
    best_iters: list = None            # лучшие итерации ранней остановки по фолдам (для дообучения на всём train)
    objective: str = "l2"              # функция потерь регрессии: l2 | l1 | huber | mape | poisson | gamma | tweedie | quantile
    objective_args: dict = field(default_factory=dict)   # alpha квантиля, степень Tweedie
    rows: dict | None = None           # номера строк (в train+test) для X_tr / X_va / X_te текущего вызова
    sample_weight: np.ndarray | None = None   # веса строк train (на всю длину train), если заданы

    def __post_init__(self):
        if self.best_iters is None:
            self.best_iters = []

    def child(self, **kw) -> "FitContext":
        """Копия с собственным списком лучших итераций (подмодели раздельной модели и multilabel)."""
        d = dict(task_type=self.task_type, n_classes=self.n_classes, cat_cols=self.cat_cols, n_jobs=self.n_jobs,
                 seed=self.seed, class_weight=self.class_weight, time_budget=self.time_budget,
                 objective=self.objective, objective_args=dict(self.objective_args), rows=self.rows,
                 sample_weight=self.sample_weight)
        d.update(kw)
        return FitContext(**d)


def _is_reg(ctx: FitContext) -> bool:
    return ctx.task_type in (REGRESSION, RANKING)


def _class_weights(y: np.ndarray, k: int, mode: str | None) -> np.ndarray | None:
    """Вес каждого класса: balanced — обратно частоте, sqrt — корень из balanced (мягче: редкие классы
    поднимаются, но частые не теряют точность так сильно)."""
    if mode not in ("balanced", "sqrt"):
        return None
    counts = np.bincount(y, minlength=k).astype(float)
    w = len(y) / (k * np.maximum(counts, 1))
    return np.sqrt(w) if mode == "sqrt" else w


def _sample_weights(y: np.ndarray, ctx: "FitContext") -> np.ndarray | None:
    w = None
    if not _is_reg(ctx):
        cw = _class_weights(np.asarray(y, dtype=int), ctx.n_classes, ctx.class_weight)
        w = None if cw is None else cw[np.asarray(y, dtype=int)]
    elif ctx.objective == "mape_w":
        w = 1.0 / np.maximum(np.abs(np.asarray(y, dtype=float)), 1e-6)
        w = w / w.mean()
    if ctx.sample_weight is not None and ctx.rows and ctx.rows.get("a") is not None:
        sw = ctx.sample_weight[ctx.rows["a"]]
        if len(sw) == len(y):
            w = sw if w is None else w * sw
    return w


def _lgb_time_limit(seconds: float):
    """Остановка бустинга по времени: модель с тем, что успела выучить, лучше таймаута эксперимента."""
    import lightgbm as lgb
    t_end = time.time() + seconds

    def cb(env):
        if time.time() > t_end:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    cb.order = 40
    return cb


def _xgb_time_limit(seconds: float):
    import xgboost as xgb

    class TimeLimit(xgb.callback.TrainingCallback):
        def __init__(self):
            super().__init__()
            self.t_end = time.time() + seconds

        def after_iteration(self, model, epoch, evals_log):
            return time.time() > self.t_end

    return TimeLimit()


def _pred(fn, X):
    return None if X is None else fn(X)


def _pad_proba(p: np.ndarray, k: int, classes_seen: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    if classes_seen is not None and p.shape[1] != k:
        full = np.zeros((p.shape[0], k))
        full[:, classes_seen] = p
        p = full
    return p


def _constant_fit(ctx: FitContext, y_tr, X_va, X_te):
    """Таргет в обучающей части один и тот же (редкая метка в фолде): константа без модели."""
    if _is_reg(ctx):
        v = float(np.mean(y_tr))
        return (None if X_va is None else np.full(len(X_va), v)), np.full(len(X_te), v), {}
    prior = np.bincount(np.asarray(y_tr, dtype=int), minlength=ctx.n_classes) / max(len(y_tr), 1)
    return (None if X_va is None else np.tile(prior, (len(X_va), 1))), np.tile(prior, (len(X_te), 1)), {}


def size_defaults(model: str, n_rows: int, n_cols: int = 0, n_classes: int = 1) -> dict:
    """Стартовые гиперпараметры по размеру данных: на малых выборках неглубокие деревья и
    сильнее регуляризация (иначе бустинг переобучается), на больших и широких — быстрее учимся
    (иначе базовая модель на 784 пикселях и 10 классах не успевает и за 5 минут)."""
    cost = n_rows * max(n_cols, 1) * max(n_classes, 1)
    if model == "lgbm":
        if n_rows < 2000:
            p = dict(learning_rate=0.03, num_leaves=7, min_child_samples=10, reg_lambda=3.0, colsample_bytree=0.9)
        elif n_rows < 20000:
            p = dict(learning_rate=0.03, num_leaves=15, min_child_samples=20, reg_lambda=2.0)
        elif n_rows < 300000:
            # умеренная регуляризация: устойчивее при сдвиге распределения (валидация по времени);
            # проверено на задаче 1 и синтетике, дальше параметры подбирает Optuna
            p = dict(learning_rate=0.03, num_leaves=31, min_child_samples=50, reg_lambda=5.0)
        else:
            p = dict(learning_rate=0.08, num_leaves=127, min_child_samples=100)
        if cost > 3e7:
            p["learning_rate"] = max(p["learning_rate"], 0.08)
        if n_cols > 300:
            p.update(colsample_bytree=0.4, max_bin=63)
        return p
    if model == "xgboost":
        if n_rows < 2000:
            p = dict(learning_rate=0.03, max_depth=3, min_child_weight=3.0, reg_lambda=3.0)
        elif n_rows < 20000:
            p = dict(learning_rate=0.03, max_depth=5, min_child_weight=2.0)
        elif n_rows < 300000:
            p = dict(learning_rate=0.05, max_depth=7)
        else:
            p = dict(learning_rate=0.08, max_depth=8, min_child_weight=5.0)
        if cost > 3e7:
            p["learning_rate"] = max(p["learning_rate"], 0.08)
        if n_cols > 300:
            p.update(colsample_bytree=0.4, max_bin=64)
        return p
    if model == "catboost":
        p = dict(learning_rate=0.06, depth=6) if n_rows < 300000 else dict(learning_rate=0.1, depth=8)
        if cost > 3e7 or n_rows > 50000:
            p["learning_rate"] = 0.1
        if n_cols > 300:
            p["rsm"] = 0.3
        return p
    if model == "extratrees":
        p = dict(min_samples_leaf=1 if n_rows < 5000 else 2)
        if n_rows > 50000 or cost > 5e7:
            p["n_estimators"] = 200
        if n_cols > 300:
            p["max_features"] = "sqrt"
        return p
    return {}


def search_space(model: str, trial) -> dict:
    """Пространство поиска Optuna для тюнинга (agent/ml/pipeline.py: tune)."""
    if model in ("lgbm", "lgbm_rank"):
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 4, 256, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 3, 200, log=True),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
    if model == "xgboost":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 10),
                    min_child_weight=trial.suggest_float("min_child_weight", 0.5, 30.0, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True))
    if model == "catboost":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    depth=trial.suggest_int("depth", 3, 9),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
                    iterations=2000)
    if model == "extratrees":
        return dict(min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20, log=True),
                    max_features=trial.suggest_float("max_features", 0.1, 1.0))
    if model == "mlp":
        return dict(width=trial.suggest_categorical("width", [64, 128, 256, 512]),
                    alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
                    learning_rate=trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True))
    if model == "linear":
        return dict(alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True),
                    C=trial.suggest_float("C", 1e-3, 1e2, log=True))
    if model == "text_linear":
        return dict(C=trial.suggest_float("C", 0.05, 30.0, log=True),
                    alpha=trial.suggest_float("alpha", 0.05, 30.0, log=True))
    if model == "cnn":
        return dict(width=trial.suggest_categorical("width", [32, 48, 64]),
                    lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True),
                    dropout=trial.suggest_float("dropout", 0.0, 0.4))
    return {}


# ---------------- функции потерь под метрику ----------------
def _lgb_objective(ctx: FitContext) -> dict:
    o, a = ctx.objective, ctx.objective_args
    if ctx.task_type == RANKING or o == "l2":
        return {}
    return {"l1": {"objective": "l1"}, "huber": {"objective": "huber"}, "mape": {"objective": "mape"},
            "mape_w": {"objective": "l1"}, "poisson": {"objective": "poisson"}, "gamma": {"objective": "gamma"},
            "tweedie": {"objective": "tweedie", "tweedie_variance_power": float(a.get("power", 1.5))},
            "quantile": {"objective": "quantile", "alpha": float(a.get("alpha", 0.5))}}.get(o, {})


def _xgb_objective(ctx: FitContext) -> dict:
    o, a = ctx.objective, ctx.objective_args
    if ctx.task_type == RANKING or o == "l2":
        return {}
    return {"l1": {"objective": "reg:absoluteerror"}, "mape": {"objective": "reg:absoluteerror"},
            "mape_w": {"objective": "reg:absoluteerror"},
            "huber": {"objective": "reg:pseudohubererror"}, "poisson": {"objective": "count:poisson"},
            "gamma": {"objective": "reg:gamma"},
            "tweedie": {"objective": "reg:tweedie", "tweedie_variance_power": float(a.get("power", 1.5))},
            "quantile": {"objective": "reg:quantileerror", "quantile_alpha": float(a.get("alpha", 0.5))}}.get(o, {})


def _cb_loss(ctx: FitContext) -> str:
    o, a = ctx.objective, ctx.objective_args
    return {"l1": "MAE", "mape": "MAPE", "mape_w": "MAE", "huber": "Huber:delta=1.0", "poisson": "Poisson",
            "gamma": "Tweedie:variance_power=1.99", "tweedie": f"Tweedie:variance_power={float(a.get('power', 1.5))}",
            "quantile": f"Quantile:alpha={float(a.get('alpha', 0.5))}"}.get(o, "RMSE")


# ---------------- LightGBM ----------------
LGBM_DEFAULTS = dict(n_estimators=4000, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def _lgb_params(params, ctx, X_tr):
    k = ctx.n_classes if ctx.task_type == MULTICLASS else 1
    p = {**LGBM_DEFAULTS, **size_defaults("lgbm", len(X_tr), X_tr.shape[1], k), **_lgb_objective(ctx), **params,
         "n_jobs": ctx.n_jobs, "random_state": ctx.seed}
    return p


def fit_lgbm(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    import lightgbm as lgb
    warnings.filterwarnings("ignore", category=Warning, module="lightgbm")
    if not _is_reg(ctx) and len(np.unique(y_tr)) < 2:
        return _constant_fit(ctx, y_tr, X_va, X_te)
    p = _lgb_params(params, ctx, X_tr)
    es = int(p.pop("early_stopping_rounds", 100 if p["learning_rate"] < 0.06 else 50))
    sw = _sample_weights(y_tr, ctx)
    cats = native_cats(X_tr, ctx.cat_cols)
    X_tr, X_va, X_te = (None if X is None else codes_frame(X, ctx.cat_cols, cats) for X in (X_tr, X_va, X_te))
    if _is_reg(ctx):
        m = lgb.LGBMRegressor(**p)
    else:
        if ctx.task_type == MULTICLASS:
            p.setdefault("objective", "multiclass")
        m = lgb.LGBMClassifier(**p)
    tl = [_lgb_time_limit(ctx.time_budget)] if ctx.time_budget else []
    if X_va is None:   # дообучение на всём train с числом деревьев из CV
        m.fit(X_tr, y_tr, sample_weight=sw, categorical_feature=cats or "auto", callbacks=tl)
    else:
        m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(es, verbose=False)] + tl, categorical_feature=cats or "auto")
        ctx.best_iters.append(int(m.best_iteration_ or m.booster_.current_iteration()))
    imp = dict(zip(X_tr.columns, m.booster_.feature_importance("gain")))
    if _is_reg(ctx):
        return _pred(m.predict, X_va), m.predict(X_te), imp
    seen = m.classes_.astype(int)
    return _pred(lambda X: _pad_proba(m.predict_proba(X), ctx.n_classes, seen), X_va), \
        _pad_proba(m.predict_proba(X_te), ctx.n_classes, seen), imp


def fit_lgbm_rank(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    """LambdaRank: учится порядку внутри групп (ранжирование), NDCG на ранней остановке.
    Релевантность -> целые коды по возрастанию значений, выигрыш кода = само значение (линейный NDCG)."""
    import lightgbm as lgb
    groups = EXTRA.get("groups")
    rows = ctx.rows or {}
    if groups is None or rows.get("a") is None or len(rows["a"]) != len(X_tr):
        return fit_lgbm(params, ctx.child(task_type=REGRESSION), X_tr, y_tr, X_va, y_va, X_te)
    vals = np.unique(np.concatenate([np.asarray(y_tr, float)] + ([np.asarray(y_va, float)] if y_va is not None else [])))
    code = lambda v: np.searchsorted(vals, np.asarray(v, float))  # noqa: E731
    # выигрыш кода = сама релевантность, как в NDCG метрики (сдвиг к нулю только для отрицательных значений)
    base = vals if vals.min() >= 0 else vals - vals.min()
    gain = base.tolist() if not params.get("exp_gain") else (2.0 ** base - 1).tolist()
    p = {**_lgb_params({k: v for k, v in params.items() if k not in ("exp_gain", "eval_k")}, ctx, X_tr),
         "objective": "lambdarank", "label_gain": gain}
    k = int(params.get("eval_k", 10))
    p.setdefault("eval_at", [k])
    es = int(p.pop("early_stopping_rounds", 100 if p["learning_rate"] < 0.06 else 50))
    cats = native_cats(X_tr, ctx.cat_cols)
    X_tr, X_va, X_te = (None if X is None else codes_frame(X, ctx.cat_cols, cats) for X in (X_tr, X_va, X_te))

    def sort_by_group(X, y, g):
        o = np.argsort(g, kind="stable")
        _, sizes = np.unique(g[o], return_counts=True)
        return X.iloc[o], code(np.asarray(y)[o]), sizes

    Xa, ya, ga = sort_by_group(X_tr, y_tr, groups[rows["a"]])
    m = lgb.LGBMRanker(**p)
    tl = [_lgb_time_limit(ctx.time_budget)] if ctx.time_budget else []
    if X_va is None or rows.get("b") is None:
        m.fit(Xa, ya, group=ga, categorical_feature=cats or "auto", callbacks=tl)
    else:
        Xb, yb, gb = sort_by_group(X_va, y_va, groups[rows["b"]])
        m.fit(Xa, ya, group=ga, eval_set=[(Xb, yb)], eval_group=[gb], eval_at=[k],
              callbacks=[lgb.early_stopping(es, verbose=False)] + tl, categorical_feature=cats or "auto")
        ctx.best_iters.append(int(m.best_iteration_ or m.booster_.current_iteration()))
    imp = dict(zip(Xa.columns, m.booster_.feature_importance("gain")))
    return _pred(m.predict, X_va), m.predict(X_te), imp


# ---------------- CatBoost ----------------
CATBOOST_DEFAULTS = dict(iterations=4000, learning_rate=0.06, depth=6, l2_leaf_reg=3.0)


def _cb_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return X


def fit_catboost(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    from catboost import CatBoostClassifier, CatBoostRegressor
    if not _is_reg(ctx) and len(np.unique(y_tr)) < 2:
        return _constant_fit(ctx, y_tr, X_va, X_te)
    k = ctx.n_classes if ctx.task_type == MULTICLASS else 1
    p = {**CATBOOST_DEFAULTS, **size_defaults("catboost", len(X_tr), X_tr.shape[1], k), **params,
         "thread_count": ctx.n_jobs, "random_seed": ctx.seed, "verbose": False, "allow_writing_files": False}
    es = int(p.pop("early_stopping_rounds", 150))
    cats = [c for c in ctx.cat_cols if c in X_tr.columns]
    sw = _sample_weights(y_tr, ctx) if _is_reg(ctx) else None
    if _is_reg(ctx):
        p.setdefault("loss_function", _cb_loss(ctx))
        m = CatBoostRegressor(**p)
    else:
        if ctx.class_weight in ("balanced", "sqrt"):
            p["auto_class_weights"] = "Balanced" if ctx.class_weight == "balanced" else "SqrtBalanced"
        p.setdefault("loss_function", "MultiClass" if ctx.task_type == MULTICLASS else "Logloss")
        m = CatBoostClassifier(**p)
    if X_va is None:
        m.fit(_cb_frame(X_tr, cats), y_tr, cat_features=cats, sample_weight=sw)
        Xv = None
    else:
        m.fit(_cb_frame(X_tr, cats), y_tr, cat_features=cats, sample_weight=sw,
              eval_set=(_cb_frame(X_va, cats), y_va), early_stopping_rounds=es)
        ctx.best_iters.append(int(m.get_best_iteration() or p["iterations"]))
        Xv = _cb_frame(X_va, cats)
    imp = dict(zip(X_tr.columns, m.get_feature_importance()))
    Xt = _cb_frame(X_te, cats)
    if _is_reg(ctx):
        pr = (lambda X: m.predict(X, prediction_type="Exponent")) if p["loss_function"].split(":")[0] in \
            ("Poisson", "Tweedie") else m.predict
        return _pred(pr, Xv), pr(Xt), imp
    seen = np.asarray(m.classes_).astype(int)
    return _pred(lambda X: _pad_proba(m.predict_proba(X), ctx.n_classes, seen), Xv), \
        _pad_proba(m.predict_proba(Xt), ctx.n_classes, seen), imp


# ---------------- XGBoost ----------------
XGB_DEFAULTS = dict(n_estimators=4000, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=1.0, reg_lambda=1.0, tree_method="hist", max_cat_to_onehot=8)


def fit_xgb(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    import xgboost as xgb
    if not _is_reg(ctx) and len(np.unique(y_tr)) < 2:
        return _constant_fit(ctx, y_tr, X_va, X_te)
    k = ctx.n_classes if ctx.task_type == MULTICLASS else 1
    p = {**XGB_DEFAULTS, **size_defaults("xgboost", len(X_tr), X_tr.shape[1], k), **_xgb_objective(ctx), **params,
         "n_jobs": ctx.n_jobs, "random_state": ctx.seed, "enable_categorical": True}
    p["early_stopping_rounds"] = int(params.get("early_stopping_rounds", 100 if p["learning_rate"] < 0.06 else 50))
    if X_va is None:
        p.pop("early_stopping_rounds")
    sw = _sample_weights(y_tr, ctx)
    if _is_reg(ctx) and ctx.objective == "mape":
        sw = 1.0 / np.maximum(np.abs(np.asarray(y_tr, float)), 1e-6)
        sw = sw / sw.mean()
    cats = native_cats(X_tr, ctx.cat_cols)
    X_tr, X_va, X_te = (None if X is None else codes_frame(X, ctx.cat_cols, cats) for X in (X_tr, X_va, X_te))
    if ctx.time_budget:
        p["callbacks"] = [_xgb_time_limit(ctx.time_budget)]
    m = xgb.XGBRegressor(**p) if _is_reg(ctx) else xgb.XGBClassifier(**p)
    if X_va is None:
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    else:
        m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
        best = getattr(m, "best_iteration", None)
        ctx.best_iters.append(int(best + 1) if best is not None else int(m.get_booster().num_boosted_rounds()))
    imp = dict(zip(X_tr.columns, m.feature_importances_))
    if _is_reg(ctx):
        return _pred(m.predict, X_va), m.predict(X_te), imp
    seen = m.classes_.astype(int)
    return _pred(lambda X: _pad_proba(m.predict_proba(X), ctx.n_classes, seen), X_va), \
        _pad_proba(m.predict_proba(X_te), ctx.n_classes, seen), imp


# ---------------- sklearn ----------------
def _grow_forest(cls, p: dict, ctx: FitContext, X, y, sw):
    """Лес растёт порциями деревьев, пока есть время (ctx.time_budget): часть леса лучше таймаута."""
    total = int(p.pop("n_estimators", 400))
    if not ctx.time_budget:
        return cls(n_estimators=total, **p).fit(X, y, sample_weight=sw)
    t0 = time.time()
    step = max(16, total // 8)
    m = cls(n_estimators=min(step, total), warm_start=True, **p)
    m.fit(X, y, sample_weight=sw)
    while m.n_estimators < total:
        per_tree = (time.time() - t0) / m.n_estimators
        if time.time() + per_tree * step > t0 + ctx.time_budget:
            break
        m.n_estimators = min(total, m.n_estimators + step)
        m.fit(X, y, sample_weight=sw)
    return m


def fit_extratrees(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    p = {"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.5,
         **size_defaults("extratrees", len(X_tr), X_tr.shape[1], ctx.n_classes), **params,
         "n_jobs": ctx.n_jobs, "random_state": ctx.seed}
    A = [None if X is None else np.nan_to_num(to_numeric_matrix(X, ctx.cat_cols), nan=-999.0) for X in (X_tr, X_va, X_te)]
    if _is_reg(ctx):
        if ctx.objective in ("poisson", "gamma", "tweedie") and np.min(y_tr) >= 0:
            p["criterion"] = "poisson"
        m = _grow_forest(ExtraTreesRegressor, p, ctx, A[0], y_tr, _sample_weights(y_tr, ctx))
        return _pred(m.predict, A[1]), m.predict(A[2]), dict(zip(X_tr.columns, m.feature_importances_))
    if len(np.unique(y_tr)) < 2:
        return _constant_fit(ctx, y_tr, X_va, X_te)
    cw = _class_weights(y_tr, ctx.n_classes, ctx.class_weight)
    if cw is not None:
        p["class_weight"] = {int(i): float(cw[i]) for i in np.unique(y_tr)}
    m = _grow_forest(ExtraTreesClassifier, p, ctx, A[0], y_tr, None)
    seen = m.classes_.astype(int)
    proba = lambda X: _pad_proba(m.predict_proba(X), ctx.n_classes, seen)  # noqa: E731
    return _pred(proba, A[1]), proba(A[2]), dict(zip(X_tr.columns, m.feature_importances_))


def _linear_pre(X_tr, ctx, scaler="standard", max_cat=200, sparse_out=True):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

    num = [c for c in X_tr.columns if c not in ctx.cat_cols]
    cats = [c for c in ctx.cat_cols if c in X_tr.columns and X_tr[c].nunique() <= max_cat]
    sc = StandardScaler() if scaler == "standard" else \
        QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=100_000, random_state=ctx.seed)
    parts = []
    if num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), sc), num))
    if cats:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                           OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                                         sparse_output=sparse_out)), cats))
    pre = ColumnTransformer(parts, sparse_threshold=0.3 if sparse_out else 0.0)
    prep = lambda X: None if X is None else X[num + cats].astype({c: "object" for c in cats})  # noqa: E731
    return pre, prep, bool(num or cats)


def fit_linear(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    from sklearn.linear_model import (GammaRegressor, HuberRegressor, LogisticRegression, PoissonRegressor, Ridge,
                                      TweedieRegressor)
    from sklearn.pipeline import make_pipeline

    pre, prep, ok = _linear_pre(X_tr, ctx)
    if not ok:
        raise ValueError("нет признаков для линейной модели")
    if _is_reg(ctx):
        a = float(params.get("alpha", 1.0))
        o = ctx.objective
        y_tr = np.asarray(y_tr, float)
        if o == "poisson" and y_tr.min() >= 0:
            est = PoissonRegressor(alpha=a / max(len(y_tr), 1), max_iter=300)
        elif o == "gamma" and y_tr.min() > 0:
            est = GammaRegressor(alpha=a / max(len(y_tr), 1), max_iter=300)
        elif o == "tweedie" and y_tr.min() >= 0:
            est = TweedieRegressor(power=float(ctx.objective_args.get("power", 1.5)), alpha=a / max(len(y_tr), 1),
                                   max_iter=300, link="log")
        elif o in ("l1", "mape", "mape_w", "huber", "quantile") and len(y_tr) <= 200_000:
            est = HuberRegressor(alpha=a * 1e-4, max_iter=200, epsilon=1.35)
        else:
            est = Ridge(alpha=a)
        model = make_pipeline(pre, est)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(est, HuberRegressor):
                mu, sd = float(np.median(y_tr)), float(np.std(y_tr) or 1.0)
                model.fit(prep(X_tr), (y_tr - mu) / sd)
                f = lambda X: model.predict(X) * sd + mu  # noqa: E731
                return _pred(f, prep(X_va)), f(prep(X_te)), {}
            model.fit(prep(X_tr), y_tr)
        return _pred(model.predict, prep(X_va)), model.predict(prep(X_te)), {}
    if len(np.unique(y_tr)) < 2:
        return _constant_fit(ctx, y_tr, X_va, X_te)
    cw = _class_weights(y_tr, ctx.n_classes, ctx.class_weight)
    cw = None if cw is None else {int(i): float(cw[i]) for i in np.unique(y_tr)}
    model = make_pipeline(pre, LogisticRegression(C=float(params.get("C", 1.0)), max_iter=600, class_weight=cw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prep(X_tr), y_tr)
    seen = model.classes_.astype(int)
    proba = lambda X: _pad_proba(model.predict_proba(X), ctx.n_classes, seen)  # noqa: E731
    return _pred(proba, prep(X_va)), proba(prep(X_te)), {}


def fit_text_linear(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    """Линейная модель на мешке слов и символьных n-грамм (TF-IDF, EXTRA['text']) плюс one-hot категорий и
    нормированные числа. Для текстовых задач без предобученных моделей это сильная и быстрая база."""
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier

    T = EXTRA.get("text")
    rows = ctx.rows or {}
    if T is None or rows.get("a") is None or len(rows["a"]) != len(X_tr):
        return fit_linear(params, ctx, X_tr, y_tr, X_va, y_va, X_te)
    pre, prep, ok = _linear_pre(X_tr, ctx)
    parts_tr = [T[rows["a"]]]
    if ok:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tab_tr = pre.fit_transform(prep(X_tr))
        parts_tr.append(sparse.csr_matrix(tab_tr) * float(params.get("tab_weight", 0.5)))

    def mat(X, key):
        if X is None:
            return None
        parts = [T[rows[key]]]
        if ok:
            parts.append(sparse.csr_matrix(pre.transform(prep(X))) * float(params.get("tab_weight", 0.5)))
        return sparse.hstack(parts).tocsr()

    A = sparse.hstack(parts_tr).tocsr()
    Bm, Tm = mat(X_va, "b"), mat(X_te, "t")
    if _is_reg(ctx):
        m = Ridge(alpha=float(params.get("alpha", 1.0)), solver="sparse_cg" if A.shape[0] > 5000 else "auto")
        y = np.asarray(y_tr, float)
        mu = float(np.mean(y))
        m.fit(A, y - mu)
        f = lambda M_: m.predict(M_) + mu  # noqa: E731
        return _pred(f, Bm), f(Tm), {}
    if len(np.unique(y_tr)) < 2:
        return _constant_fit(ctx, y_tr, X_va, X_te)
    cw = _class_weights(y_tr, ctx.n_classes, ctx.class_weight)
    cw = None if cw is None else {int(i): float(cw[i]) for i in np.unique(y_tr)}
    solver = params.get("solver") or ("lr" if A.shape[0] <= 20000 else "sgd")
    C = float(params.get("C", 2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if solver == "sgd":
            m = SGDClassifier(loss="log_loss", alpha=1.0 / (C * A.shape[0]), max_iter=40, tol=1e-4,
                              class_weight=cw, random_state=ctx.seed, n_jobs=ctx.n_jobs).fit(A, y_tr)
        else:
            m = LogisticRegression(C=C, max_iter=300, class_weight=cw).fit(A, y_tr)
    seen = m.classes_.astype(int)
    proba = lambda M_: _pad_proba(m.predict_proba(M_), ctx.n_classes, seen)  # noqa: E731
    return _pred(proba, Bm), proba(Tm), {}


def fit_mlp(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    """Нейросеть (sklearn MLP): числовые признаки — квантильная нормализация, категории — one-hot.
    На табличных данных редко лучше бустинга, но ошибается иначе и полезна в ансамбле."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline

    pre, prep, ok = _linear_pre(X_tr, ctx, scaler="quantile", max_cat=100, sparse_out=False)
    if not ok:
        raise ValueError("нет признаков для нейросети")
    width = int(params.get("width", 256))
    kw = dict(hidden_layer_sizes=(width, width // 2), alpha=float(params.get("alpha", 1e-4)),
              learning_rate_init=float(params.get("learning_rate", 1e-3)), batch_size=512,
              max_iter=int(params.get("max_iter", 60)), early_stopping=True, validation_fraction=0.1,
              n_iter_no_change=6, random_state=ctx.seed)
    if _is_reg(ctx):
        mu, sd = float(np.mean(y_tr)), float(np.std(y_tr) or 1.0)
        model = make_pipeline(pre, MLPRegressor(**kw))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prep(X_tr), (np.asarray(y_tr, float) - mu) / sd)
        back = lambda X: model.predict(X) * sd + mu  # noqa: E731
        return _pred(back, prep(X_va)), back(prep(X_te)), {}
    if len(np.unique(y_tr)) < 2:
        return _constant_fit(ctx, y_tr, X_va, X_te)
    model = make_pipeline(pre, MLPClassifier(**kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prep(X_tr), y_tr)
    seen = model.classes_.astype(int)
    proba = lambda X: _pad_proba(model.predict_proba(X), ctx.n_classes, seen)  # noqa: E731
    return _pred(proba, prep(X_va)), proba(prep(X_te)), {}


# ---------------- свёрточная сеть (torch) для пикселей и сигналов ----------------
def fit_cnn(params, ctx: FitContext, X_tr, y_tr, X_va, y_va, X_te):
    """Небольшая свёрточная сеть на «сетке» признаков (EXTRA['grid']): 2D для изображений (C, H, W),
    1D для многоканальных сигналов (C, L). Ранняя остановка по проверочной части фолда, при дообучении —
    медиана лучших эпох. Время одного обучения ограничено ctx.time_budget."""
    import torch
    from torch import nn

    G = EXTRA.get("grid")
    rows = ctx.rows or {}
    if G is None or rows.get("a") is None or len(rows["a"]) != len(X_tr):
        raise ValueError("нет сетки признаков для свёрточной сети")
    torch.set_num_threads(max(1, ctx.n_jobs))
    torch.manual_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    two_d = G.ndim == 4
    reg = _is_reg(ctx)
    k_out = 1 if reg else ctx.n_classes
    width = int(params.get("width", 32))
    drop = float(params.get("dropout", 0.2))
    lr = float(params.get("lr", 3e-3))
    max_epochs = int(params.get("epochs", 30))
    bs = int(params.get("batch_size", 256))
    a_idx = rows["a"]
    mean = float(np.mean(G[a_idx[: 5000]]))
    std = float(np.std(G[a_idx[: 5000]]) or 1.0)
    if reg:
        y_mu, y_sd = float(np.mean(y_tr)), float(np.std(y_tr) or 1.0)

    def block(cin, cout, double=True):
        conv, bn, mp = (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d) if two_d else (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d)
        ks, pad = (3, 1) if two_d else (5, 2)
        layers = [conv(cin, cout, ks, padding=pad), bn(cout), nn.ReLU()]
        if double:
            layers += [conv(cout, cout, ks, padding=pad), bn(cout), nn.ReLU()]
        return nn.Sequential(*layers, mp(2), nn.Dropout(drop / 2))

    C = G.shape[1]
    pool = nn.AdaptiveAvgPool2d(1) if two_d else nn.AdaptiveAvgPool1d(1)
    # первый блок с одной свёрткой: на полном разрешении она самая дорогая, а пользы от второй там меньше всего
    net = nn.Sequential(block(C, width, double=False), block(width, width * 2), block(width * 2, width * 4), pool,
                        nn.Flatten(), nn.Linear(width * 4, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, k_out))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    w = _class_weights(np.asarray(y_tr, dtype=int), ctx.n_classes, ctx.class_weight) if not reg else None
    loss_fn = nn.MSELoss() if reg else nn.CrossEntropyLoss(
        weight=None if w is None else torch.tensor(w, dtype=torch.float32), label_smoothing=0.05)

    def tensor(idx):
        return torch.from_numpy(((G[idx] - mean) / std).astype(np.float32))

    def target(y):
        return torch.tensor((np.asarray(y, float) - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1) if reg \
            else torch.tensor(np.asarray(y, dtype=np.int64))

    def augment(xb):
        s_ = int(params.get("shift", 2))
        if s_ == 0:
            return xb
        if two_d:
            dx, dy = rng.integers(-s_, s_ + 1, size=2)
            return torch.roll(xb, shifts=(int(dx), int(dy)), dims=(2, 3))
        return torch.roll(xb, shifts=int(rng.integers(-2 * s_, 2 * s_ + 1)), dims=2)

    def predict(idx):
        net.eval()
        out = []
        with torch.inference_mode():   # пачки по 256: на CPU вдвое быстрее, чем по 1024 (кэш)
            for i in range(0, len(idx), 256):
                o = net(tensor(idx[i:i + 256]))
                out.append(o.numpy()[:, 0] * y_sd + y_mu if reg else torch.softmax(o, 1).numpy())
        return np.concatenate(out) if out else np.zeros((0,) if reg else (0, k_out))

    yt = target(y_tr)
    b_idx = rows.get("b") if X_va is not None else None
    fixed = None if b_idx is not None else max(2, int(np.median(ctx.best_iters))) if ctx.best_iters else None
    epochs = fixed or max_epochs
    steps = (len(a_idx) + bs - 1) // bs
    t_start = time.time()
    t_end = t_start + (ctx.time_budget or 1e9)
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    sched = None
    # предсказание проверки и теста тоже занимает время: запас на него — по замеру прямого прохода
    n_pred = (len(b_idx) if b_idx is not None else 0) + len(rows["t"])
    t_p = time.time()
    predict(a_idx[:bs])
    reserve = 1.2 * n_pred * (time.time() - t_p) / min(bs, len(a_idx))
    if ctx.time_budget and reserve > 0.7 * ctx.time_budget:
        # даже без обучения не успеть предсказать: лучше сразу отказаться, чем выйти за таймаут
        raise ValueError(f"мало времени для сети: только прогноз займёт ~{reserve:.0f} с из {ctx.time_budget:.0f}")
    pv_best = None
    out_of_time = False
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(a_idx))
        for i in range(0, len(perm), bs):
            if ep == 0:   # разогрев: линейный рост шага в первую эпоху
                for g_ in opt.param_groups:
                    g_["lr"] = lr * min(1.0, (i // bs + 1) / steps)
            sel = perm[i:i + bs]
            xb = augment(tensor(a_idx[sel]))
            opt.zero_grad()
            loss = loss_fn(net(xb), yt[sel])
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if time.time() + reserve > t_end:   # время вышло посреди эпохи
                out_of_time = True
                break
        if out_of_time and b_idx is None:
            break
        if ep == 0 and not out_of_time:
            # косинусное затухание на столько эпох, сколько успеем по времени
            ep_time = (time.time() - t_start) + reserve * (len(b_idx) if b_idx is not None else 0) / n_pred
            left = max(1, min(epochs - 1, int((t_end - time.time()) / max(ep_time * 1.15, 1e-3))))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=left * steps, eta_min=lr * 0.02)
            epochs = 1 + left
        if b_idx is not None:
            pv = predict(b_idx)
            yv = np.asarray(y_va)
            vl = float(np.mean((pv - yv) ** 2)) if reg else \
                float(-np.mean(np.log(np.clip(pv[np.arange(len(yv)), yv.astype(int)], 1e-9, 1))))
            if vl < best - 1e-4:
                best, best_ep, bad, pv_best = vl, ep + 1, 0, pv
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= int(params.get("patience", 5)):
                    break
        if out_of_time or time.time() > t_end:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
        ctx.best_iters.append(best_ep)
    pv = None if b_idx is None else pv_best if pv_best is not None else predict(b_idx)
    return pv, predict(rows["t"]), {}


def refit_params(model: str, params: dict, best_iters: list[int]) -> dict:
    """Параметры дообучения на всём train: число итераций — медиана лучших по фолдам, +10% на объём."""
    if not best_iters:
        return dict(params)
    it = int(np.median(best_iters) * 1.1) + 1
    key = {"lgbm": "n_estimators", "lgbm_rank": "n_estimators", "xgboost": "n_estimators",
           "catboost": "iterations", "cnn": "epochs"}.get(model)
    return {**params, key: it} if key else dict(params)


def effective_params(model: str, n_rows: int, params: dict, n_cols: int = 0, n_classes: int = 1) -> dict:
    """Параметры, с которыми модель реально обучится (для стартовой точки тюнинга)."""
    base = {"lgbm": LGBM_DEFAULTS, "lgbm_rank": LGBM_DEFAULTS, "xgboost": XGB_DEFAULTS,
            "catboost": CATBOOST_DEFAULTS}.get(model, {})
    return {**base, **size_defaults("lgbm" if model == "lgbm_rank" else model, n_rows, n_cols, n_classes), **params}


MODELS: dict[str, Any] = {}
if HAS_LGBM:
    MODELS["lgbm"] = fit_lgbm
    MODELS["lgbm_rank"] = fit_lgbm_rank
if HAS_CATBOOST:
    MODELS["catboost"] = fit_catboost
if HAS_XGB:
    MODELS["xgboost"] = fit_xgb
MODELS["extratrees"] = fit_extratrees
MODELS["linear"] = fit_linear
MODELS["text_linear"] = fit_text_linear
MODELS["mlp"] = fit_mlp
if HAS_TORCH:
    MODELS["cnn"] = fit_cnn

MODEL_DOCS = {
    "lgbm": "LightGBM, нативные категории; params: learning_rate, num_leaves, min_child_samples, colsample_bytree, subsample, reg_lambda, reg_alpha, max_depth",
    "lgbm_rank": "LightGBM LambdaRank (только ранжирование): учится порядку внутри групп; params как у lgbm",
    "catboost": "CatBoost, сильный на категориальных; params: learning_rate, depth, l2_leaf_reg, iterations",
    "xgboost": "XGBoost hist; params: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda",
    "extratrees": "ExtraTrees (sklearn); params: n_estimators, min_samples_leaf, max_features",
    "linear": "Ridge / LogisticRegression (или Poisson/Gamma/Huber под метрику) с one-hot; params: alpha (регрессия) или C (классификация)",
    "text_linear": "линейная модель на TF-IDF слов и символьных n-грамм текстовых колонок + таблица (только если есть текст); params: C, alpha, solver (lr|sgd)",
    "mlp": "нейросеть (sklearn MLP, 2 слоя) на нормализованных признаках; params: width, alpha, learning_rate, max_iter",
    "cnn": "свёрточная сеть (torch) на пикселях изображения или многоканальном сигнале (только если найдена сетка признаков); params: width, lr, dropout, epochs",
}

# Модели, которым нужна модальность: без неё в очереди и в ответах LLM они не предлагаются
NEEDS = {"text_linear": "text", "cnn": "grid", "lgbm_rank": "groups"}


def default_model() -> str:
    return next(iter(MODELS))


def task_kind_ok(task_type: str) -> bool:
    return task_type in (REGRESSION, BINARY, MULTICLASS, RANKING)
