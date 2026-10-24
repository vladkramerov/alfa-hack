"""Постобработка под метрику, ансамбль и запись сабмита.

Все подборы (веса ансамбля, веса классов, порог) делаются на половине A OOF и
принимаются, только если улучшают половину B. Это защита от подгонки под CV."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import ag_ml_metrics as M
from ag_schemas import MULTILABEL, RANKING, REGRESSION, RunResult, TaskSpec

log = logging.getLogger("agent.finalize")


@dataclass
class Solution:
    name: str
    oof: np.ndarray
    test_pred: np.ndarray
    post: dict = field(default_factory=dict)
    cv: float = float("nan")          # на всём OOF с постобработкой
    holdout_b: float = float("nan")   # на половине B
    members: dict[int, float] = field(default_factory=dict)


# ---------------- постобработка ----------------
def fit_postprocess(metric: M.Metric, y: np.ndarray, oof: np.ndarray, A: np.ndarray, B: np.ndarray) -> dict:
    """Подбор на половине A, принимается, только если половина B стала лучше:
    * метки из вероятностей — порог (бинарная), веса классов (мультикласс), пороги кодов (порядковая),
      порог каждой метки (multilabel);
    * регрессия — масштаб предсказаний (MAPE выгоднее занижать) и обрезка по диапазону таргета."""
    oof = np.asarray(oof)
    if metric.kind == "value":
        return _fit_value_post(metric, y, oof, A, B) if REGRESSION in metric.tasks else {}
    if metric.kind != "label":
        return {}
    mode = M.DECODE.get("mode")
    if mode == "ordinal" or oof.ndim == 1:
        return _fit_ordinal(metric, y, oof, A, B)
    if mode == "multilabel":
        return _fit_multilabel(metric, y, oof, A, B)
    k = oof.shape[1]
    base_b = M.score(metric, y[B], oof[B])
    if k == 2:
        grid = np.unique(np.quantile(oof[A, 1], np.linspace(0.02, 0.98, 97)))
        scores = [M.score(metric, y[A], oof[A], {"threshold": t}) for t in grid]
        i = int(np.argmax(scores)) if metric.greater_is_better else int(np.argmin(scores))
        t = float(grid[i])
        if metric.better(M.score(metric, y[B], oof[B], {"threshold": t}), base_b):
            return {"threshold": t}
        return {}
    w = np.ones(k)
    mults = np.array([0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 3.0])
    cur = M.score(metric, y[A], oof[A], {"weights": w})
    for _round in range(3):
        improved = False
        for c in range(k):
            best_m, best_s = 1.0, cur
            for m in mults:
                w2 = w.copy()
                w2[c] *= m
                s = M.score(metric, y[A], oof[A], {"weights": w2})
                if metric.better(s, best_s) and abs(s - best_s) > 1e-6:
                    best_m, best_s = m, s
            if best_m != 1.0:
                w[c] *= best_m
                cur = best_s
                improved = True
        if not improved:
            break
    if metric.better(M.score(metric, y[B], oof[B], {"weights": w}), base_b):
        return {"weights": (w / w.max()).tolist()}
    return {}


def _coord_thresholds(score_fn, t0: np.ndarray, cands: list[np.ndarray], gib: bool, rounds: int = 3) -> np.ndarray:
    """Покоординатный подбор порогов: по очереди каждый порог по сетке кандидатов при остальных фиксированных."""
    t = t0.astype(float).copy()
    sign = 1 if gib else -1
    cur = sign * score_fn(t)
    for _ in range(rounds):
        moved = False
        for i in range(len(t)):
            best_v, best_s = t[i], cur
            for v in cands[i]:
                t2 = t.copy()
                t2[i] = v
                s = sign * score_fn(t2)
                if s > best_s + 1e-9:
                    best_v, best_s = v, s
            if best_v != t[i]:
                t[i], cur, moved = best_v, best_s, True
        if not moved:
            break
    return t


def _fit_ordinal(metric, y, oof, A, B) -> dict:
    """Порядковые классы: пороги между кодами подбираются под метрику (QWK) вместо k-0.5."""
    k = int(M.DECODE.get("k") or (int(np.max(y)) + 1))
    if k < 2:
        return {}
    t0 = np.arange(k - 1) + 0.5
    lo, hi = float(np.min(oof[A])), float(np.max(oof[A]))
    cands = [np.linspace(max(lo, c - 0.7), min(hi, c + 0.7), 29) for c in t0]
    fn = lambda t: M.score(metric, y[A], oof[A], {"thresholds": np.sort(t).tolist()})  # noqa: E731
    t = np.sort(_coord_thresholds(fn, t0, cands, metric.greater_is_better))
    if metric.better(M.score(metric, y[B], oof[B], {"thresholds": t.tolist()}), M.score(metric, y[B], oof[B])):
        return {"thresholds": t.tolist()}
    return {}


def _fit_multilabel(metric, y, oof, A, B) -> dict:
    """Multilabel с метрикой по меткам: свой порог у каждой метки."""
    L = oof.shape[1]
    t0 = np.full(L, 0.5)
    cands = [np.unique(np.quantile(oof[A, j], np.linspace(0.05, 0.95, 31))) for j in range(L)]
    fn = lambda t: M.score(metric, y[A], oof[A], {"thresholds": list(t)})  # noqa: E731
    t = _coord_thresholds(fn, t0, cands, metric.greater_is_better, rounds=2)
    if metric.better(M.score(metric, y[B], oof[B], {"thresholds": t.tolist()}), M.score(metric, y[B], oof[B])):
        return {"thresholds": t.tolist()}
    return {}


def _fit_value_post(metric, y, oof, A, B) -> dict:
    """Регрессия: обязательная обрезка (метрика требует положительных или неотрицательных значений),
    обрезка по диапазону таргета train и, для относительных ошибок (MAPE), общий масштаб — последние два
    только если улучшают B. Для MAE/RMSE масштаб — это поправка на сдвиг уровня в проверочных периодах:
    при валидации по времени она переносится на тест плохо (flight_delays: масштаб 1.14 по OOF)."""
    y = np.asarray(y, dtype=float)
    lo, hi = float(np.min(y)), float(np.max(y))
    floor = 1e-6 if metric.positive else 0.0 if metric.nonneg else None
    base = {"clip": [floor, None]} if floor is not None else {}
    cur_b = M.score(metric, y[B], oof[B], base)
    best = base
    clip = {"clip": [lo if floor is None else max(lo, floor), hi]}
    if metric.better(M.score(metric, y[B], oof[B], clip), cur_b) and \
            not metric.better(M.score(metric, y[A], oof[A], base), M.score(metric, y[A], oof[A], clip)):
        best, cur_b = clip, M.score(metric, y[B], oof[B], clip)
    if not metric.relative:
        return best
    grid = np.round(np.arange(0.80, 1.201, 0.01), 3)
    sa = [M.score(metric, y[A], oof[A], {**best, "scale": c}) for c in grid]
    c = float(grid[int(np.argmax(sa)) if metric.greater_is_better else int(np.argmin(sa))])
    if c != 1.0:
        cand = {**best, "scale": c}
        sb = M.score(metric, y[B], oof[B], cand)
        # масштаб должен заметно улучшать B, иначе это подгонка под шум
        if metric.better(sb, cur_b) and abs(sb - cur_b) > 1e-4 * max(abs(cur_b), 1e-12):
            best = cand
    return best


def _cv(metric: M.Metric, y, oof, post, A, B) -> float:
    """Метрика на строках с OOF-предсказанием (при валидации по времени это не весь train)."""
    ab = np.concatenate([A, B])
    return M.score(metric, y[ab], oof[ab], post)


def make_single(metric: M.Metric, y: np.ndarray, r: RunResult, A: np.ndarray, B: np.ndarray) -> Solution:
    post = fit_postprocess(metric, y, r.oof, A, B)
    return Solution(name=f"exp{r.exp_id}:{r.hypothesis.spec.model}", oof=r.oof, test_pred=r.test_pred, post=post,
                    cv=_cv(metric, y, r.oof, post, A, B), holdout_b=M.score(metric, y[B], r.oof[B], post),
                    members={r.exp_id: 1.0})


# ---------------- ансамбль ----------------
def _proxy_metric(metric: M.Metric, oof: np.ndarray) -> M.Metric:
    """Гладкая замена ступенчатой метрики по меткам для подбора состава ансамбля."""
    if metric.kind != "label":
        return metric
    mode = M.DECODE.get("mode")
    if mode == "multilabel":
        return M.METRICS["ml_logloss"]
    if oof.ndim == 1:   # порядковая: близость к коду класса
        return M.METRICS["rmse"]
    return M.METRICS["logloss"]


def _standardized(runs: list[RunResult]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Ранжирование: у разных моделей оценки в разных шкалах (LambdaRank и регрессия на релевантность),
    перед смешиванием приводим к общей шкале по OOF."""
    out = []
    for r in runs:
        mu, sd = float(np.mean(r.oof)), float(np.std(r.oof) or 1.0)
        out.append(((r.oof - mu) / sd, (np.asarray(r.test_pred) - mu) / sd))
    return out


def greedy_ensemble(metric: M.Metric, y: np.ndarray, runs: list[RunResult], A: np.ndarray, B: np.ndarray,
                    max_models: int, steps: int = 25, step_gain: float = 1e-3) -> Solution | None:
    """Caruana: жадно добавляем модели (с повторами) по качеству на половине A.
    Шаг принимается, только если улучшает A хотя бы на step_gain (относительно): модель, которая
    почти ничего не добавляет, в ансамбль не попадает, и ансамбль остаётся маленьким."""
    runs = [r for r in runs if r.oof is not None][:max_models]
    if len(runs) < 2:
        return None
    final_metric = metric
    # метрика по меткам (F1, accuracy) после argmax ступенчатая и к тому же считается до постобработки:
    # смесь, которая заметно лучше после подбора весов классов, по argmax может выглядеть равной.
    # Состав подбираем по гладкой оценке (logloss, для порядковой — RMSE по кодам), а решение «ансамбль или
    # одиночная» принимает choose() по настоящей метрике после постобработки на половине B.
    # Задача 2: LightGBM+ExtraTrees по argmax 0.6551 против 0.6546, после постобработки 0.668 против 0.653.
    metric = _proxy_metric(metric, runs[0].oof)
    preds = _standardized(runs) if RANKING in final_metric.tasks else [(r.oof, r.test_pred) for r in runs]
    sign = 1 if metric.greater_is_better else -1
    counts = np.zeros(len(runs))
    blend = np.zeros_like(preds[0][0], dtype=float)
    best_score = -np.inf
    for _ in range(steps):
        cand_best, cand_i = -np.inf, -1
        n = counts.sum()
        for i, (o, _t) in enumerate(preds):
            trial = (blend * n + o) / (n + 1)
            s = sign * M.score(metric, y[A], trial[A])
            if s > cand_best:
                cand_best, cand_i = s, i
        if counts.sum() > 0 and cand_best <= best_score + step_gain * abs(best_score):
            break
        n = counts.sum()
        blend = (blend * n + preds[cand_i][0]) / (n + 1)
        counts[cand_i] += 1
        best_score = cand_best
    if (counts > 0).sum() < 2:
        return None
    w = counts / counts.sum()
    oof = sum(wi * o for wi, (o, _t) in zip(w, preds) if wi > 0)
    test = sum(wi * np.asarray(t) for wi, (_o, t) in zip(w, preds) if wi > 0)
    metric = final_metric
    post = fit_postprocess(metric, y, oof, A, B)
    members = {r.exp_id: round(float(wi), 3) for wi, r in zip(w, runs) if wi > 0}
    return Solution(name="ensemble", oof=oof, test_pred=test, post=post, cv=_cv(metric, y, oof, post, A, B),
                    holdout_b=M.score(metric, y[B], oof[B], post), members=members)


@dataclass
class Choice:
    solution: Solution
    reason: str
    gain: float = float("nan")       # относительный выигрыш ансамбля на B (>0 — ансамбль лучше)
    win_rate: float = float("nan")   # доля бутстрэп-выборок B, где ансамбль лучше


def paired_win_rate(metric: M.Metric, y: np.ndarray, a: Solution, b: Solution, idx: np.ndarray,
                    n_boot: int = 200, seed: int = 0) -> float:
    """Доля бутстрэп-выборок из idx, на которых b лучше a (парный бутстрэп по одним и тем же строкам)."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if metric.better(M.score(metric, y[s], b.oof[s], b.post), M.score(metric, y[s], a.oof[s], a.post)):
            wins += 1
    return wins / n_boot


def choose(metric: M.Metric, y: np.ndarray, single: Solution | None, ens: Solution | None, B: np.ndarray,
           min_gain: float = 0.003, confidence: float = 0.9) -> Choice | None:
    """Одиночная модель или ансамбль — решение по половине B (не участвовала в подборе весов).
    Ансамбль сложнее и дольше, поэтому берём его, только если он лучше лучшей одиночной модели
    заметно (min_gain относительно) и устойчиво (в доле confidence бутстрэп-выборок B).
    Иначе выигрыш — шум, и остаётся одна модель."""
    if single is None:
        return Choice(ens, "одиночной модели нет") if ens else None
    if ens is None:
        return Choice(single, "ансамбль не собран: ни одна модель не улучшила лучшую одиночную на половине A")
    sign = 1 if metric.greater_is_better else -1
    gain = sign * (ens.holdout_b - single.holdout_b) / max(abs(single.holdout_b), 1e-12)
    if gain < min_gain:
        return Choice(single, f"ансамбль не взят: выигрыш на B {gain:+.2%} меньше порога {min_gain:.2%}", gain)
    wr = paired_win_rate(metric, y, single, ens, B)
    if wr < confidence:
        return Choice(single, f"ансамбль не взят: выигрыш {gain:+.2%} неустойчив (лучше в {wr:.0%} бутстрэп-выборок)",
                      gain, wr)
    return Choice(ens, f"ансамбль взят: выигрыш на B {gain:+.2%}, лучше в {wr:.0%} бутстрэп-выборок", gain, wr)


# ---------------- сабмит ----------------
def _class_columns(task: TaskSpec) -> dict[int, str] | None:
    cols = [c for c in task.submission_columns if c != task.id_col]
    if len(cols) < 2:
        return None
    by_name = {str(c): c for c in cols}
    mapping = {}
    for i, cl in enumerate(task.classes):
        for cand in (str(cl), f"{task.target}_{cl}", f"class_{cl}", f"proba_{cl}"):
            if cand in by_name:
                mapping[i] = by_name[cand]
                break
    if len(mapping) == len(task.classes):
        return mapping
    if len(cols) == len(task.classes):
        return dict(enumerate(cols))
    return None


def _label(v):
    """Метка в сабмит в том же виде, что в train: 1.0 -> 1, если все классы целые."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    return v.item() if isinstance(v, np.generic) else v


def predictions_frame(task: TaskSpec, test_ids: pd.Series | None, pred: np.ndarray, post: dict,
                      n_test: int) -> pd.DataFrame:
    out = pd.DataFrame()
    post = post or {}
    if task.id_col:
        out[task.id_col] = test_ids.values if test_ids is not None else np.arange(n_test)
    target_cols = [c for c in task.submission_columns if c != task.id_col] or [task.target]
    pred = np.asarray(pred, dtype=float)
    if task.task_type == MULTILABEL:
        cols = [c for c in task.targets]
        if task.proba_output:
            vals = np.clip(pred, 0.0, 1.0)
        else:
            vals = M.labels_from_proba(pred, thresholds=post.get("thresholds"), threshold=post.get("threshold"))
        for j, c in enumerate(cols):
            out[c] = vals[:, j]
        return out
    if task.task_type in (REGRESSION, RANKING):
        out[target_cols[0]] = M.apply_value_post(pred, post) if task.task_type == REGRESSION else pred
        return out
    if task.ordinal:
        idx = M.labels_from_proba(pred, thresholds=post.get("thresholds"))
        out[target_cols[0]] = [_label(task.classes[i]) for i in idx]
        return out
    if task.proba_output:
        mapping = _class_columns(task)
        if mapping:
            for i, col in mapping.items():
                out[col] = pred[:, i]
            return out
        out[target_cols[0]] = pred[:, 1] if pred.shape[1] == 2 else pred.max(axis=1)
        return out
    w = np.asarray(post["weights"]) if post.get("weights") is not None else None
    cost = M.METRICS[task.metric].args.get("cost_matrix") if task.metric in M.METRICS else None
    idx = M.labels_from_proba(pred, w, post.get("threshold"), None, cost)
    out[target_cols[0]] = [_label(task.classes[i]) for i in idx]
    return out


def write_submission(df: pd.DataFrame, path: Path, sample: pd.DataFrame | None, task: TaskSpec) -> None:
    if sample is not None and task.id_col and task.id_col in sample.columns and len(sample) == len(df):
        # порядок строк как в sample_submission
        order = sample[[task.id_col]].merge(df, on=task.id_col, how="left")
        if order.drop(columns=[task.id_col]).notna().all().all():
            df = order
    cols = [c for c in task.submission_columns if c in df.columns] or list(df.columns)
    df = df[cols]
    if df.isna().any().any():
        num = df.select_dtypes("number").columns
        df[num] = df[num].fillna(df[num].median())
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log.info("сабмит записан: %s (%d строк)", path, len(df))


def trivial_prediction(task: TaskSpec, y: np.ndarray, n_test: int) -> np.ndarray:
    """Константное предсказание: страховка на случай, если не отработает ни одна модель."""
    if task.task_type == MULTILABEL:
        return np.tile(np.asarray(y, dtype=float).mean(axis=0), (n_test, 1))
    if task.task_type == RANKING:
        return np.zeros(n_test)
    if task.ordinal:
        return np.full(n_test, float(np.median(y)))
    if not task.is_classification:
        metric = M.METRICS.get(task.metric)
        med = metric is not None and metric.objective in ("l1", "mape", "log_l1", "log_l2")
        return np.full(n_test, float(np.median(y)) if med else float(np.mean(y)))
    prior = np.bincount(y, minlength=len(task.classes)) / len(y)
    return np.tile(prior, (n_test, 1))
