"""Реестр метрик. Метрику всегда считает этот код, а не сгенерированный.

Соглашения о предсказаниях и таргете (y — то, что лежит в Dataset.y):
  регрессия      — y (n,), предсказание (n,)
  классификация  — y коды 0..K-1, предсказание — матрица вероятностей (n, K)
  порядковая     — y коды 0..K-1, предсказание — число (n,) на шкале кодов; метки — по порогам
  multilabel     — y (n, L) из 0/1, предсказание — вероятности (n, L)
  ранжирование   — y (n, 2): релевантность и код группы, предсказание — оценка (n,)
Для метрик по меткам (F1, accuracy) метки получаются из вероятностей: argmax(proba * веса классов),
порог для бинарной, пороги для порядковой, минимум ожидаемой стоимости для матрицы стоимости.

Метрики с параметром (NDCG@k, квантиль, степень Tweedie, beta у F-beta, матрица стоимости)
регистрируются по требованию (`ensure`) под именем вида `ndcg@10` в родительском процессе до fork,
поэтому видны в экспериментах.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn import metrics as skm

from ag_schemas import BINARY, MULTICLASS, MULTILABEL, RANKING, REGRESSION


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool
    kind: str                 # value | label | proba
    tasks: tuple[str, ...]
    objective: str = "l2"     # функция потерь регрессии, которая ближе всего к метрике (models.py)
    positive: bool = False    # предсказание должно быть > 0 (Poisson, Gamma, Tweedie)
    nonneg: bool = False      # предсказание должно быть >= 0 (RMSLE)
    relative: bool = False    # ошибка относительно y (MAPE): оптимум смещён от медианы, общий масштаб законен
    args: dict = field(default_factory=dict, hash=False, compare=False)

    def better(self, a: float, b: float) -> bool:
        """a лучше b?"""
        if np.isnan(b):
            return not np.isnan(a)
        return a > b if self.greater_is_better else a < b


# ---------------------------------------------------------------- регрессия
def _f(a):
    return np.asarray(a, dtype=float)


def _rmse(y, p):
    return float(np.sqrt(np.mean((_f(y) - _f(p)) ** 2)))


def _rmsle(y, p):
    p = np.clip(_f(p), 0, None)
    return float(np.sqrt(np.mean((np.log1p(np.clip(_f(y), 0, None)) - np.log1p(p)) ** 2)))


def _mape(y, p):
    y = _f(y)
    return float(np.mean(np.abs((y - _f(p)) / np.where(np.abs(y) < 1e-9, 1e-9, np.abs(y)))))


def _smape(y, p):
    y, p = _f(y), _f(p)
    d = np.abs(y) + np.abs(p)
    return float(np.mean(np.where(d < 1e-12, 0.0, 2 * np.abs(y - p) / np.where(d < 1e-12, 1, d))))


def _wape(y, p):
    y = _f(y)
    return float(np.sum(np.abs(y - _f(p))) / max(np.sum(np.abs(y)), 1e-12))


def _rmspe(y, p):
    y = _f(y)
    ok = np.abs(y) > 1e-9
    return float(np.sqrt(np.mean(((y[ok] - _f(p)[ok]) / y[ok]) ** 2))) if ok.any() else 0.0


def _poisson(y, p):
    return float(skm.mean_poisson_deviance(_f(y), np.clip(_f(p), 1e-9, None)))


def _gamma(y, p):
    return float(skm.mean_gamma_deviance(np.clip(_f(y), 1e-9, None), np.clip(_f(p), 1e-9, None)))


def _tweedie(power):
    return lambda y, p: float(skm.mean_tweedie_deviance(_f(y), np.clip(_f(p), 1e-9, None), power=power))


def _pinball(alpha):
    return lambda y, p: float(skm.mean_pinball_loss(_f(y), _f(p), alpha=alpha))


# ---------------------------------------------------------------- классификация
def _proba_pos(proba):
    proba = np.asarray(proba)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba


def _auc(y, proba):
    proba = np.asarray(proba)
    if proba.ndim == 1 or proba.shape[1] == 2:
        return float(skm.roc_auc_score(y, _proba_pos(proba)))
    present = np.unique(y)
    p = proba[:, present]
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    if len(present) == 2:
        return float(skm.roc_auc_score(y == present[1], p[:, 1]))
    return float(skm.roc_auc_score(np.searchsorted(present, y), p, multi_class="ovr", average="macro"))


def _logloss(y, proba):
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    proba = np.clip(proba, 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(proba[np.arange(len(y)), np.asarray(y, dtype=int)])))


def _brier(y, proba):
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1 or proba.shape[1] == 2:
        return float(np.mean(((np.asarray(y) == 1).astype(float) - _proba_pos(proba)) ** 2))
    onehot = np.eye(proba.shape[1])[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((onehot - proba) ** 2, axis=1)))


def _fbeta(beta):
    return lambda y, l: float(skm.fbeta_score(y, l, beta=beta, average="binary", zero_division=0))


def _cost(matrix: np.ndarray):
    """Средняя стоимость: matrix[истинный класс, предсказанный класс]."""
    def fn(y, l):
        return float(np.mean(matrix[np.asarray(y, dtype=int), np.asarray(l, dtype=int)]))
    return fn


# ---------------------------------------------------------------- multilabel
def _ml_cols(Y, P, fn):
    Y, P = np.asarray(Y), np.asarray(P, dtype=float)
    vals = [fn(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean(vals)) if vals else 0.5


def _ml_auc(Y, P):
    return _ml_cols(Y, P, lambda y, p: skm.roc_auc_score(y, p))


def _ml_ap(Y, P):
    return _ml_cols(Y, P, lambda y, p: skm.average_precision_score(y, p))


def _ml_logloss(Y, P):
    Y, P = np.asarray(Y, dtype=float), np.clip(np.asarray(P, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


# ---------------------------------------------------------------- ранжирование
def _group_positions(groups: np.ndarray, *keys) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Порядок строк: по группе, внутри — по ключам; начала групп и позиция строки в своей группе."""
    order = np.lexsort(tuple(reversed(keys)) + (groups,)) if keys else np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(g)])
    pos = np.arange(len(g)) - np.repeat(starts, sizes)
    return order, starts, pos


def _ndcg(k: int, exp_gain: bool = False):
    def fn(y2, score):
        y2 = np.asarray(y2, dtype=float)
        rel, groups, s = y2[:, 0], y2[:, 1], np.asarray(score, dtype=float)
        gain = (lambda r: np.power(2.0, r) - 1) if exp_gain else (lambda r: r)
        # при равных оценках — худший порядок (иначе константа получает бонус за удачную сортировку)
        order, starts, pos = _group_positions(groups, -s, rel)
        disc = np.where(pos < k, 1.0 / np.log2(pos + 2.0), 0.0)
        dcg = np.add.reduceat(gain(rel[order]) * disc, starts)
        order_i, _, _ = _group_positions(groups, -rel)
        idcg = np.add.reduceat(gain(rel[order_i]) * disc, starts)
        ok = idcg > 0
        return float(np.mean(dcg[ok] / idcg[ok])) if ok.any() else 0.0
    return fn


def _map_at(k: int):
    def fn(y2, score):
        y2 = np.asarray(y2, dtype=float)
        rel, groups, s = (y2[:, 0] > 0).astype(float), y2[:, 1], np.asarray(score, dtype=float)
        order, starts, pos = _group_positions(groups, -s, rel)
        r = rel[order] * (pos < k)
        sizes = np.diff(np.r_[starts, len(r)])
        csum = np.cumsum(r)
        before = np.repeat(np.r_[0, csum[starts[1:] - 1]], sizes)
        hits = csum - before
        prec = np.where(r > 0, hits / (pos + 1.0), 0.0)
        n_rel = np.minimum(np.add.reduceat(rel[order], starts), k)
        ap = np.add.reduceat(prec, starts) / np.maximum(n_rel, 1)
        ok = n_rel > 0
        return float(np.mean(ap[ok])) if ok.any() else 0.0
    return fn


def _mrr(y2, score):
    y2 = np.asarray(y2, dtype=float)
    rel, groups, s = (y2[:, 0] > 0), y2[:, 1], np.asarray(score, dtype=float)
    order, starts, pos = _group_positions(groups, -s, rel.astype(float))
    first = np.where(rel[order], pos + 1.0, np.inf)
    best = np.minimum.reduceat(first, starts)
    ok = np.isfinite(best)
    return float(np.mean(1.0 / best[ok])) if ok.any() else 0.0


_CLS = (BINARY, MULTICLASS)
_REG = (REGRESSION,)
METRICS: dict[str, Metric] = {
    "rmse": Metric("rmse", _rmse, False, "value", _REG),
    "mse": Metric("mse", lambda y, p: _rmse(y, p) ** 2, False, "value", _REG),
    "mae": Metric("mae", lambda y, p: float(np.mean(np.abs(_f(y) - _f(p)))), False, "value", _REG, objective="l1"),
    "medae": Metric("medae", lambda y, p: float(np.median(np.abs(_f(y) - _f(p)))), False, "value", _REG,
                    objective="l1"),
    "rmsle": Metric("rmsle", _rmsle, False, "value", _REG, objective="log_l2", nonneg=True),
    "msle": Metric("msle", lambda y, p: _rmsle(y, p) ** 2, False, "value", _REG, objective="log_l2", nonneg=True),
    # MAPE: L2 по log1p(y) сходится быстро и близка к оптимуму относительной ошибки; objective mape у
    # LightGBM (L1 с весами 1/y) на diamonds за 120 с не сошлась (0.083 против 0.063 у log1p)
    "mape": Metric("mape", _mape, False, "value", _REG, objective="log_l2", relative=True),
    "smape": Metric("smape", _smape, False, "value", _REG, objective="log_l1", relative=True),
    "wape": Metric("wape", _wape, False, "value", _REG, objective="l1"),
    "rmspe": Metric("rmspe", _rmspe, False, "value", _REG, objective="log_l2", relative=True),
    "r2": Metric("r2", lambda y, p: float(skm.r2_score(y, p)), True, "value", _REG),
    "poisson": Metric("poisson", _poisson, False, "value", _REG, objective="poisson", positive=True),
    "gamma": Metric("gamma", _gamma, False, "value", _REG, objective="gamma", positive=True),
    "macro_f1": Metric("macro_f1", lambda y, l: float(skm.f1_score(y, l, average="macro")), True, "label", _CLS),
    "weighted_f1": Metric("weighted_f1", lambda y, l: float(skm.f1_score(y, l, average="weighted")), True, "label", _CLS),
    "micro_f1": Metric("micro_f1", lambda y, l: float(skm.f1_score(y, l, average="micro")), True, "label", _CLS),
    "f1": Metric("f1", lambda y, l: float(skm.f1_score(y, l, average="binary", zero_division=0)), True, "label",
                 (BINARY,)),
    "precision": Metric("precision", lambda y, l: float(skm.precision_score(y, l, zero_division=0)), True, "label",
                        (BINARY,)),
    "recall": Metric("recall", lambda y, l: float(skm.recall_score(y, l, zero_division=0)), True, "label", (BINARY,)),
    "accuracy": Metric("accuracy", lambda y, l: float(skm.accuracy_score(y, l)), True, "label", _CLS),
    "balanced_accuracy": Metric("balanced_accuracy", lambda y, l: float(skm.balanced_accuracy_score(y, l)), True,
                                "label", _CLS),
    "mcc": Metric("mcc", lambda y, l: float(skm.matthews_corrcoef(y, l)), True, "label", _CLS),
    "kappa": Metric("kappa", lambda y, l: float(skm.cohen_kappa_score(y, l)), True, "label", _CLS),
    "qwk": Metric("qwk", lambda y, l: float(skm.cohen_kappa_score(y, l, weights="quadratic")), True, "label", _CLS),
    "roc_auc": Metric("roc_auc", _auc, True, "proba", _CLS),
    "gini": Metric("gini", lambda y, p: 2 * _auc(y, p) - 1, True, "proba", _CLS),
    "pr_auc": Metric("pr_auc", lambda y, p: float(skm.average_precision_score(y, _proba_pos(p))), True, "proba",
                     (BINARY,)),
    "logloss": Metric("logloss", _logloss, False, "proba", _CLS),
    "brier": Metric("brier", _brier, False, "proba", _CLS),
    "ml_auc": Metric("ml_auc", _ml_auc, True, "proba", (MULTILABEL,)),
    "ml_ap": Metric("ml_ap", _ml_ap, True, "proba", (MULTILABEL,)),
    "ml_logloss": Metric("ml_logloss", _ml_logloss, False, "proba", (MULTILABEL,)),
    "ml_micro_f1": Metric("ml_micro_f1", lambda Y, L: float(skm.f1_score(Y, L, average="micro", zero_division=0)),
                          True, "label", (MULTILABEL,)),
    "ml_macro_f1": Metric("ml_macro_f1", lambda Y, L: float(skm.f1_score(Y, L, average="macro", zero_division=0)),
                          True, "label", (MULTILABEL,)),
    "ml_samples_f1": Metric("ml_samples_f1", lambda Y, L: float(skm.f1_score(Y, L, average="samples",
                                                                             zero_division=0)),
                            True, "label", (MULTILABEL,)),
    "ml_hamming": Metric("ml_hamming", lambda Y, L: float(np.mean(np.asarray(Y) != np.asarray(L))), False, "label",
                         (MULTILABEL,)),
    "mrr": Metric("mrr", _mrr, True, "value", (RANKING,)),
}

METRIC_DOCS = {
    "rmse": "ошибка в единицах таргета, сильно штрафует большие промахи",
    "mse": "квадрат RMSE",
    "mae": "средняя абсолютная ошибка, устойчива к выбросам (оптимум — медиана)",
    "medae": "медиана абсолютной ошибки",
    "rmsle": "ошибка в логарифмах, для скошенного неотрицательного таргета, важна относительная ошибка",
    "msle": "квадрат RMSLE",
    "mape": "средняя относительная ошибка в долях",
    "smape": "симметричная относительная ошибка",
    "wape": "сумма абсолютных ошибок / сумма таргета (прогноз спроса)",
    "rmspe": "корень из средней квадратичной относительной ошибки",
    "r2": "доля объяснённой дисперсии",
    "poisson": "Poisson deviance: счётный таргет (число событий), предсказание > 0",
    "gamma": "Gamma deviance: положительный скошенный таргет (суммы), предсказание > 0",
    "macro_f1": "F1, усреднённый по классам поровну: важны редкие классы",
    "weighted_f1": "F1, взвешенный по частоте классов",
    "micro_f1": "F1 по всем объектам (= accuracy для одной метки)",
    "f1": "F1 положительного класса (бинарная)",
    "precision": "точность положительного класса",
    "recall": "полнота положительного класса",
    "accuracy": "доля верных ответов, для сбалансированных классов",
    "balanced_accuracy": "средняя полнота по классам, для несбалансированных",
    "mcc": "коэффициент корреляции Мэтьюса",
    "kappa": "каппа Коэна",
    "qwk": "квадратично взвешенная каппа: порядковые классы, штраф растёт с расстоянием",
    "roc_auc": "качество ранжирования вероятностей, не зависит от порога",
    "gini": "коэффициент Джини = 2*AUC - 1 (кредитный скоринг)",
    "pr_auc": "average precision: ранжирование при редком положительном классе",
    "logloss": "качество вероятностей (калибровка)",
    "brier": "средний квадрат ошибки вероятностей",
    "ml_auc": "multilabel: средний ROC-AUC по меткам",
    "ml_ap": "multilabel: средний average precision по меткам",
    "ml_logloss": "multilabel: средний logloss по меткам",
    "ml_micro_f1": "multilabel: micro F1",
    "ml_macro_f1": "multilabel: macro F1 по меткам",
    "ml_samples_f1": "multilabel: F1 по объектам",
    "ml_hamming": "multilabel: доля неверных меток",
    "mrr": "ранжирование: средний обратный ранг первого релевантного",
}

# Для своей метрики и метрик с параметром: исходные метки классов (заполняет оркестратор после чтения
# задачи), чтобы пользовательская функция получала настоящие метки, а не коды 0..K-1.
CUSTOM_CLASSES: list = []
# Как из предсказания получаются метки (configure): "argmax" — классы по вероятностям, "ordinal" — число на
# шкале кодов -> код по порогам (k классов), "multilabel" — порог у каждой метки.
DECODE: dict = {"mode": "argmax", "k": 0}


def configure(task) -> None:
    """Глобальные настройки под задачу; вызывать в родительском процессе до запуска экспериментов (fork)."""
    CUSTOM_CLASSES[:] = list(task.classes) if task.task_type in (BINARY, MULTICLASS) else []
    DECODE.clear()
    if task.task_type == MULTILABEL:
        DECODE.update(mode="multilabel", k=len(task.targets))
    elif getattr(task, "ordinal", False):
        DECODE.update(mode="ordinal", k=len(task.classes))
    else:
        DECODE.update(mode="argmax", k=len(task.classes))
    ensure(task.metric)


def ensure(name: str) -> str | None:
    """Регистрирует метрику с параметром по имени (`ndcg@10`, `map@5`, `tweedie@1.5`, `pinball@0.9`,
    `fbeta@2`) и возвращает имя; для обычных метрик — имя, если она есть."""
    if name in METRICS:
        return name
    m = re.fullmatch(r"(ndcg|ndcg2|map|tweedie|pinball|fbeta)@([\d.]+)", name or "")
    if not m:
        return None
    base, v = m.group(1), float(m.group(2))
    if base in ("ndcg", "ndcg2", "map"):
        k = max(1, int(v))
        fn = _ndcg(k, exp_gain=base == "ndcg2") if base != "map" else _map_at(k)
        METRICS[name] = Metric(name, fn, True, "value", (RANKING,), args={"k": k})
        METRIC_DOCS[name] = f"ранжирование: {'NDCG' if base != 'map' else 'MAP'}@{k} внутри групп"
    elif base == "tweedie":
        METRICS[name] = Metric(name, _tweedie(v), False, "value", _REG, objective="tweedie", positive=True,
                               args={"power": v})
        METRIC_DOCS[name] = f"Tweedie deviance, степень {v}"
    elif base == "pinball":
        if not 0 < v < 1:
            return None
        METRICS[name] = Metric(name, _pinball(v), False, "value", _REG, objective="quantile", args={"alpha": v})
        METRIC_DOCS[name] = f"квантильная ошибка (pinball), квантиль {v}"
    else:
        METRICS[name] = Metric(name, _fbeta(v), True, "label", (BINARY,), args={"beta": v})
        METRIC_DOCS[name] = f"F-beta положительного класса, beta={v}"
    return name


def register_cost(matrix: list[list[float]], name: str = "cost") -> str:
    """Своя метрика стоимости: matrix[i][j] — стоимость ответа j при истинном классе i (коды классов)."""
    C = np.asarray(matrix, dtype=float)
    METRICS[name] = Metric(name, _cost(C), False, "label", _CLS, args={"cost_matrix": C.tolist()})
    METRIC_DOCS[name] = "средняя стоимость ошибок по матрице стоимости из условия"
    return name


def load_custom_metric(path: str) -> str:
    """Регистрирует метрику пользователя из .py-файла и возвращает её имя.

    Файл определяет функцию `metric(y_true, y_pred) -> float` и, по желанию:
      NAME = "my_metric"
      GREATER_IS_BETTER = True
      KIND = "label" | "proba" | "value"   # что передаётся в y_pred
    label — метки классов (те же значения, что в таргете); proba — для бинарной вектор вероятности
    второго класса, для многоклассовой матрица (n, K) в порядке отсортированных классов;
    value — числа (регрессия). y_true — исходные значения таргета.
    Код пишет сам пользователь стенда, поэтому он исполняется как обычный Python."""
    ns: dict = {}
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, path, "exec"), ns)  # noqa: S102 — метрика пользователя, не код от LLM
    fn = ns.get("metric") or ns.get("score")
    if not callable(fn):
        raise ValueError(f"{path}: нужна функция metric(y_true, y_pred)")
    name = str(ns.get("NAME", "custom")).strip() or "custom"
    kind = str(ns.get("KIND", "label")).lower()
    if kind not in ("label", "proba", "value"):
        raise ValueError(f"{path}: KIND должен быть label, proba или value")
    gib = bool(ns.get("GREATER_IS_BETTER", True))
    return register_function(name, fn, gib, kind)


def register_function(name: str, fn: Callable, greater_is_better: bool, kind: str,
                      tasks: tuple = (REGRESSION, BINARY, MULTICLASS)) -> str:
    """Метрика из готовой функции на исходных метках (пользователь стенда или код, проверенный агентом)."""
    def decode(v):
        return np.asarray(CUSTOM_CLASSES, dtype=object)[np.asarray(v, dtype=int)] if CUSTOM_CLASSES else v

    def wrapped(y, pred):
        if kind == "value":
            return float(fn(np.asarray(y), np.asarray(pred)))
        yt = decode(y)
        if kind == "label":
            return float(fn(yt, decode(pred)))
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] == 2:
            pred = pred[:, 1]
        return float(fn(yt, pred))

    METRICS[name] = Metric(name, wrapped, greater_is_better, kind, tasks)
    METRIC_DOCS[name] = "метрика пользователя"
    return name


# ---------------------------------------------------------------- разбор названия метрики из текста
# (шаблон, имя или функция от совпадения -> имя). Порядок важен: более специфичные раньше.
_K = r"\s*@\s*(\d+)"
_ALIASES: list[tuple[str, object]] = [
    (r"ndcg" + _K, lambda m: f"ndcg@{m.group(1)}"),
    (r"\bndcg\b", "ndcg@10"),
    (r"\bmap" + _K, lambda m: f"map@{m.group(1)}"),
    (r"mean\s+average\s+precision", "map@10"),
    (r"\bmrr\b|mean\s+reciprocal\s+rank|обратн\w*\s+ранг", "mrr"),
    (r"\brmsle\b|root\s+mean\s+squared\s+log", "rmsle"),
    (r"\bmsle\b|mean\s+squared\s+log", "msle"),
    (r"\brmspe\b", "rmspe"),
    (r"\bs-?mape\b|symmetric\s+mean\s+absolute\s+percentage|симметричн\w*\s+(?:средн\w*\s+)?относительн", "smape"),
    (r"\bwape\b|\bwmape\b|weighted\s+absolute\s+percentage|взвешенн\w*\s+абсолютн\w*\s+процентн", "wape"),
    (r"\bmape\b|mean\s+absolute\s+percentage|средн\w*\s+абсолютн\w*\s+процентн|средн\w*\s+относительн\w*\s+ошибк",
     "mape"),
    (r"tweedie[^\d\n]{0,40}?(1\.\d+)", lambda m: f"tweedie@{m.group(1)}"),
    (r"tweedie|твиди", "tweedie@1.5"),
    (r"poisson|пуассон", "poisson"),
    (r"gamma\s+deviance|гамма[\s-]+девианс", "gamma"),
    (r"(?:pinball|quantile\s+loss|квантильн\w*\s+(?:ошибк|потер|функци))[^\d\n]{0,40}?(0\.\d+)",
     lambda m: f"pinball@{m.group(1)}"),
    (r"pinball|quantile\s+loss|квантильн\w*\s+(?:ошибк|потер)", "pinball@0.5"),
    (r"\brmse\b|root\s+mean\s+squared\s+error|корень\s+из\s+среднеквадратичн|корень\s+из\s+средн\w*\s+квадрат", "rmse"),
    (r"\bmse\b|mean\s+squared\s+error|среднеквадратичн|средн\w*\s+квадрат\w*\s+ошибк", "mse"),
    (r"median\s+absolute\s+error|медиан\w*\s+абсолютн", "medae"),
    (r"\bmae\b|mean\s+absolute\s+error|средн\w*\s+абсолютн", "mae"),
    (r"\br2\b|r\^2|r²|коэффициент\s+детерминации", "r2"),
    (r"quadratic(?:ally)?\s+weighted\s+kappa|\bqwk\b|квадратичн\w*\s+(?:взвешенн\w*\s+)?капп|weights\s*=\s*['\"]quadratic",
     "qwk"),
    (r"cohen\w*\s+kappa|капп\w*\s+коэн|\bkappa\b|\bкаппа\b", "kappa"),
    (r"matthews|\bmcc\b|мэтьюс", "mcc"),
    (r"macro[\s_-]*f1|f1[\s_-]*macro|макро[\s-]*f1|average\s*=\s*['\"]macro", "macro_f1"),
    (r"weighted[\s_-]*f1|f1[\s_-]*weighted|взвешенн\w*\s+f1|average\s*=\s*['\"]weighted", "weighted_f1"),
    (r"micro[\s_-]*f1|f1[\s_-]*micro|average\s*=\s*['\"]micro", "micro_f1"),
    (r"pr[\s_-]*auc|average\s+precision|auc[\s_-]*pr|precision[\s-]+recall\s+auc|точность[\s-]+полнота", "pr_auc"),
    (r"gini|джини", "gini"),
    (r"roc[\s_-]*auc|\bauc\b|roc_auc", "roc_auc"),
    (r"log[\s_-]*loss|cross[\s-]*entropy|логлосс|логарифмическ\w*\s+(?:потер|функци)|кросс[\s-]*энтроп", "logloss"),
    (r"brier|брайер", "brier"),
    (r"balanced[\s_]+accuracy|сбалансированн\w*\s+точност", "balanced_accuracy"),
    (r"\bf[\s_-]*beta\b[^\d\n]{0,20}?(\d+(?:\.\d+)?)|\bf(\d)[\s-]*(?:score|мер)", lambda m: f"fbeta@{m.group(1) or m.group(2)}"
     if (m.group(1) or m.group(2)) not in ("1", "1.0") else "f1"),
    (r"\bf1\b|f-?мер", "f1"),
    (r"\bprecision\b", "precision"),
    (r"\brecall\b", "recall"),
    (r"accuracy|точност\w*\s+классификац|доля\s+(?:верн|правильн)", "accuracy"),
]

# Метрика по меткам на multilabel и по меткам/вероятностям — соответствие имён
_ML_MAP = {"roc_auc": "ml_auc", "gini": "ml_auc", "pr_auc": "ml_ap", "logloss": "ml_logloss",
           "macro_f1": "ml_macro_f1", "micro_f1": "ml_micro_f1", "f1": "ml_micro_f1", "accuracy": "ml_hamming",
           "weighted_f1": "ml_macro_f1"}


def find_metrics(text: str) -> list[tuple[int, str]]:
    """Все упоминания метрик в тексте: (позиция, имя) по возрастанию позиции."""
    out = []
    for pat, name in _ALIASES:
        for m in re.finditer(pat, text, re.I):
            n = name(m) if callable(name) else name
            if n and ensure(n):
                out.append((m.start(), n))
    # одно место текста — одна метрика (самый специфичный шаблон, он идёт раньше в списке)
    seen, res = set(), []
    for pos, n in sorted(out, key=lambda t: t[0]):
        if any(abs(pos - p) < 3 for p in seen):
            continue
        seen.add(pos)
        res.append((pos, n))
    return res


def adapt(name: str, task_type: str | None) -> str | None:
    """Метрика, приведённая к типу задачи (F1 мультикласса -> macro F1, AUC multilabel -> средний AUC)."""
    if task_type is None:
        return name
    if task_type == MULTICLASS and name == "f1":
        name = "macro_f1"
    if task_type == MULTILABEL and name in _ML_MAP:
        name = _ML_MAP[name]
    m = METRICS.get(name)
    return name if m and task_type in m.tasks else None


def parse_metric(text: str, task_type: str | None = None) -> str | None:
    """Первая метрика, упомянутая как основная. Ищем сначала строки со словами
    «метрика»/«metric», потом весь текст."""
    if not text:
        return None
    lines = [ln for ln in text.splitlines() if re.search(r"метрик|metric|оценива|score|качеств", ln, re.I)]
    main = [ln for ln in lines if re.search(r"основн|главн|main|primary|целев|итогов", ln, re.I)]
    for chunk in ("\n".join(main), "\n".join(lines), text):
        for _pos, name in find_metrics(chunk):
            n = adapt(name, task_type)
            if n:
                return n
    return None


def default_metric(task_type: str) -> str:
    return {"regression": "rmse", "binary": "roc_auc", "multiclass": "macro_f1", "multilabel": "ml_auc",
            "ranking": ensure("ndcg@10")}[task_type]


# ---------------------------------------------------------------- метки из предсказаний и подсчёт
def labels_from_proba(proba: np.ndarray, weights: np.ndarray | None = None, threshold: float | None = None,
                      thresholds: list | None = None, cost_matrix=None) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.ndim == 1:
        if DECODE.get("mode") == "ordinal" or thresholds is not None:
            k = int(DECODE.get("k") or (len(thresholds) + 1 if thresholds is not None else 2))
            cuts = np.asarray(thresholds if thresholds is not None else np.arange(k - 1) + 0.5, dtype=float)
            return np.searchsorted(np.sort(cuts), proba, side="right").clip(0, k - 1)
        proba = np.column_stack([1 - proba, proba])
    if DECODE.get("mode") == "multilabel":
        t = np.asarray(thresholds if thresholds is not None else [threshold or 0.5] * proba.shape[1], dtype=float)
        return (proba >= t[None, :]).astype(int)
    if threshold is not None and proba.shape[1] == 2:   # подобранный порог главнее правила стоимости
        return (proba[:, 1] >= threshold).astype(int)
    if cost_matrix is not None:
        P = proba * (weights[None, :] if weights is not None else 1.0)
        return np.argmin(P @ np.asarray(cost_matrix, dtype=float), axis=1)
    if weights is not None:
        proba = proba * weights[None, :]
    return proba.argmax(axis=1)


def score(metric: Metric, y: np.ndarray, pred: np.ndarray, post: dict | None = None) -> float:
    """post — параметры постобработки: {'weights': [...]}, {'threshold': t}, {'thresholds': [...]},
    {'scale': c} (регрессия), {'clip': [lo, hi]}."""
    post = post or {}
    if metric.kind == "label":
        w = np.asarray(post["weights"]) if post.get("weights") is not None else None
        cm = metric.args.get("cost_matrix")
        return metric.fn(y, labels_from_proba(pred, w, post.get("threshold"), post.get("thresholds"), cm))
    if metric.kind == "value" and (post.get("scale") is not None or post.get("clip") is not None):
        pred = apply_value_post(pred, post)
    return metric.fn(y, pred)


def apply_value_post(pred: np.ndarray, post: dict) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if post.get("scale") is not None:
        pred = pred * float(post["scale"])
    if post.get("clip") is not None:
        lo, hi = post["clip"]
        pred = np.clip(pred, lo, hi)
    return pred
