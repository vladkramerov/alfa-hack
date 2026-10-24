"""Заранее составленная очередь гипотез. Используется, когда LLM недоступна,
вернула мусор или кончился бюджет токенов, а также в начале исследования: данные и выбор модели."""
from __future__ import annotations

import copy

import numpy as np

import ag_ml_metrics as M
from ag_ml_models import MODELS
from ag_research_advisor import ModelAdvisor
from ag_research_memory import Memory
from ag_schemas import RANKING, REGRESSION, Hypothesis, PipelineSpec, TaskSpec


MACRO_METRICS = ("macro_f1", "balanced_accuracy")
TABLE_BLIND = ("cnn",)   # модели, которые не смотрят на табличные признаки: гипотезы о признаках к ним не применяем
# Альтернативные функции потерь регрессии к той, что подобрана под метрику (Metric.objective)
OBJECTIVE_ALTERNATIVES = {"l2": ["huber"], "l1": ["l2", "huber"], "mape": ["log_l1", "l1"], "log_l1": ["mape", "l1"],
                          "log_l2": ["l2"], "poisson": ["tweedie", "l2"], "gamma": ["tweedie", "log_l2"],
                          "tweedie": ["poisson", "l2"], "quantile": ["l1"]}
OBJECTIVE_NAMES = {"l2": "квадратичная (L2)", "l1": "абсолютная (L1, медиана)", "huber": "Huber (устойчива к выбросам)",
                   "mape": "относительная ошибка (MAPE)", "log_l1": "L1 по log1p(y) (относительная ошибка)",
                   "log_l2": "L2 по log1p(y)", "poisson": "Пуассон (счётный таргет)",
                   "tweedie": "Tweedie (много нулей и тяжёлый хвост)", "gamma": "Gamma"}


def default_class_weight(task: TaskSpec, profile: dict) -> str | None:
    """Метрика, усредняющая классы поровну, при заметном дисбалансе: мягкие веса классов с самого начала
    (на задаче 2 +0.016 macro F1 к модели без весов), чтобы все сравнения моделей и признаков шли уже с ними."""
    if task.is_classification and task.metric in MACRO_METRICS and profile.get("imbalance_ratio", 1) >= 3:
        return "sqrt"
    return None


def baseline_spec(task: TaskSpec | None = None, profile: dict | None = None, ds=None) -> PipelineSpec:
    """Стартовая модель под модальность данных: ранжирование — LambdaRank, текст — бустинг с прогнозом
    линейной модели по тексту как признаком, остальное — LightGBM. Картинка тоже начинается с бустинга:
    он за минуты даёт честный сабмит, а свёрточная сеть идёт первой гипотезой со своим (большим) таймаутом
    — бейзлайн-сеть на коротком бюджете успевала 1–2 эпохи и проигрывала бустингу."""
    model = "lgbm" if "lgbm" in MODELS else next(iter(MODELS))
    cw = default_class_weight(task, profile or {}) if task is not None else None
    if task is not None and task.task_type == RANKING and "lgbm_rank" in MODELS:
        return PipelineSpec(model="lgbm_rank")
    if ds is not None and ds.text is not None:
        return PipelineSpec(model=model, class_weight=cw, features=["text_oof"])
    return PipelineSpec(model=model, class_weight=cw)


class FallbackQueue:
    def __init__(self, task: TaskSpec, profile: dict, seed: int, tune_trials: int = 40,
                 advisor: ModelAdvisor | None = None):
        self.task = task
        self.advisor = advisor
        self.tune_trials = tune_trials
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.n_core = 0           # первые n_core гипотез — обязательный этап (данные + выбор модели)
        self.static = self._static()

    def _static(self) -> list[tuple[str, PipelineSpec | None]]:
        """Сначала данные (на быстрой базовой модели), потом выбор семейства модели на готовых признаках,
        потом настройка. Каждая гипотеза улучшает одно лучшее решение; ансамбль собирается только в конце
        и только если он надёжно лучше лучшей одиночной модели (agent/finalize.py: choose)."""
        t = self.task
        q: list[tuple[str, PipelineSpec | None]] = []
        n_cat = self.profile.get("n_categorical", 0)
        data = self.profile.get("data") or {}
        grid = data.get("grid")
        # -1. модальность: модель под неё — раньше гипотез о табличных признаках
        if t.task_type == RANKING and "lgbm_rank" in MODELS:
            q.append(("Семейство моделей: LambdaRank — учится порядку внутри группы (NDCG), а не значению",
                      PipelineSpec(model="lgbm_rank")))
            q.append(("Семейство моделей: LightGBM-регрессия на релевантность", PipelineSpec(model="lgbm")))
        if data.get("text_columns"):
            q.append(("Семейство моделей: линейная модель на TF-IDF слов и символьных n-грамм текста",
                      PipelineSpec(model="text_linear")))
            q.append(("Текст для бустинга: прогноз линейной модели по TF-IDF как признак (внутри фолда)", None))
        if grid and "cnn" in MODELS:
            q.append(("Семейство моделей: свёрточная сеть на " + ("пикселях изображения" if grid["kind"] == "image"
                                                                 else "сырых сигналах"), PipelineSpec(model="cnn")))
        if grid and grid["kind"] == "image" and "cnn" in MODELS:
            # изображение: таблица пикселей для деревьев почти ничего не добавляет к свёрточной сети, время
            # лучше потратить на саму сеть; бустинг на пикселях остаётся как второе мнение для ансамбля
            q.append(("Свёрточная сеть шире: 64 канала в первом блоке", None))
            q.append(("Свёрточная сеть: дольше обучение (до 60 эпох, терпение 8)", None))
            q.append(("Свёрточная сеть: сильнее регуляризация (dropout 0.4)", None))
            q.append(("Семейство моделей: LightGBM на пикселях", PipelineSpec(model="lgbm")))
            self.n_core = len(q)
            if t.is_classification and self.profile.get("imbalance_ratio", 1) >= 3:
                q.append(("Дисбаланс классов: мягкие веса (корень из balanced) поднимут редкие классы, "
                          "почти не теряя на частых", None))
            return q
        # 0. структура данных: если у объектов есть история в прошлых периодах, это обычно главный сигнал
        st = self.profile.get("structure") or {}
        if st.get("entity_key"):
            q.append((f"История объекта ({', '.join(st['entity_key'])}): таргет того же объекта в прошлых периодах "
                      f"и изменение признаков с прошлого наблюдения", None))
            if t.task_type == REGRESSION:
                q.append(("Раздельная модель: для объектов с историей — изменение таргета относительно прошлого "
                          "значения, для новых объектов — отдельная модель без истории", None))
        # 1. данные: чистка -> удаление лишнего -> генерация и отбор признаков
        q.append(("Чистка: значения-заглушки (0 у года/площади, -999, 9999) считать пропусками", None))
        q.append(("Удаление лишних признаков: почти константы, дубликаты и сильно коррелирующие (|corr| >= 0.98)",
                  None))
        dup = self.profile.get("duplicated_rows_share", 0)
        if dup >= 0.02:
            q.append((f"Память по дублям: {dup:.0%} строк train — полные дубли других строк "
                      f"(с разным таргетом — {self.profile.get('conflicting_duplicates_share', 0):.0%}); "
                      "добавить распределение таргета у таких же строк (внутри фолда) и число повторов строки", None))
        q.append(("Память по точным значениям: распределение таргета у строк с тем же значением часто повторяющихся "
                  "числовых признаков (размеры, счётчики), внутри фолда", None))
        q.append(("Автопоиск признаков: пачками генерируем отношения, произведения, разности важных признаков и "
                  "статистики по категориям, отбираем важнее шума, убираем шумовые", None))
        if n_cat >= 1:
            q.append(("Target encoding категорий (внутри фолда, без утечки)", None))
            q.append(("Частотное кодирование категорий + число пропусков в строке", None))
        # 2. какое семейство моделей лучше подходит к этим данным (на тех же признаках, что у лучшего)
        q.append(("Семейство моделей: линейная (с обрезкой выбросов и log скошенных признаков) — "
                  "не гладкая ли зависимость", PipelineSpec(model="linear")))
        if "catboost" in MODELS:
            q.append(("Семейство моделей: CatBoost — сильнее на категориальных признаках", PipelineSpec(model="catboost")))
        if "xgboost" in MODELS:
            q.append(("Семейство моделей: XGBoost — другой способ строить деревья", PipelineSpec(model="xgboost")))
        q.append(("Семейство моделей: ExtraTrees — случайные пороги, устойчивее к шуму", PipelineSpec(model="extratrees")))
        if "mlp" in MODELS:
            q.append(("Семейство моделей: нейросеть (MLP) на нормализованных признаках", PipelineSpec(model="mlp")))
        self.n_core = len(q)
        # 2.5 функция потерь регрессии: подобранная под метрику и альтернативы
        if t.task_type == REGRESSION:
            m = M.METRICS.get(t.metric)
            cur = m.objective if m else "l2"
            alts = list({"mape": ["log_l1", "mape"]}.get(t.metric) or OBJECTIVE_ALTERNATIVES.get(cur, []))
            tgt = self.profile.get("target", {})
            if cur in ("l1", "l2") and tgt.get("nonnegative") and tgt.get("zero_share", 0) >= 0.1:
                alts.insert(0, "tweedie")
            for alt in alts:
                q.append((f"Функция потерь: {OBJECTIVE_NAMES.get(alt, alt)} вместо {OBJECTIVE_NAMES.get(cur, cur)}", None))
        if t.task_type == RANKING:
            q.append(("Признаки относительно группы: отклонение от среднего по группе и ранг внутри группы", None))
        n_ids = len(data.get("id_columns") or []) + (1 if data.get("ranking_groups") else 0)
        if n_ids >= 2 and (t.learns_as_regression or t.task_type == "binary"):
            q.append(("Скрытые факторы пар идентификаторов (матричная факторизация, как в рекомендациях)", None))
        if grid and "cnn" in MODELS:
            q.append(("Свёрточная сеть шире: 64 канала в первом блоке", None))
            q.append(("Свёрточная сеть: сильнее регуляризация (dropout 0.4)", None))
        # 3. особенности таргета
        if t.is_classification and self.profile.get("imbalance_ratio", 1) >= 3:
            q.append(("Дисбаланс классов: мягкие веса (корень из balanced) поднимут редкие классы, "
                      "почти не теряя на частых", None))  # None = модифицировать лучший
            q.append(("Дисбаланс классов: полные веса balanced", None))
        log_obj = (M.METRICS[t.metric].objective.startswith("log_") if t.metric in M.METRICS else False)
        if t.task_type == REGRESSION and self.profile.get("target", {}).get("nonnegative") and not log_obj \
                and abs(self.profile.get("target", {}).get("skew", 0)) > 1:
            q.append(("Таргет скошен и неотрицателен: обучение на log1p(y)", None))
        # 4. осмысленный подбор гиперпараметров лучшей модели (Optuna TPE) вместо ручных правок
        if self.tune_trials > 0:
            q.append(("Optuna: подбор гиперпараметров лучшей модели", None))
        # 5. второй проход по признакам уже для лучшей модели
        q.append(("Автопоиск признаков для лучшей модели: ещё один проход генерации и отбора", None))
        q.append(("Автопоиск: только отбор — убрать признаки, которые модель ценит не больше шума", None))
        q.append(("Статистики по строке для числовых признаков", None))
        if n_cat >= 2:
            q.append(("Пары категориальных признаков", None))
        if self.tune_trials > 0:
            q.append(("Optuna: повторный подбор гиперпараметров после изменения признаков", None))
        # 6. ручные правки гиперпараметров
        q.append(("Меньше learning rate и больше деревьев у лучшей модели", None))
        q.append(("Более сильная регуляризация лучшей модели", None))
        return q

    def _modify_best(self, text: str, best: PipelineSpec) -> PipelineSpec:
        s = copy.deepcopy(best)
        s.feature_code = best.feature_code
        add = lambda *blocks: sorted(set(s.features) | set(blocks))  # noqa: E731
        if text.startswith("Функция потерь"):
            rev = {v: k for k, v in OBJECTIVE_NAMES.items()}
            s.objective = rev.get(text.split(": ", 1)[1].split(" вместо ")[0])
        elif text.startswith("Признаки относительно группы"):
            s.features = add("group_relative")
        elif text.startswith("Скрытые факторы"):
            s.features = add("id_factors", "target_encoding")
        elif text.startswith("Текст для бустинга"):
            s.features = add("text_oof")
        elif text.startswith("Свёрточная сеть шире"):
            s.params = {**s.params, "width": 64}
        elif text.startswith("Свёрточная сеть: дольше"):
            s.params = {**s.params, "epochs": 60, "patience": 8}
        elif text.startswith("Свёрточная сеть: сильнее"):
            s.params = {**s.params, "dropout": 0.4}
        elif text.startswith("Дисбаланс"):
            s.class_weight = "sqrt" if "мягкие" in text else "balanced"
        elif text.startswith("Таргет скошен"):
            s.target_transform = "log1p"
        elif text.startswith("История объекта"):
            s.features = add("entity_history")
        elif text.startswith("Раздельная модель"):
            s.features = add("entity_history")
            s.entity_mode = "split"
        elif text.startswith("Память по точным значениям"):
            s.features = add("value_memory")
        elif text.startswith("Память по дублям"):
            s.features = add("row_memory", "duplicate_count")
        elif text.startswith("Чистка"):
            s.features = add("placeholders_as_missing")
        elif text.startswith("Удаление лишних"):
            s.features = add("drop_redundant")
        elif text.startswith("Автопоиск: только отбор"):
            s.feature_search = "select"
        elif text.startswith("Автопоиск"):
            s.feature_search = "full"
        elif text.startswith("Target encoding"):
            s.features = add("target_encoding")
        elif text.startswith("Optuna"):
            s.tune_trials = self.tune_trials
        elif text.startswith("Частотное"):
            s.features = add("freq_encoding", "missing_count")
        elif text.startswith("Статистики"):
            s.features = add("row_stats")
        elif text.startswith("Пары"):
            s.features = add("cat_pairs")
        elif text.startswith("Меньше learning rate"):
            lr = {"lgbm": 0.05, "xgboost": 0.05, "catboost": 0.06}.get(s.model, 0.05)
            s.params = {**s.params, "learning_rate": float(s.params.get("learning_rate", lr)) / 2}
            if s.model == "catboost":
                s.params["iterations"] = 8000
            elif s.model in ("lgbm", "xgboost"):
                s.params["n_estimators"] = 8000
        elif text.startswith("Более сильная регуляризация"):
            if s.model == "lgbm":
                s.params = {**s.params, "reg_lambda": 10.0, "min_child_samples": 50, "colsample_bytree": 0.6}
            elif s.model == "catboost":
                s.params = {**s.params, "l2_leaf_reg": 10.0}
            elif s.model == "xgboost":
                s.params = {**s.params, "reg_lambda": 10.0, "min_child_weight": 5.0, "colsample_bytree": 0.6}
        return s

    def _base_for(self, text: str, memory: Memory):
        """От какого эксперимента строить гипотезу: настройка свёрточной сети — от лучшей сети, гипотезы о
        признаках и функции потерь — от лучшей модели, которая смотрит на таблицу."""
        runs = memory.ok_runs()
        if not runs:
            return None
        sign = 1 if memory.metric.greater_is_better else -1
        pick = lambda rs: max(rs, key=lambda r: sign * r.cv_mean) if rs else None  # noqa: E731
        if text.startswith("Свёрточная сеть"):
            return pick([r for r in runs if r.hypothesis.spec.model == "cnn"])
        best = memory.best()
        if best is not None and best.hypothesis.spec.model not in TABLE_BLIND:
            return best
        return pick([r for r in runs if r.hypothesis.spec.model not in TABLE_BLIND])

    def _random_params(self, model: str) -> dict:
        r = self.rng
        if model == "lgbm":
            return {"learning_rate": float(r.choice([0.02, 0.03, 0.05])), "num_leaves": int(r.choice([15, 31, 63, 127, 255])),
                    "min_child_samples": int(r.choice([5, 10, 20, 50, 100])), "colsample_bytree": float(r.choice([0.4, 0.6, 0.8, 1.0])),
                    "subsample": float(r.choice([0.6, 0.8, 1.0])), "reg_lambda": float(r.choice([0.0, 1.0, 5.0, 20.0]))}
        if model == "catboost":
            return {"depth": int(r.choice([4, 6, 8])), "learning_rate": float(r.choice([0.03, 0.05, 0.08])),
                    "l2_leaf_reg": float(r.choice([1, 3, 10]))}
        if model == "xgboost":
            return {"max_depth": int(r.choice([4, 6, 8, 10])), "learning_rate": float(r.choice([0.02, 0.05])),
                    "min_child_weight": float(r.choice([1, 3, 10])), "colsample_bytree": float(r.choice([0.5, 0.7, 0.9]))}
        if model == "linear":
            v = float(10 ** r.uniform(-3, 2))
            return {"alpha": v, "C": v}
        if model == "mlp":
            return {"width": int(r.choice([64, 128, 256, 512])), "alpha": float(10 ** r.uniform(-6, -1))}
        if model == "extratrees":
            return {"min_samples_leaf": int(r.choice([1, 2, 4, 8])), "max_features": float(r.choice([0.3, 0.5, 0.7, 1.0]))}
        return {}

    def next(self, memory: Memory, time_left: float = float("inf"), typical: float = 0.0) -> Hypothesis | None:
        best = memory.best()
        while self.static:
            text, spec = self.static.pop(0)
            if spec is not None and self.advisor is not None:
                ok, why = self.advisor.check(spec.model, memory, time_left, typical)
                if not ok:   # модель обдумана и отклонена без запуска
                    memory.skip(text, why)
                    continue
            if spec is None:
                base = self._base_for(text, memory)
                if base is None:
                    continue
                spec = self._modify_best(text, base.hypothesis.spec)
                parent = base.exp_id
            else:
                # другое семейство моделей сравнивается на тех же данных, что у лучшего решения
                if best is not None:
                    b = best.hypothesis.spec
                    spec.features = list(b.features)
                    spec.gen_features = list(b.gen_features)
                    spec.drop_columns = list(b.drop_columns)
                    spec.feature_code = b.feature_code
                    spec.target_transform = b.target_transform
                    spec.entity_mode = b.entity_mode
                    spec.class_weight = b.class_weight
                if spec.model == "linear":   # линейной модели нужна нормальная форма признаков
                    spec.features = sorted(set(spec.features) | {"clip_outliers", "log_skewed"})
                parent = best.exp_id if best else None
            if not memory.tried(spec.key()):
                return Hypothesis(text=text, spec=spec, parent_id=parent, source="fallback")
        # дальше — случайный поиск по гиперпараметрам лучшей модели
        if best is None:
            return None
        for _ in range(20):
            s = copy.deepcopy(best.hypothesis.spec)
            s.params = self._random_params(s.model)
            if s.params and not memory.tried(s.key()):
                return Hypothesis(text=f"Случайный поиск гиперпараметров {s.model}", spec=s,
                                  parent_id=best.exp_id, source="fallback")
        return None
