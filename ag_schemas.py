"""Контракты между модулями."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

REGRESSION = "regression"
BINARY = "binary"
MULTICLASS = "multiclass"
MULTILABEL = "multilabel"   # несколько бинарных таргетов (по колонке на метку)
RANKING = "ranking"         # оценка для сортировки внутри группы (запрос, пользователь)


@dataclass
class TaskSpec:
    task_type: str                 # regression | binary | multiclass | multilabel | ranking
    target: str
    id_col: str | None
    metric: str                    # ключ из agent.ml.metrics.METRICS
    submission_columns: list[str]  # порядок колонок сабмита
    proba_output: bool = False     # сабмит требует вероятности, а не метки
    classes: list[Any] = field(default_factory=list)  # исходные метки классов (для классификации)
    description: str = ""
    feature_descriptions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)   # multilabel: колонки-метки в порядке сабмита
    group_col: str | None = None   # ранжирование: группа (запрос, пользователь), внутри которой сортируем
    ordinal: bool = False          # классы упорядочены (QWK): учимся как регрессия на кодах, метки по порогам
    positive_class: Any = None     # бинарная: какой исходный класс — положительный (он идёт вторым в classes)
    column_types: dict[str, str] = field(default_factory=dict)  # тип колонки из data_dictionary: category | numeric | text | datetime

    @property
    def is_classification(self) -> bool:
        """Одна метка на строку и вероятности классов (бинарная, мультикласс). Порядковая задача —
        тоже классификация по постановке, но учится как регрессия (см. learns_as_regression)."""
        return self.task_type in (BINARY, MULTICLASS)

    @property
    def learns_as_regression(self) -> bool:
        """Модель предсказывает одно число на строку: регрессия, ранжирование, порядковые классы."""
        return self.task_type in (REGRESSION, RANKING) or self.ordinal

    @property
    def n_outputs(self) -> int:
        """Ширина OOF-предсказания: 1 для чисел, K для вероятностей классов, L для multilabel."""
        if self.learns_as_regression:
            return 1
        if self.task_type == MULTILABEL:
            return len(self.targets)
        return len(self.classes)

    def short(self) -> dict:
        d = asdict(self)
        d.pop("feature_descriptions")
        d.pop("column_types", None)
        d["description"] = self.description[:1500]
        d["classes"] = [str(c) for c in self.classes][:50]
        return d


@dataclass
class PipelineSpec:
    """Полное описание эксперимента. Хэшируемое представление = name + json."""
    model: str = "lgbm"
    params: dict[str, Any] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)       # имена блоков из agent.ml.features
    feature_code: str | None = None                         # make_features(df) от LLM
    drop_columns: list[str] = field(default_factory=list)   # исходные или итоговые (после блоков) признаки
    target_transform: str | None = None                     # None | "log1p"
    class_weight: str | None = None                         # None | "balanced" | "sqrt" (мягкие веса)
    tune_trials: int = 0                                    # >0: сначала Optuna-тюнинг params на первом фолде
    tune_seconds: float = 0.0                               # бюджет тюнинга (ставит оркестратор)
    gen_features: list[str] = field(default_factory=list)   # сгенерированные признаки "op|a|b" (agent/ml/features.py)
    feature_search: str | None = None                       # None | "full" | "select": сначала автопоиск признаков
    search_seconds: float = 0.0                             # бюджет автопоиска (ставит оркестратор)
    entity_mode: str | None = None                          # None | "split": отдельные модели для объектов с историей и новых
    objective: str | None = None                            # функция потерь регрессии (None — по метрике, models.py)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def key(self) -> str:
        import json
        d = self.to_dict()
        return json.dumps(d, sort_keys=True, default=str)


@dataclass
class Hypothesis:
    text: str
    spec: PipelineSpec
    parent_id: int | None = None
    source: str = "fallback"     # fallback | llm
    expected: str = ""


@dataclass
class RunResult:
    exp_id: int
    hypothesis: Hypothesis
    ok: bool
    cv_mean: float = float("nan")
    cv_std: float = float("nan")
    fold_scores: list[float] = field(default_factory=list)
    duration: float = 0.0
    error: str | None = None
    oof: np.ndarray | None = None        # (n,) регрессия | (n, K) вероятности
    test_pred: np.ndarray | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    conclusion: str = ""

    def summary(self) -> dict:
        return {
            "exp_id": self.exp_id,
            "hypothesis": self.hypothesis.text,
            "source": self.hypothesis.source,
            "parent_id": self.hypothesis.parent_id,
            "spec": self.hypothesis.spec.to_dict(),
            "ok": self.ok,
            "cv_mean": None if not self.ok else round(float(self.cv_mean), 6),
            "cv_std": None if not self.ok else round(float(self.cv_std), 6),
            "duration_s": round(self.duration, 1),
            "error": (self.error or "")[-600:] or None,
            "top_features": dict(list(self.feature_importance.items())[:10]),
            "conclusion": self.conclusion,
        }
