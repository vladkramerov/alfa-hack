"""Обдуманный выбор семейства моделей: прежде чем запускать сложную модель, агент решает, есть ли
у неё шанс на этих данных и хватит ли на неё времени. Решение и причина пишутся в лог и отчёт.

Правила общие, без знаний о конкретном датасете:
* мало данных -> сложные модели (нейросеть, CatBoost с глубокими деревьями) переобучаются и не нужны;
* линейная модель почти не уступает бустингу -> зависимость простая, тяжёлые модели не нужны;
* нейросеть на таблицах редко лучше бустинга: пробуем, только если данных много и зависимость
  гладкая (линейная модель близка к бустингу), а время на неё есть;
* модель без ранней остановки (ExtraTrees, MLP) на очень больших данных идёт слишком долго."""
from __future__ import annotations

from ag_research_memory import Memory

SMALL_ROWS = 2_000          # меньше — только простые модели
MLP_MIN_ROWS = 10_000       # нейросеть имеет смысл только на больших данных
SLOW_MAX_ROWS = 300_000     # ExtraTrees / MLP дальше слишком медленные
SIMPLE_GAP = 0.01           # линейная хуже лучшей не больше чем на 1% -> зависимость простая
MLP_MAX_GAP = 0.10          # нейросеть пробуем, только если линейная хуже лучшей не больше чем на 10%


class ModelAdvisor:
    def __init__(self, profile: dict, metric):
        self.n = int(profile.get("n_train", 0))
        self.metric = metric
        self.data = profile.get("data") or {}

    def _gap(self, memory: Memory, model: str) -> float | None:
        """Насколько лучшая модель семейства model хуже лучшей вообще (относительно, >= 0)."""
        if memory is None:
            return None
        runs = [r for r in memory.ok_runs() if r.hypothesis.spec.model == model]
        best = memory.best()
        if not runs or best is None:
            return None
        sign = 1 if self.metric.greater_is_better else -1
        own = max(runs, key=lambda r: sign * r.cv_mean).cv_mean
        return max(0.0, sign * (best.cv_mean - own) / max(abs(best.cv_mean), 1e-12))

    def check(self, model: str, memory: Memory, time_left: float, typical: float) -> tuple[bool, str]:
        """Стоит ли пробовать семейство model. Возвращает (да/нет, причина)."""
        if model == "text_linear":
            return (True, "в данных есть текст") if self.data.get("text_columns") else \
                (False, "в данных нет текстовых колонок")
        if model == "lgbm_rank":
            return (True, "задача ранжирования") if self.data.get("ranking_groups") else \
                (False, "не задача ранжирования")
        if model == "cnn":
            g = self.data.get("grid")
            if not g:
                return False, "нет сетки признаков (пикселей или сигнала): свёрточной сети не на чем учиться"
            if self.n < 1000:
                return False, f"мало примеров для нейросети ({self.n})"
            return True, f"признаки образуют {'изображение' if g['kind'] == 'image' else 'сигнал'} {g['shape']}"
        if model in ("lgbm", "linear"):
            return True, "дешёвая модель"
        lin_gap = self._gap(memory, "linear")
        simple = lin_gap is not None and lin_gap <= SIMPLE_GAP
        if self.n < SMALL_ROWS and model in ("mlp", "catboost", "xgboost"):
            return False, f"мало данных ({self.n} строк < {SMALL_ROWS}): сложная модель переобучится, хватит линейной и LightGBM"
        if simple and model in ("mlp", "catboost", "xgboost", "extratrees"):
            return False, (f"линейная модель хуже лучшей всего на {lin_gap:.1%}: зависимость простая, "
                           f"сложная модель не нужна")
        if model in ("mlp", "extratrees") and self.n > SLOW_MAX_ROWS:
            return False, f"{self.n} строк: модель без ранней остановки не уложится во время"
        if model == "mlp":
            if self.n < MLP_MIN_ROWS:
                return False, f"нейросети нужно много данных ({self.n} строк < {MLP_MIN_ROWS})"
            if lin_gap is None:
                return False, "линейная модель ещё не проверена: нейросети не на что опереться"
            if lin_gap > MLP_MAX_GAP:
                return False, (f"линейная модель хуже лучшей на {lin_gap:.0%}: зависимость сильно нелинейная, "
                               f"на таблицах её лучше ловят деревья, чем нейросеть")
            if time_left < 20 * typical:
                return False, "мало времени: нейросеть обучается в разы дольше бустинга"
        return True, "есть шанс улучшить"

    def notes(self, memory: Memory) -> dict:
        """Сводка для LLM: какие семейства уже проверены и насколько они хуже лучшего."""
        out = {"n_train": self.n}
        for m in ("linear", "lgbm", "catboost", "xgboost", "extratrees", "mlp", "text_linear", "cnn", "lgbm_rank"):
            g = self._gap(memory, m)
            if g is not None:
                out[f"{m}_worse_than_best"] = f"{g:.1%}"
        return out
