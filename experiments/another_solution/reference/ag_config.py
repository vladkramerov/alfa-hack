"""Настройки агента. Всё, что зависит от времени, задаётся долями от TIME_LIMIT,
чтобы один и тот же код работал и 3 часа на платформе, и 5 минут на локальном стенде."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_opt_float(name: str) -> float | None:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        return None


def detect_cpus() -> int:
    """Число доступных CPU с учётом квоты cgroup (в Docker os.cpu_count() видит все ядра хоста)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


@dataclass
class Config:
    # Общий лимит на задачу (сек). Платформа даёт 3 часа.
    time_limit: float = field(default_factory=lambda: _env_float("TIME_LIMIT", 3 * 3600))
    # Запас до принудительной остановки: доля лимита, но не больше 15 минут и не меньше 20 с.
    safety_fraction: float = 0.08
    safety_max: float = 900.0
    safety_min: float = 20.0
    # Доля полезного времени (после запаса), после которой цикл гипотез останавливается
    # и начинается финализация.
    research_end_fraction: float = 0.90

    n_jobs: int = field(default_factory=lambda: _env_int("N_JOBS", detect_cpus()))
    seed: int = 42
    n_folds: int = 5

    # Улучшение принимается (эксперимент становится новой точкой опоры), если оно больше min_delta_rel
    # относительно И устойчиво: в парном бутстрэпе по строкам OOF новый эксперимент лучше в доле
    # improve_confidence выборок. Иначе выигрыш — шум, и следующие гипотезы строятся от прежнего лучшего.
    min_delta_rel: float = 5e-4
    improve_confidence: float = field(default_factory=lambda: _env_float("IMPROVE_CONFIDENCE", 0.8))

    # Цикл AutoResearch
    max_iterations: int = field(default_factory=lambda: _env_int("MAX_ITERATIONS", 60))
    patience: int = 12                   # итераций подряд без улучшения
    max_fix_attempts: int = 2            # починок сгенерированного кода фичей
    max_experiment_fraction: float = 0.35  # один эксперимент не дольше этой доли оставшегося времени

    # Режим «до результата»: исследование останавливается, как только CV лучшей модели достигает
    # TARGET_SCORE, или после TARGET_PATIENCE гипотез подряд без улучшения. TIME_LIMIT остаётся потолком.
    target_score: float | None = field(default_factory=lambda: _env_opt_float("TARGET_SCORE"))
    target_patience: int = field(default_factory=lambda: _env_int("TARGET_PATIENCE", 10))

    # Метрика: METRIC — принудительно одна из встроенных; CUSTOM_METRIC — путь к .py со своей метрикой
    # (см. agent/ml/metrics.py: load_custom_metric). Без них метрика выбирается по тексту задачи и данным.
    metric_override: str | None = field(default_factory=lambda: os.environ.get("METRIC") or None)
    custom_metric_path: str | None = field(default_factory=lambda: os.environ.get("CUSTOM_METRIC") or None)

    # Optuna-тюнинг лучшей модели: число проб и доля таймаута эксперимента на него
    tune_trials: int = field(default_factory=lambda: _env_int("TUNE_TRIALS", 40))
    tune_fraction: float = 0.6

    # LLM
    llm_enabled: bool = field(default_factory=lambda: os.environ.get("LLM_DISABLED", "0") != "1")
    llm_max_tokens_total: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 400_000))
    llm_timeout: float = 120.0
    llm_temperature: float = 0.4
    llm_no_think: bool = field(default_factory=lambda: os.environ.get("LLM_NO_THINK", "1") == "1")
    # доля лимита времени, которую можно потратить на ожидание ответов LLM: если модель рассуждает и отвечает
    # по минуте, без потолка планирование съело бы час из трёх, а модели обучались бы меньше
    llm_time_share: float = field(default_factory=lambda: _env_float("LLM_TIME_SHARE", 0.2))

    # Финализация. Ансамбль берётся вместо лучшей одиночной модели, только если на половине B
    # он лучше хотя бы на ensemble_min_gain (относительно) и выигрывает в ensemble_confidence
    # бутстрэп-выборок B. Модель добавляется в ансамбль, если улучшает A хотя бы на ensemble_step_gain.
    # Веса подбираются на A, а B честная, поэтому порог — про устойчивость, а не про размер выигрыша:
    # смесь бустингов обычно даёт +0.1–0.3%, и порог 0.3% отсекал её почти всегда (flight_delays +0.17%).
    ensemble_max_models: int = 8
    ensemble_min_gain: float = field(default_factory=lambda: _env_float("ENSEMBLE_MIN_GAIN", 0.0005))
    ensemble_confidence: float = 0.8
    ensemble_step_gain: float = 2e-4

    def safety_margin(self) -> float:
        return min(self.safety_max, max(self.safety_min, self.time_limit * self.safety_fraction))
