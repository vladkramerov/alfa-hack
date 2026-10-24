"""Учёт времени до дедлайна."""
from __future__ import annotations

import time

from ag_config import Config

# Момент старта процесса: импорт этого модуля из solution.py происходит первым делом.
PROCESS_START = time.time()


class Budget:
    def __init__(self, cfg: Config, start: float = PROCESS_START):
        self.cfg = cfg
        self.start = start
        self.hard_deadline = start + cfg.time_limit
        self.deadline = self.hard_deadline - cfg.safety_margin()
        usable = self.deadline - start
        self.research_deadline = start + usable * cfg.research_end_fraction

    def now(self) -> float:
        return time.time()

    def elapsed(self) -> float:
        return self.now() - self.start

    def left(self) -> float:
        return max(0.0, self.deadline - self.now())

    def research_left(self) -> float:
        return max(0.0, self.research_deadline - self.now())

    def experiment_timeout(self, phase_left: float | None = None) -> float:
        left = self.research_left() if phase_left is None else phase_left
        return max(5.0, left * self.cfg.max_experiment_fraction) if left > 5 else 0.0

    def __repr__(self) -> str:
        return f"Budget(elapsed={self.elapsed():.0f}s, left={self.left():.0f}s, research_left={self.research_left():.0f}s)"
