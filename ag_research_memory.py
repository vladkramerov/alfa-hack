"""Журнал экспериментов: что пробовали, с каким результатом и какой сделан вывод.

Полные OOF/test-предсказания держим в памяти (они нужны для ансамбля),
в JSONL на диск пишем только сводку."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ag_ml_metrics import Metric
from ag_schemas import RunResult


class Memory:
    def __init__(self, metric: Metric, path: Path):
        self.metric = metric
        self.path = path
        self.runs: list[RunResult] = []
        self.seen_keys: set[str] = set()
        self.skipped: list[tuple[str, str]] = []   # гипотезы, отклонённые без запуска: (текст, причина)
        # точка опоры: лучшее решение, принятое с выигрышем больше шума (ставит оркестратор).
        # Эксперимент, который лучше лишь на шум, базой для следующих гипотез не становится.
        self.incumbent: RunResult | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def next_id(self) -> int:
        return len(self.runs)

    def add(self, r: RunResult) -> None:
        self.runs.append(r)
        self.seen_keys.add(r.hypothesis.spec.key())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r.summary(), ensure_ascii=False, default=str) + "\n")

    def skip(self, text: str, reason: str) -> None:
        import logging
        logging.getLogger("agent").info("гипотеза отклонена без запуска: %s — %s", text[:80], reason)
        self.skipped.append((text, reason))

    def tried(self, key: str) -> bool:
        return key in self.seen_keys

    def ok_runs(self) -> list[RunResult]:
        return [r for r in self.runs if r.ok and np.isfinite(r.cv_mean)]

    def ranked(self) -> list[RunResult]:
        sign = -1 if self.metric.greater_is_better else 1
        return sorted(self.ok_runs(), key=lambda r: sign * r.cv_mean)

    def best(self) -> RunResult | None:
        """Принятое лучшее решение (точка опоры); до первого принятого — лучшее по CV."""
        if self.incumbent is not None:
            return self.incumbent
        r = self.ranked()
        return r[0] if r else None

    def get(self, exp_id: int | None) -> RunResult | None:
        if exp_id is None or not (0 <= exp_id < len(self.runs)):
            return None
        return self.runs[exp_id]

    def prompt_summary(self, top_k: int = 6, last_k: int = 6) -> list[dict]:
        """Сжатая история для промпта: лучшие + последние (в т.ч. неудачные)."""
        chosen: dict[int, RunResult] = {}
        for r in self.ranked()[:top_k]:
            chosen[r.exp_id] = r
        for r in self.runs[-last_k:]:
            chosen[r.exp_id] = r
        out = []
        for r in sorted(chosen.values(), key=lambda x: x.exp_id):
            s = r.summary()
            s.pop("source", None)
            spec = s["spec"]
            if spec.get("feature_code"):
                spec["feature_code"] = spec["feature_code"][:400]
            s["top_features"] = dict(list(s["top_features"].items())[:6])
            out.append(s)
        return out
