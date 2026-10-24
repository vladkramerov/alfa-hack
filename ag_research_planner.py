"""Планировщик гипотез: LLM с проверкой ответа и фолбэком на очередь."""
from __future__ import annotations

import copy
import json
import logging

import ag_llm_prompts as prompts
from ag_llm_client import LLMClient
from ag_ml_features import FEATURE_BLOCK_DOCS, FEATURE_BLOCKS, UnsafeCode, check_code
from ag_ml_metrics import Metric
from ag_ml_supervised import SUPERVISED_BLOCKS
from ag_ml_models import MODEL_DOCS, MODELS, NEEDS
from ag_research_fallback import FallbackQueue
from ag_research_memory import Memory
from ag_schemas import RANKING, REGRESSION, Hypothesis, PipelineSpec, TaskSpec
from ag_understand_profiler import compact

log = logging.getLogger("agent.planner")

OBJECTIVES = ("l2", "l1", "huber", "mape", "log_l1", "log_l2", "poisson", "gamma", "tweedie", "quantile")


class Planner:
    def __init__(self, task: TaskSpec, profile: dict, metric: Metric, llm: LLMClient, fallback: FallbackQueue,
                 feature_columns: list[str], warmup_fallback: int = 4, tune_trials: int = 40):
        self.task = task
        self.profile = profile
        self.metric = metric
        self.llm = llm
        self.fallback = fallback
        self.columns = feature_columns
        self.warmup = warmup_fallback   # первые N гипотез из очереди: данные и выбор семейства модели
        self.llm_rejects = 0
        self.force_fallback = False
        self.turn = 0                   # после разогрева: две гипотезы от LLM, одна из очереди
        self.tune_trials = tune_trials
        self.diagnostics: dict = {}     # разбор ошибок лучшего решения (agent/research/diagnostics.py)

    def next(self, memory: Memory, time_left: float, typical: float) -> Hypothesis | None:
        use_llm = (self.llm.available() and not self.force_fallback and len(memory.runs) > self.warmup
                   and self.llm_rejects < 6)
        if use_llm:
            self.turn += 1
            # каждая третья — из проверенной очереди (Optuna, веса классов, семейства моделей): если идеи LLM
            # слабые, сильные шаги очереди не ждут плато
            use_llm = self.turn % 3 != 0
        if use_llm:
            h = self._llm_next(memory, time_left, typical)
            if h is not None:
                return h
            self.llm_rejects += 1
        return self.fallback.next(memory, time_left, typical)

    # ------------------------------------------------------------------
    def _llm_next(self, memory: Memory, time_left: float, typical: float) -> Hypothesis | None:
        best = memory.best()
        self._time_left, self._typical = time_left, typical
        user = prompts.PLAN_TEMPLATE.format(
            task=json.dumps(self.task.short(), ensure_ascii=False),
            profile=json.dumps(compact(self.profile, 60), ensure_ascii=False),
            models=json.dumps({k: MODEL_DOCS[k] for k in self._usable_models()}, ensure_ascii=False),
            blocks=json.dumps(self._usable_blocks(), ensure_ascii=False),
            history=json.dumps(memory.prompt_summary(), ensure_ascii=False, default=str),
            advice=json.dumps(self.fallback.advisor.notes(memory) if self.fallback.advisor else {}, ensure_ascii=False)
            + "; отклонено без запуска: " + json.dumps(memory.skipped[-6:], ensure_ascii=False)
            + "; РАЗБОР ОШИБОК лучшего решения (где теряется метрика — гипотеза должна бить сюда): "
            + json.dumps(self.diagnostics, ensure_ascii=False, default=str),
            metric=self.metric.name,
            direction="больше — лучше" if self.metric.greater_is_better else "меньше — лучше",
            best_id=best.exp_id if best else None,
            time_left=int(time_left),
            typical=int(typical),
        )
        ans = self.llm.chat_json(prompts.SYSTEM_RESEARCHER, user, max_tokens=2500)
        if not ans:
            return None
        try:
            return self._parse(ans, memory)
        except (ValueError, KeyError, TypeError, UnsafeCode) as e:
            log.info("ответ LLM отклонён: %s", e)
            return None

    def _usable_models(self) -> list[str]:
        adv = self.fallback.advisor
        if adv is None:
            return list(MODELS)
        return [m for m in MODELS if m not in NEEDS or adv.check(m, None, float("inf"), 0.0)[0]]

    def _usable_blocks(self) -> dict:
        data = self.profile.get("data") or {}
        skip = set()
        if not data.get("text_columns"):
            skip.add("text_oof")
        if self.task.task_type != RANKING:
            skip.add("group_relative")
        if not (self.profile.get("structure") or {}).get("entity_key"):
            skip.add("entity_history")
        return {k: v for k, v in FEATURE_BLOCK_DOCS.items() if k not in skip}

    def _parse(self, ans: dict, memory: Memory) -> Hypothesis:
        base_id = ans.get("base_exp_id")
        base_run = memory.get(base_id if isinstance(base_id, int) else None) or memory.best()
        base = copy.deepcopy(base_run.hypothesis.spec) if base_run else PipelineSpec()
        ch = ans.get("change") or {}
        if not isinstance(ch, dict):
            raise ValueError("change должен быть объектом")
        spec = copy.deepcopy(base)
        if ch.get("model"):
            if ch["model"] not in MODELS:
                raise ValueError(f"неизвестная модель {ch['model']}")
            if ch["model"] != base.model and self.fallback.advisor is not None:
                ok, why = self.fallback.advisor.check(ch["model"], memory, self._time_left, self._typical)
                if not ok:
                    memory.skip(str(ans.get("hypothesis", ""))[:200], why)
                    raise ValueError(f"модель {ch['model']} отклонена: {why}")
            if ch["model"] != base.model:
                spec.params = {}
            spec.model = ch["model"]
        if isinstance(ch.get("params"), dict):
            spec.params = {**spec.params, **_clean_params(ch["params"])}
        if isinstance(ch.get("features"), list):
            spec.features = sorted({f for f in ch["features"] if f in FEATURE_BLOCKS or f in SUPERVISED_BLOCKS
                                    or f == "group_relative"})
        if isinstance(ch.get("drop_columns"), list):
            spec.drop_columns = [c for c in ch["drop_columns"] if c in self.columns]
        if "target_transform" in ch:
            tt = ch["target_transform"]
            spec.target_transform = tt if tt == "log1p" and self.task.task_type == REGRESSION else None
        if "class_weight" in ch:
            spec.class_weight = ch["class_weight"] if ch["class_weight"] in ("balanced", "sqrt") \
                and self.task.is_classification else None
        spec.tune_trials = self.tune_trials if ch.get("tune") is True else 0
        if "objective" in ch:
            o = ch["objective"]
            spec.objective = o if o in OBJECTIVES and self.task.task_type == REGRESSION else None
        if "entity_mode" in ch:
            spec.entity_mode = "split" if ch["entity_mode"] == "split" and self.task.task_type == REGRESSION else None
        fs = ch.get("feature_search")
        spec.feature_search = fs if fs in ("full", "select") else None
        if ch.get("gen_features") == []:
            spec.gen_features = []
        code = ch.get("feature_code")
        if code == "KEEP":
            pass
        elif isinstance(code, str) and code.strip():
            check_code(code)
            spec.feature_code = code
        elif "feature_code" in ch and code is None:
            spec.feature_code = None
        if memory.tried(spec.key()):
            raise ValueError("такой эксперимент уже был")
        return Hypothesis(text=str(ans.get("hypothesis", ""))[:500], spec=spec,
                          parent_id=base_run.exp_id if base_run else None, source="llm",
                          expected=str(ans.get("expected", ""))[:300])

    # ------------------------------------------------------------------
    def fix_code(self, code: str, error: str) -> str | None:
        if not self.llm.available():
            return None
        ans = self.llm.chat_json("Ты исправляешь код pandas. Отвечай только JSON.",
                                 prompts.FIX_TEMPLATE.format(columns=self.columns[:120], code=code, error=error[-1500:]),
                                 max_tokens=1500, temperature=0.1)
        new = (ans or {}).get("feature_code")
        if isinstance(new, str) and new.strip():
            try:
                check_code(new)
                return new
            except UnsafeCode:
                return None
        return None

    def reflect(self, hyp: Hypothesis, result: str, best: str, features: dict) -> str:
        if not self.llm.available() or hyp.source != "llm":
            return ""
        ans = self.llm.chat_json("Ты ML-исследователь. Отвечай только JSON.",
                                 prompts.REFLECT_TEMPLATE.format(hypothesis=hyp.text, result=result, best=best,
                                                                 features=json.dumps(dict(list(features.items())[:8]))),
                                 max_tokens=300, temperature=0.2)
        return str((ans or {}).get("conclusion", ""))[:400]


_PARAM_LIMITS = {
    "learning_rate": (0.005, 0.3), "num_leaves": (4, 512), "max_depth": (-1, 16), "depth": (2, 10),
    "n_estimators": (50, 20000), "iterations": (50, 20000), "min_child_samples": (1, 500),
    "subsample": (0.3, 1.0), "colsample_bytree": (0.2, 1.0), "reg_lambda": (0.0, 100.0), "reg_alpha": (0.0, 100.0),
    "l2_leaf_reg": (0.0, 100.0), "min_child_weight": (0.0, 100.0), "min_samples_leaf": (1, 200),
    "max_features": (0.05, 1.0), "alpha": (1e-4, 1e4), "C": (1e-4, 1e4),
}
_BLOCKED_PARAMS = {"n_jobs", "thread_count", "random_state", "random_seed", "verbose", "device", "gpu_id",
                   "task_type", "train_dir", "allow_writing_files", "nthread"}


def _clean_params(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        if k in _BLOCKED_PARAMS or not isinstance(v, (int, float, str, bool)):
            continue
        if k in _PARAM_LIMITS and isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = _PARAM_LIMITS[k]
            v = type(v)(min(max(v, lo), hi))
        out[k] = v
    return out
