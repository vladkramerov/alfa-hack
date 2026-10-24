"""Главный цикл агента: понимание задачи -> baseline -> AutoResearch -> финализация."""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ag_finalize as F
from ag_budget import Budget
from ag_config import Config
from ag_exec_runner import ExperimentError, ExperimentTimeout, run_with_timeout
from ag_io_contract import TaskPaths, llm_settings, resolve_task
from ag_llm_client import LLMClient
from ag_ml_models import MODELS
import ag_ml_dataset as DS, ag_ml_metrics as M, ag_ml_structure as structure
from ag_ml_pipeline import run_pipeline, warm_text_cache
import ag_research_diagnostics as D
from ag_ml_shift import adjusted_estimate, composition, test_like_folds
from ag_ml_validation import fold_groups, make_folds, split_halves
from ag_research_advisor import ModelAdvisor
from ag_research_fallback import FallbackQueue, baseline_spec, default_class_weight
from ag_research_memory import Memory
from ag_research_planner import Planner
from ag_schemas import BINARY, MULTICLASS, RANKING, REGRESSION, Hypothesis, PipelineSpec, RunResult, TaskSpec
from ag_understand_profiler import profile as make_profile
from ag_understand_task_reader import read_task

log = logging.getLogger("agent")


def _setup_logging(work_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(work_dir / "agent.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


class Agent:
    def __init__(self, paths: TaskPaths, cfg: Config, budget: Budget):
        self.paths = paths
        self.cfg = cfg
        self.budget = budget
        self.llm = LLMClient(llm_settings(), cfg)
        self.best_written: F.Solution | None = None

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        p = self.paths
        read = lambda f: pd.read_csv(f, low_memory=False)  # noqa: E731
        self.train = read(p.train)
        self.test = read(p.test)
        self.dd = read(p.data_dictionary) if p.data_dictionary else None
        self.sample = read(p.sample_submission) if p.sample_submission else None
        log.info("train %s, test %s, LLM: %s", self.train.shape, self.test.shape,
                 "есть" if self.llm.available() else self.llm.disabled_reason)

        override = self.cfg.metric_override
        if self.cfg.custom_metric_path:
            try:
                override = M.load_custom_metric(self.cfg.custom_metric_path)
                log.info("загружена метрика пользователя %s из %s", override, self.cfg.custom_metric_path)
            except Exception as e:  # noqa: BLE001 — без своей метрики работаем дальше
                log.warning("метрика пользователя не загрузилась: %s", e)
        self.task: TaskSpec = read_task(self.train, self.test, self.dd, self.sample, p.description_text, self.llm,
                                        metric_override=override)
        t = self.task
        M.configure(t)
        target_cols = t.targets or [t.target]
        self.train = self.train[self.train[target_cols].notna().all(axis=1)].reset_index(drop=True)
        self.feature_cols = DS.feature_columns(t, self.train, self.test)
        t0 = time.time()
        self.ds = DS.build(t, self.train, self.test, self.feature_cols)
        if self.ds.info:
            log.info("данные: %s (%.0f с)", json.dumps(self.ds.info, ensure_ascii=False, default=str), time.time() - t0)
        dt_cols = self.ds.datetime_cols
        y = self.ds.y
        self.structure = self._detect_structure(dt_cols)
        self.metric = M.METRICS[t.metric]

        # страховочный сабмит ещё до первой модели
        self._write(F.Solution("trivial", oof=np.empty(0), test_pred=F.trivial_prediction(t, y, len(self.test))))

        self.profile = make_profile(self.train, self.test, t, self.feature_cols, dt_cols)
        self.profile["data"] = self.ds.info
        # категории — по таблице, которую видят модели (текст ушёл в TF-IDF, id из словаря стали категориями)
        X = self.ds.X_train
        self.profile["n_categorical"] = int(sum(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns
                                                if c not in dt_cols))
        (p.work_dir / "task.json").write_text(json.dumps(t.short(), ensure_ascii=False, indent=1, default=str), "utf-8")
        (p.work_dir / "profile.json").write_text(json.dumps(self.profile, ensure_ascii=False, indent=1, default=str), "utf-8")

        self.profile["structure"] = self.structure
        self.folds = None
        if self.ds.period is not None:
            self.folds = structure.time_folds(
                self.ds.period[:len(y)], self.cfg.n_folds,
                test_share=len(self.test) / max(len(y), 1) if self.ds.entity is not None else None,
                rare=structure.rare_class_mask(y, t.is_classification))
            if len(self.folds) < 2:
                log.info("валидация по времени: мало периодов, остаюсь на случайной CV")
                self.folds = None
                self.structure["validation"] = "random"
            else:
                self.ds.refit = True
                log.info("валидация по времени: %s", [(len(a), len(b)) for a, b in self.folds])
        if self.folds is None:
            self.folds = self._group_time_folds()
        if self.folds is None:
            groups = fold_groups(t, self.ds)
            if groups is not None:
                log.info("ранжирование: группы теста новые, фолды по группам")
                self.structure["validation"] = "group"
            self.folds = make_folds(t, y, self._n_folds(), self.cfg.seed, groups)
        # валидация, похожая на тест: строки с категориями, которых нет в тесте как класса «новых», не мерим
        self.folds, shift_info = test_like_folds(self.ds.X_train, self.ds.X_test, self.folds)
        if any(shift_info["removed"]):
            log.info("валидация как тест: в тесте новых категорий %.2f%%, из проверки фолдов убраны строки "
                     "с новыми категориями %s", 100 * shift_info["test_new_share"], shift_info["removed"])
            self.structure["test_like_removed"] = shift_info["removed"]
        self.composition = composition(self.ds.entity, self.folds, len(y))
        if self.composition:
            log.info("известные объекты: в проверке %.0f%%, в тесте %.0f%%",
                     100 * self.composition["valid_seen_share"], 100 * self.composition["test_seen_share"])
            self.structure["seen_share"] = {"valid": round(self.composition["valid_seen_share"], 3),
                                            "test": round(self.composition["test_seen_share"], 3)}
        if t.task_type == RANKING:
            self._fold_groups()
        covered = np.sort(np.concatenate([va for _, va in self.folds]))
        a, b = split_halves(len(covered), y[covered], t.is_classification, self.cfg.seed,
                            y[covered, 1] if t.task_type == RANKING else None)
        self.A, self.B = covered[a], covered[b]
        self._text_warm = self.ds.text is None
        self.memory = Memory(self.metric, p.work_dir / "experiments.jsonl")
        self.fallback = FallbackQueue(t, self.profile, self.cfg.seed, self.cfg.tune_trials,
                                      advisor=ModelAdvisor(self.profile, self.metric))
        (p.work_dir / "profile.json").write_text(json.dumps(self.profile, ensure_ascii=False, indent=1, default=str), "utf-8")
        self.planner = Planner(t, self.profile, self.metric, self.llm, self.fallback, self.feature_cols,
                               warmup_fallback=self.fallback.n_core, tune_trials=self.cfg.tune_trials)
        self.stop_reason = ""

    def _fold_groups(self) -> None:
        """Ранжирование: метрика по OOF считается внутри пары (группа, фолд). Иначе в одной группе сравнивались бы
        оценки разных моделей фолдов: у LambdaRank их шкалы не согласованы, и метрика по всему OOF врёт
        (movie_ranking: по фолдам регрессия 0.924 против LambdaRank 0.920, по всему OOF наоборот)."""
        y = self.ds.y
        fold = np.full(len(y), -1)
        for i, (_, va) in enumerate(self.folds):
            fold[va] = i
        y[:, 1] = pd.factorize(pd.Series(y[:, 1].astype(np.int64)).astype(str) + "|" + pd.Series(fold).astype(str))[0]

    def _n_folds(self) -> int:
        """Тяжёлые данные (сетка для свёрточной сети, сотни тысяч строк): 3 фолда вместо 5 — каждое обучение
        дорогое, а оценке качества на таком объёме хватает и трёх."""
        if self.ds.grid is not None or len(self.ds.y) > 200_000:
            return min(3, self.cfg.n_folds)
        return self.cfg.n_folds

    def _detect_structure(self, dt_cols: list[str]) -> dict:
        """Период (test позже train) -> валидация вперёд по времени; ключ объекта -> история объекта."""
        info = {"validation": "random", "period": None, "entity_key": None}
        try:
            period = structure.detect_period(self.train, self.test, self.feature_cols, dt_cols)
            if period is None:
                return info
            per = np.concatenate([structure.period_values(self.train, period, period in dt_cols),
                                  structure.period_values(self.test, period, period in dt_cols)])
            self.ds.period = per
            info.update(validation="time", period=period)
            log.info("найден период: %s (test целиком позже train) -> валидация вперёд по времени", period)
            key = structure.detect_entity(self.train, self.test, period, self.feature_cols)
            if key:
                both = pd.concat([self.train[key], self.test[key]], ignore_index=True)
                self.ds.entity = both.astype(str).agg("|".join, axis=1).pipe(pd.factorize)[0]
                info["entity_key"] = key
        except Exception as e:  # noqa: BLE001 — структура необязательна
            log.warning("поиск структуры данных не удался: %s", e)
        return info

    def _group_time_folds(self):
        """Тест — последние наблюдения каждой группы: проверка — такой же хвост каждой группы в train."""
        try:
            gt = structure.detect_group_time(self.train, self.test, self.feature_cols, self.ds.datetime_cols,
                                             self.task.group_col)
            if gt is None:
                return None
            n = len(self.ds.y)
            codes = pd.factorize(self.train[gt["group"]])[0]
            per = structure.period_values(self.train, gt["time"], gt["time"] in self.ds.datetime_cols)
            folds = structure.group_time_folds(per[:n], codes, gt["ratio"], self.cfg.n_folds)
        except Exception as e:  # noqa: BLE001 — структура необязательна
            log.warning("поиск хвостов групп по времени не удался: %s", e)
            return None
        if len(folds) < 2:
            return None
        self.ds.refit = True
        self.structure.update(validation="group_time", group_time=gt)
        log.info("тест — последние по %s наблюдения каждой группы %s (%.0f%% строк test позже train своей "
                 "группы): проверка — такой же хвост, фолды %s", gt["time"], gt["group"], 100 * gt["share_later"],
                 [(len(a), len(b)) for a, b in folds])
        return folds

    # ------------------------------------------------------------------ experiments
    def _warm_text(self, spec) -> None:
        """Прогнозы линейной модели по тексту для фиксированных фолдов — один раз в основном процессе
        перед первым экспериментом, которому они нужны (эксперимент идёт в дочернем процессе, и его
        кэш пропадает вместе с ним)."""
        if self._text_warm or "text_oof" not in spec.features:
            return
        self._text_warm = True
        t0 = time.time()
        try:
            warm_text_cache(self.ds, self.task, self.folds, self.cfg.seed)
            log.info("прогнозы по тексту для фолдов посчитаны заранее (%.0f с)", time.time() - t0)
        except Exception as e:  # noqa: BLE001 — кэш только ускоряет
            log.warning("прогнозы по тексту заранее не посчитались: %s", e)

    def execute(self, hyp: Hypothesis, timeout: float, seed: int | None = None) -> RunResult:
        t_warm = time.time()
        self._warm_text(hyp.spec)
        if time.time() - t_warm > 1:
            # кэш текста съел часть бюджета: таймаут — от того, что осталось (сам эксперимент теперь
            # быстрее — прогнозы по тексту готовы)
            timeout = min(timeout, max(30.0, self.budget.left() * 0.5))
        exp_id = self.memory.next_id()
        t0 = time.time()
        attempts = 0
        if hyp.spec.tune_trials > 0 or hyp.spec.feature_search:
            self.memory.seen_keys.add(hyp.spec.key())   # чтобы тот же поиск не запускался повторно
            # тюнингу и автопоиску признаков — доля таймаута, но так, чтобы после них успела пройти
            # полная CV с найденным
            same = [r.duration for r in self.memory.ok_runs() if r.hypothesis.spec.model == hyp.spec.model]
            # после тюнинга модель может учиться в разы дольше (меньше learning_rate, больше листьев)
            cv_time = (4.0 if hyp.spec.tune_trials > 0 else 2.0) * (min(same) if same else timeout / 4)
            prep = max(5.0, min(timeout * self.cfg.tune_fraction, timeout - cv_time - 5))
            both = hyp.spec.tune_trials > 0 and bool(hyp.spec.feature_search)
            if hyp.spec.tune_trials > 0:
                hyp.spec.tune_seconds = round(prep / (2 if both else 1), 1)
            if hyp.spec.feature_search:
                hyp.spec.search_seconds = round(prep / (2 if both else 1), 1)
        while True:
            try:
                out = run_with_timeout(run_pipeline, (self.ds, self.task, hyp.spec, self.folds, self.cfg.n_jobs,
                                                      self.cfg.seed if seed is None else seed,
                                                      max(5.0, timeout * 0.9 - 3)), timeout)
                if out.get("tuned_params"):
                    hyp.spec.params = {**hyp.spec.params, **out["tuned_params"]}
                    hyp.text += f" → {_fmt_params(out['tuned_params'])}"
                found = out.get("found_features")
                if found:
                    hyp.spec.gen_features = hyp.spec.gen_features + found["gen_features"]
                    hyp.spec.drop_columns = hyp.spec.drop_columns + found["drop_columns"]
                    hyp.text += (f" → добавлено {len(found['gen_features'])}, удалено {len(found['drop_columns'])}"
                                 + (f" ({', '.join(found['gen_features'][:3])}{', …' if len(found['gen_features']) > 3 else ''})"
                                    if found["gen_features"] else ""))
                hyp.spec.tune_trials, hyp.spec.tune_seconds = 0, 0.0
                hyp.spec.feature_search, hyp.spec.search_seconds = None, 0.0
                r = RunResult(exp_id, hyp, True, out["cv_mean"], out["cv_std"], out["fold_scores"], time.time() - t0,
                              oof=out["oof"], test_pred=out["test_pred"], feature_importance=out["importance"])
                break
            except ExperimentTimeout as e:
                r = RunResult(exp_id, hyp, False, duration=time.time() - t0, error=str(e))
                break
            except ExperimentError as e:
                err = f"{e}\n{e.tb[-1500:]}"
                if hyp.spec.feature_code and attempts < self.cfg.max_fix_attempts and "make_features" in err:
                    fixed = self.planner.fix_code(hyp.spec.feature_code, err)
                    if fixed:
                        attempts += 1
                        hyp.spec.feature_code = fixed
                        timeout = max(5.0, timeout - (time.time() - t0))
                        log.info("код признаков исправлен, повтор %d", attempts)
                        continue
                r = RunResult(exp_id, hyp, False, duration=time.time() - t0, error=err)
                break
        best_before = self.memory.best()
        self.memory.add(r)
        if r.ok:
            log.info("exp%d [%s] %s: %s=%.5f ±%.5f за %.0f с", exp_id, hyp.source, hyp.text[:80], self.metric.name,
                     r.cv_mean, r.cv_std, r.duration)
            if best_before is not None:
                r.conclusion = self.planner.reflect(hyp, f"{r.cv_mean:.5f}", f"{best_before.cv_mean:.5f}",
                                                    r.feature_importance)
        else:
            log.info("exp%d [%s] %s: ОШИБКА %s", exp_id, hyp.source, hyp.text[:80], (r.error or "")[:300])
            r.conclusion = "эксперимент не выполнился: " + (r.error or "")[:200]
        return r

    def _is_improvement(self, r: RunResult, best: RunResult | None) -> bool:
        """Выигрыш больше порога и больше шума: парный бутстрэп по строкам OOF (одни и те же строки
        у обоих экспериментов). Разброс между фолдами здесь не годится: он отражает разную сложность
        фолдов (например, разных лет), а не шум сравнения двух моделей."""
        if not r.ok:
            return False
        if best is None or best.oof is None:
            return True
        sign = 1 if self.metric.greater_is_better else -1
        gain = sign * (r.cv_mean - best.cv_mean) / max(abs(best.cv_mean), 1e-12)
        if gain <= self.cfg.min_delta_rel:
            return False
        covered = np.concatenate([self.A, self.B])
        new, old = F.Solution("new", r.oof, r.test_pred), F.Solution("old", best.oof, best.test_pred)
        wr = F.paired_win_rate(self.metric, self.ds.y, old, new, covered, n_boot=100, seed=r.exp_id)
        if wr < self.cfg.improve_confidence:
            log.info("exp%d лучше на %.2f%%, но неустойчиво (лучше в %.0f%% бутстрэп-выборок): база остаётся exp%d",
                     r.exp_id, 100 * gain, 100 * wr, best.exp_id)
            r.conclusion = (r.conclusion + " " if r.conclusion else "") + \
                f"выигрыш {gain:+.2%} в пределах шума ({wr:.0%} бутстрэп-выборок), база не меняется"
            return False
        return True

    def _write(self, sol: F.Solution) -> None:
        df = F.predictions_frame(self.task, self.ds.test_ids if hasattr(self, "ds") else None, sol.test_pred,
                                 sol.post, len(self.test))
        F.write_submission(df, self.paths.output_path, self.sample, self.task)
        for extra in self.paths.extra_outputs:
            try:
                F.write_submission(df, extra, self.sample, self.task)
            except OSError as e:
                log.warning("не удалось записать копию %s: %s", extra, e)
        self.best_written = sol

    def _write_best_single(self) -> None:
        best = self.memory.best()
        if best is None:
            return
        sol = F.make_single(self.metric, self.ds.y, best, self.A, self.B)
        self._write(sol)
        log.info("лучший: %s, %s=%.5f (B=%.5f), постобработка=%s", sol.name, self.metric.name, sol.cv,
                 sol.holdout_b, sol.post or "нет")
        self.diagnostics = self._diagnose(sol)
        self.planner.diagnostics = self.diagnostics
        if self.diagnostics:
            log.info("разбор ошибок: %s", json.dumps(self.diagnostics, ensure_ascii=False, default=str))

    def _diagnose(self, sol: F.Solution) -> dict:
        """Где лучшее решение теряет метрику (agent/research/diagnostics.py)."""
        try:
            idx = np.concatenate([self.A, self.B])
            t = self.task
            if t.task_type in (BINARY, MULTICLASS) and not t.ordinal:
                return D.classification(self.ds.y, sol.oof, idx, list(t.classes), sol.post)
            if t.task_type == REGRESSION:
                comp = getattr(self, "composition", None)
                return D.regression(self.ds.y, sol.oof, idx, self.ds.X_train, comp["seen"] if comp else None)
            return {}
        except Exception as e:  # noqa: BLE001 — разбор необязателен
            log.info("разбор ошибок не удался: %s", e)
            return {}

    # ------------------------------------------------------------------ phases
    def run_baseline(self) -> None:
        cw = default_class_weight(self.task, self.profile)
        candidates = [baseline_spec(self.task, self.profile, self.ds), PipelineSpec(model="linear", class_weight=cw),
                      PipelineSpec(model="extratrees", class_weight=cw)]
        if (candidates[0].model != "lgbm" or candidates[0].features) and "lgbm" in MODELS:   # запасной — обычный бустинг
            candidates.insert(1, PipelineSpec(model="lgbm", class_weight=cw))
        for spec in candidates:
            hyp = Hypothesis("Baseline: модель по умолчанию без генерации признаков", spec, source="baseline")
            r = self.execute(hyp, timeout=max(30.0, self.budget.left() * 0.5))
            if r.ok:
                self.memory.incumbent = r
                self._write_best_single()
                return
        log.error("ни один baseline не отработал, остаётся константный сабмит")

    def _target_reached(self) -> bool:
        best = self.memory.best()
        target = self.cfg.target_score
        if target is None or best is None:
            return False
        return best.cv_mean >= target if self.metric.greater_is_better else best.cv_mean <= target

    def research(self) -> None:
        target_mode = self.cfg.target_score is not None
        max_iter = 10_000 if target_mode else self.cfg.max_iterations
        no_improve = 0
        it = 0
        plateaus = 0
        while it < max_iter:
            if self._target_reached():
                self.stop_reason = (f"достигнут целевой скор: {self.metric.name}={self.memory.best().cv_mean:.5f} "
                                    f"(цель {self.cfg.target_score})")
                break
            if target_mode and no_improve >= self.cfg.target_patience:
                self.stop_reason = f"{no_improve} гипотез подряд без улучшения, цель {self.cfg.target_score} не достигнута"
                break
            if no_improve >= self.cfg.patience:
                # первое плато: переключаемся на очередь/случайный поиск; второе плато: стоп
                plateaus += 1
                if plateaus >= 2:
                    self.stop_reason = "метрика вышла на плато"
                    break
                log.info("плато: переключаюсь на фолбэк-очередь и случайный поиск")
                self.planner.force_fallback = True
                no_improve = 0
            left = self.budget.research_left()
            durations = [r.duration for r in self.memory.ok_runs()
                         if "Optuna" not in r.hypothesis.text and "Автопоиск" not in r.hypothesis.text]
            typical = float(np.median(durations)) if durations else 60.0
            if left < typical * 1.1 or left < 10:
                self.stop_reason = f"время исследования закончилось (осталось {left:.0f} с, эксперимент ~{typical:.0f} с)"
                break
            hyp = self.planner.next(self.memory, left, typical)
            if hyp is None:
                self.stop_reason = "гипотезы закончились"
                break
            it += 1
            best = self.memory.best()
            factor = 8.0 if hyp.spec.tune_trials or hyp.spec.feature_search else \
                {"catboost": 6.0, "linear": 2.0, "mlp": 6.0, "cnn": 6.0, "text_linear": 2.0}.get(hyp.spec.model, 3.0)
            # таймаут от типичного эксперимента и от длительности той же модели раньше
            # (ExtraTrees или MLP заметно медленнее бустинга с ранней остановкой)
            same = [r.duration for r in self.memory.ok_runs() if r.hypothesis.spec.model == hyp.spec.model]
            base = max(typical, float(np.median(same)) if same else typical)
            # нижняя граница: автопоиску признаков нужно много быстрых обучений, а MLP и ExtraTrees
            # без ранней остановки на десятках тысяч строк идут минуты
            # Optuna и автопоиск: на длинном бюджете — заметная доля времени (на 3 часах ~7 минут), иначе
            # тюнинг успевает 3–5 проб или не успевает полную CV с найденными параметрами
            search_floor = max(150.0, min(900.0, 0.04 * left))
            floor = search_floor if hyp.spec.feature_search or hyp.spec.tune_trials else \
                {"mlp": 180.0, "extratrees": 90.0, "cnn": max(300.0, min(1800.0, 0.2 * left))}.get(hyp.spec.model, 60.0)
            timeout = min(left, max(base * factor, floor), self.budget.experiment_timeout() * 2)
            r = self.execute(hyp, timeout)
            if self._is_improvement(r, best):
                no_improve = 0
                self.memory.incumbent = r
                self._write_best_single()
            else:
                no_improve += 1
        else:
            self.stop_reason = f"достигнут лимит итераций ({max_iter})"
        log.info("исследование остановлено: %s", self.stop_reason)
        if not self.stop_reason.startswith(("время", "достигнут целевой")):
            self._polish()

    def _polish(self) -> None:
        """Очередь кончилась или метрика вышла на плато, а времени ещё много (малые и средние данные на
        3 часах): тратим его так, как потратил бы инженер, — долгий Optuna лучшей модели по всем фолдам и
        долгий Optuna лучшего запуска другого семейства бустингов (смеси он нужен сильным, а не по
        умолчанию). Принимается, как всё остальное, только устойчивое улучшение."""
        tunable = ("lgbm", "xgboost", "catboost")
        done: set[str] = set()
        for _ in range(2):
            left = self.budget.research_left()
            runs = [r for r in self.memory.ok_runs() if r.hypothesis.spec.model in tunable]
            durations = [r.duration for r in runs if r.hypothesis.spec.tune_trials == 0]
            typical = float(np.median(durations)) if durations else 60.0
            if not runs or left < max(300.0, 15 * typical):
                return
            best = self.memory.best()
            sign = 1 if self.metric.greater_is_better else -1
            pool = [best] if best.hypothesis.spec.model in tunable and best.hypothesis.spec.model not in done else []
            pool += sorted((r for r in runs if r.hypothesis.spec.model not in done | {best.hypothesis.spec.model}),
                           key=lambda r: -sign * r.cv_mean)
            if not pool:
                return
            base = pool[0]
            done.add(base.hypothesis.spec.model)
            spec = copy.deepcopy(base.hypothesis.spec)
            spec.params = {k: v for k, v in spec.params.items() if k not in ("n_estimators", "iterations")}
            spec.tune_trials = 300
            role = "лучшей модели" if base is best else f"лучшего {base.hypothesis.spec.model} (для ансамбля)"
            hyp = Hypothesis(f"Optuna на оставшемся времени: долгий подбор {role} по всем фолдам", spec,
                             parent_id=base.exp_id, source="polish")
            if spec.key() in self.memory.seen_keys:
                continue
            r = self.execute(hyp, timeout=min(left * 0.5, self.budget.experiment_timeout() * 2 + 600))
            if self._is_improvement(r, self.memory.best()):
                self.memory.incumbent = r
                self._write_best_single()

    def finalize(self) -> None:
        best = self.memory.best()
        if best is None:
            return
        single = F.make_single(self.metric, self.ds.y, best, self.A, self.B)
        bag = self._seed_bag(best, single)
        if bag is not None:
            single = bag
        ens = None
        if self.budget.left() > 5:
            ens = F.greedy_ensemble(self.metric, self.ds.y, self._ensemble_candidates(best, single), self.A,
                                    self.B, self.cfg.ensemble_max_models, step_gain=self.cfg.ensemble_step_gain)
        choice = F.choose(self.metric, self.ds.y, single, ens, self.B, self.cfg.ensemble_min_gain,
                          self.cfg.ensemble_confidence)
        log.info("финал: одиночная B=%.5f, ансамбль B=%s -> %s (%s)", single.holdout_b,
                 f"{ens.holdout_b:.5f}" if ens else "нет", choice.solution.name, choice.reason)
        self._write(choice.solution)
        self.report(choice, single, ens)
        try:   # код решения для повторного запуска без агента; на сабмит не влияет
            from ag_export import export_solution
            export_solution(self.paths.work_dir, self.task, choice.solution, self.memory, self.structure,
                            self.feature_cols, self.ds.datetime_cols, len(self.folds), self.cfg.seed)
        except Exception as e:  # noqa: BLE001
            log.warning("выгрузка кода решения не удалась: %s", e)

    def _ensemble_candidates(self, best: RunResult, single: F.Solution) -> list[RunResult]:
        """Кандидаты в ансамбль: лучшая модель (усреднённая по seed, если усреднение взято), лучший запуск
        каждого семейства (CatBoost и XGBoost чуть хуже LightGBM по CV, но ошибаются по-другому — смесь
        выигрывает именно от них), затем остальные по CV."""
        runs = self.memory.ranked()
        first = dataclasses.replace(best, oof=single.oof, test_pred=single.test_pred)   # то же, что у одиночной
        out, models = [first], {best.hypothesis.spec.model}
        for r in runs:
            if r is not best and r.hypothesis.spec.model not in models:
                out.append(r)
                models.add(r.hypothesis.spec.model)
        out += [r for r in runs if r is not best and all(r is not o for o in out)]
        return out

    def _seed_bag(self, best: RunResult, single: F.Solution) -> F.Solution | None:
        """Тот же лучший эксперимент с другими seed (те же фолды) и среднее предсказаний: дисперсия модели
        меньше, сложность решения та же. Берётся, если половина B не хуже одиночного запуска."""
        runs = [best]
        for k in (1, 2, 3, 4):
            # два seed — если успеваем; ещё два — только если времени с большим запасом
            if self.budget.left() < best.duration * (1.4 if k <= 2 else 4.0) + 30:
                break
            hyp = Hypothesis(f"Усреднение по seed: exp{best.exp_id} с seed +{k}", copy.deepcopy(best.hypothesis.spec),
                             parent_id=best.exp_id, source="bagging")
            r = self.execute(hyp, timeout=min(self.budget.left() - 20, best.duration * 2 + 30),
                             seed=self.cfg.seed + 1000 * k)
            if r.ok:
                runs.append(r)
        if len(runs) < 2:
            return None
        oof = np.mean([r.oof for r in runs], axis=0)
        test = np.mean([np.asarray(r.test_pred) for r in runs], axis=0)
        post = F.fit_postprocess(self.metric, self.ds.y, oof, self.A, self.B)
        ab = np.concatenate([self.A, self.B])
        bag = F.Solution(f"seed-bag({len(runs)}):exp{best.exp_id}", oof, test, post,
                         cv=M.score(self.metric, self.ds.y[ab], oof[ab], post),
                         holdout_b=M.score(self.metric, self.ds.y[self.B], oof[self.B], post),
                         members={r.exp_id: round(1 / len(runs), 3) for r in runs})
        better = not self.metric.better(single.holdout_b, bag.holdout_b)
        log.info("усреднение по %d seed: B=%.5f против %.5f у одного запуска -> %s", len(runs), bag.holdout_b,
                 single.holdout_b, "берём" if better else "не берём")
        return bag if better else None

    def report(self, choice: F.Choice, single: F.Solution, ens: F.Solution | None) -> None:
        chosen = choice.solution
        lines = [f"# Отчёт агента\n", f"Задача: {self.task.task_type}, таргет `{self.task.target}`, метрика {self.metric.name}",
                 f"Время: {self.budget.elapsed():.0f} с, экспериментов: {len(self.memory.runs)}, LLM: {self.llm.stats()}",
                 f"Режим: " + (f"до целевого скора {self.cfg.target_score}" if self.cfg.target_score is not None
                               else "по времени") + f"; остановка: {self.stop_reason or 'нет'}",
                 "Выбор задачи: " + "; ".join(self.task.notes),
                 f"Финал: **{chosen.name}**, {self.metric.name} OOF={chosen.cv:.5f}, B={chosen.holdout_b:.5f}, "
                 f"состав={chosen.members}, постобработка={chosen.post or 'нет'}",
                 f"Одиночная или ансамбль: лучшая одиночная {single.name} B={single.holdout_b:.5f}; ансамбль "
                 + (f"{ens.members} B={ens.holdout_b:.5f}" if ens else "не собран") + f". Решение: {choice.reason}",
                 ] + self._validation_lines(chosen) + [
                 "", "| # | источник | гипотеза | модель | CV | ±std | сек | вывод |", "|---|---|---|---|---|---|---|---|"]
        diag = getattr(self, "diagnostics", None)
        if diag:
            lines[-2:-2] = ["Разбор ошибок лучшей одиночной модели: `"
                            + json.dumps(diag, ensure_ascii=False, default=str) + "`", ""]
        tail = [f"- {text} — {why}" for text, why in self.memory.skipped]
        for r in self.memory.runs:
            cv = f"{r.cv_mean:.5f}" if r.ok else "ошибка"
            sd = f"{r.cv_std:.5f}" if r.ok else ""
            lines.append(f"| {r.exp_id} | {r.hypothesis.source} | {r.hypothesis.text[:90]} | {r.hypothesis.spec.model} | "
                         f"{cv} | {sd} | {r.duration:.0f} | {(r.conclusion or '')[:120]} |")
        if tail:
            lines += ["", "Отклонено без запуска (модель обдумана заранее):"] + tail
        (self.paths.work_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


    def _validation_lines(self, chosen: F.Solution) -> list[str]:
        """Как устроена валидация и насколько её состав похож на тест."""
        st = self.structure
        out = [f"Валидация: {st.get('validation')}" + (f" по `{st['period']}`" if st.get("period") else "")
               + (f", ключ объекта {st['entity_key']}" if st.get("entity_key") else "")]
        if st.get("test_like_removed"):
            out.append(f"Валидация как тест: в тесте нет новых категорий, поэтому из проверки фолдов убраны строки "
                       f"с категориями, которых не было в обучении фолда: {st['test_like_removed']} по фолдам")
        if st.get("validation") == "time" and chosen.oof is not None and len(chosen.oof) and len(self.folds) > 1:
            va = self.folds[-1][1]
            last = M.score(self.metric, self.ds.y[va], chosen.oof[va], chosen.post)
            out.append(f"Последний фолд (обучение на всей истории до него — ближе всего к тесту): "
                       f"{self.metric.name}={last:.5f}; ранние фолды учатся на меньшей истории и обычно хуже")
        comp = getattr(self, "composition", None)
        if comp and chosen.oof is not None and len(chosen.oof):
            idx = np.concatenate([self.A, self.B])
            est = adjusted_estimate(self.metric.name, self.ds.y, chosen.oof, idx, comp["seen"],
                                    comp["test_seen_share"])
            line = (f"Известные объекты (ключ уже встречался): в проверке {comp['valid_seen_share']:.0%}, "
                    f"в тесте {comp['test_seen_share']:.0%}")
            if est is not None:
                line += f"; оценка {self.metric.name} для теста с поправкой на эту долю: **{est:.5f}**"
            out.append(line)
        return out


def _fmt_params(p: dict) -> str:
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def run(argv: list[str] | None = None) -> int:
    cfg = Config()
    budget = Budget(cfg)
    paths = resolve_task(argv)
    _setup_logging(paths.work_dir)
    log.info("условие: %s; данные: %s; выход: %s (%s)", paths.task_md, paths.data_dir, paths.output_path,
             paths.output_source)
    log.info("старт: лимит %.0f с, запас %.0f с, n_jobs=%d, режим: %s", cfg.time_limit, cfg.safety_margin(),
             cfg.n_jobs, f"до скора {cfg.target_score}" if cfg.target_score is not None else "по времени")
    agent = Agent(paths, cfg, budget)
    try:
        agent.load()
        agent.run_baseline()
        agent.research()
        agent.finalize()
    except Exception:  # noqa: BLE001 — сабмит уже на диске, падать нельзя
        log.exception("агент упал; на диске остался последний записанный сабмит")
        if agent.best_written is None and hasattr(agent, "memory"):
            agent._write_best_single()
    log.info("готово за %.0f с; %s", budget.elapsed(), budget)
    return 0
