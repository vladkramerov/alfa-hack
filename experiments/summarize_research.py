"""Build a compact, reproducible registry from public research run artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERIES = ("public-v1", "public-v2", "public-v3", "public-v4", "public-v5")
HISTORICAL_OUTER_KEYS = {
    "regression": "68ef03446e22982a2ce32859ef7bdf4139ff943c133f1c1de95394f92d7f66fe",
    "multiclass": "239200bc6da7d2190167f04a816e82fd7fc2c29ec36eca7f72c29ff1a19fef32",
}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def metric_direction(metric: str | None) -> str | None:
    if not metric:
        return None
    metric = metric.lower()
    return "minimize" if any(x in metric for x in ("rmse", "mae", "mse", "log_loss", "error")) else "maximize"


def score_conclusion(score: Any, control: Any, direction: str | None) -> str | None:
    if not isinstance(score, (int, float)) or not isinstance(control, (int, float)) or not direction:
        return None
    delta = float(score) - float(control)
    improved = delta < 0 if direction == "minimize" else delta > 0
    label = "improved" if improved else "regressed" if delta else "tied"
    return f"{label} vs frozen inner control; delta={delta:.9f}"


def candidate_config(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    return {k: v for k, v in candidate.items() if k not in {"hypothesis", "reason", "seed", "threads"}}


def _candidate_fit_count(candidate: Any) -> int | None:
    if not isinstance(candidate, dict):
        return None
    if candidate.get("family") == "blend":
        members = candidate.get("members")
        counts = [_candidate_fit_count(member) for member in members] if isinstance(members, list) else []
        return sum(counts) if counts and all(value is not None for value in counts) else None
    return 1


def _completed_fit_count(experiment: Any, cache_supported: bool, candidate: Any = None) -> int | None:
    if not isinstance(experiment, dict) or experiment.get("status") != "completed":
        return None
    if isinstance(experiment.get("model_fit_count"), (int, float)):
        return int(experiment["model_fit_count"])
    candidate = experiment.get("candidate", candidate)
    folds = experiment.get("fold_results")
    if isinstance(folds, list):
        counts = [_completed_fit_count(fold, cache_supported, candidate) for fold in folds]
        return sum(counts) if counts and all(value is not None for value in counts) else None
    if experiment.get("training_reused"):
        return 0
    if not cache_supported or (isinstance(experiment.get("fit_seconds"), (int, float)) and experiment["fit_seconds"] > 0):
        return _candidate_fit_count(candidate)
    return None


def score_for_candidate(audit: Any, candidate: dict[str, Any] | None, metric: str | None) -> tuple[Any, str]:
    if not isinstance(audit, list):
        return None, "missing"
    for row in audit:
        if not isinstance(row, dict):
            continue
        if candidate and row.get("candidate") == candidate and (metric is None or row.get("metric") == metric):
            status = row.get("status", "unknown")
            return (row.get("score") if status == "completed" else None), status
    statuses = [r.get("status") for r in audit if isinstance(r, dict)]
    if statuses and all(s in {"skipped", "not_run", "timeout", "failed"} for s in statuses):
        return None, "skipped/failed"
    return None, "not_found_for_inner-selected-candidate"


def frozen_control(experiments: Any, frozen: Any, metric: str | None) -> tuple[Any, dict[str, Any] | None, str | None, str]:
    candidates = frozen.get("candidates", []) if isinstance(frozen, dict) else []
    baseline = candidates[0] if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else None
    if not isinstance(experiments, list):
        return None, baseline, None, "frozen baseline score unavailable"
    if baseline is None:
        # Adaptive runs may omit frozen_selection.json; the controller records its
        # frozen baseline first as an adaptive_fallback experiment.
        baseline_row = next((item for item in experiments if isinstance(item, dict)
                             and item.get("status") == "completed"
                             and item.get("source") == "adaptive_fallback"
                             and (not metric or item.get("metric") == metric)
                             and isinstance(item.get("candidate"), dict)), None)
        if baseline_row is None:
            return None, None, None, "frozen baseline candidate unavailable"
        baseline = baseline_row["candidate"]
    target = candidate_config(baseline)
    for item in experiments:
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        if metric and item.get("metric") != metric:
            continue
        if candidate_config(item.get("candidate")) == target:
            return item.get("score"), baseline, item.get("id"), "completed frozen-baseline experiment"
    return None, baseline, None, "frozen baseline candidate has no completed matching experiment"


def novelty(candidate: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not isinstance(candidate, dict):
        return "unknown", []
    mechanisms = []
    if float(candidate.get("decision_power", 0) or 0) != 0:
        mechanisms.append("decision_power")
    if float(candidate.get("class_weight_power", 0) or 0) != 0:
        mechanisms.append("softened_class_weighting")
    if candidate.get("frequency_columns"):
        mechanisms.append("train_only_frequency_features")
    if candidate.get("expert_column"):
        mechanisms.append("group_experts")
    operations = candidate.get("feature_operations") or []
    if operations:
        mechanisms.append("auto_numeric_interactions" if any(
            isinstance(op, dict) and (op.get("operation") or op.get("op") or op.get("kind")) in {"mul", "div"}
            for op in operations
        ) else "auto_generated_numeric_features")
    if candidate.get("blend") or candidate.get("family") == "blend":
        mechanisms.append("ensemble")
    if candidate.get("drop_columns"):
        context = " ".join(str(candidate.get(k, "")) for k in ("hypothesis", "reason")).lower()
        normalized = context.replace(" ", "").replace("_", "").replace("-", "")
        if "covariateshift" in normalized:
            mechanisms.append("drift_features")
    if any(x in mechanisms for x in ("decision_power", "auto_numeric_interactions",
            "auto_generated_numeric_features", "drift_features", "group_experts")):
        return "novel_mechanism", sorted(set(mechanisms))
    if "softened_class_weighting" in mechanisms:
        return "new_weighting_strategy_configuration", sorted(set(mechanisms))
    if mechanisms:
        return "new_configuration_of_known_mechanism", sorted(set(mechanisms))
    if not mechanisms:
        return "known_reproduction_family", []
    return "known_reproduction_family", []


def task_mode_from_dir(path: Path) -> tuple[str, str]:
    name = path.name
    for task in ("regression", "multiclass"):
        prefix = task + "-"
        if name.startswith(prefix):
            return task, name[len(prefix):]
    return "unknown", name


def source_catboost_default(source_text: str) -> str | None:
    if not source_text:
        return None
    match = re.search(r"candidate\.get\(['\"]catboost_encoding['\"],\s*['\"](native|onehot)['\"]\)",
                      source_text)
    if match:
        return match.group(1)
    if "native_catboost" in source_text and "cat_features" in source_text:
        return "native"
    return None


def catboost_encodings(candidate: Any, default: str | None, path: str = "candidate") -> list[dict[str, str]]:
    if not isinstance(candidate, dict):
        return []
    if candidate.get("family") == "catboost":
        encoding = candidate.get("catboost_encoding", default)
        return ([{"path": path, "encoding": str(encoding),
                  "source_default_used": str("catboost_encoding" not in candidate).lower()}]
                if encoding is not None else [])
    result = []
    for index, member in enumerate(candidate.get("members", [])):
        result.extend(catboost_encodings(member, default, f"{path}.members[{index}]"))
    return result


def _evaluation_kind(experiment: Any) -> str:
    if not isinstance(experiment, dict):
        return "unknown"
    folds = experiment.get("fold_results")
    if isinstance(folds, list):
        reused = any(isinstance(row, dict) and row.get("training_reused") for row in folds)
        fitted = any(isinstance(row, dict) and float(row.get("fit_seconds", 0) or 0) > 0 for row in folds)
        if reused and fitted:
            return "mixed_fold_fit_and_cached_decision"
        if reused:
            return "cached_decision_reuse"
    return "cached_decision_reuse" if experiment.get("training_reused") else "model_fit"


def controller_source_details(series_dir: Path) -> dict[str, Any]:
    def read(name: str) -> str:
        try:
            return (series_dir / "source" / name).read_text(encoding="utf-8")
        except OSError:
            return ""
    engine, worker, models = (read("automl_engine.py"), read("automl_worker.py"), read("automl_models.py"))
    rolling = (("earlier_split" in engine and "rolling-mean2:" in engine)
               or ("search_split-01" in engine and "rolling-origin" in engine))
    condition = re.search(r"if nested and bundle\.task_type == 'regression'.{0,180}?budget\.seconds >= (\d+)",
                          engine, re.S)
    if condition is None:
        condition = re.search(r"if budget_seconds < (\d+)", engine)
    return {
        "probability_cache_supported_in_source": "probability_cache" in engine and "probability_cache" in worker,
        "rolling_temporal_windows_supported_in_source": rolling,
        "rolling_window_budget_threshold_seconds": int(condition.group(1)) if condition else None,
        "group_experts_supported_in_source": "GroupExpertModel" in models and "expert_column" in models,
        "catboost_default_encoding_from_source": source_catboost_default(models),
    }


def find_controller_fields(value: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(term in normalized for term in ("trial", "cost_cap", "time_budget", "budget_seconds")):
                if isinstance(child, (str, int, float, bool)) or child is None:
                    found[str(key)] = child
            found.update(find_controller_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(find_controller_fields(child))
    return found


def public_run_metadata(runs: Any) -> list[dict[str, Any]]:
    """Keep operational metadata only; exclude audit diagnostics and row-level values."""
    if not isinstance(runs, list):
        return []
    summary_fields = {
        "status", "metric", "task_type", "first_submission_seconds", "submission_rows",
        "selected_search_score", "experiments", "distinct_experiments", "llm_calls",
        "llm_status", "api_configured", "audit_status", "audit_used_for_selection",
        "independent_search_and_audit", "has_disjoint_search_and_audit", "budget_seconds",
        "seconds_remaining", "training_seconds",
        "search_window_count", "completed_model_fits", "cached_decision_experiments",
    }
    result = []
    for row in runs:
        if not isinstance(row, dict):
            continue
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
        result.append({
            "task": row.get("task"), "mode": row.get("mode"),
            "process_returncode": row.get("process_returncode"), "wall_seconds": row.get("wall_seconds"),
            "archive_sha256": row.get("archive_sha256"),
            "summary": {key: summary.get(key) for key in summary_fields if key in summary},
        })
    return result


def load_series(series: str, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    series_dir = root / "results" / series
    runs = read_json(series_dir / "runs.json") or []
    runs_by_result = {
        str(r.get("result_directory", "")): r for r in runs if isinstance(r, dict)
    }
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    missing: list[str] = []
    if not series_dir.is_dir():
        return records, comparisons, [f"{series}: directory missing"]

    task_dirs = sorted(p for p in series_dir.iterdir() if p.is_dir() and p.name not in {"source"})
    try:
        source_models = (series_dir / "source" / "automl_models.py").read_text(encoding="utf-8")
        source_default_catboost = source_catboost_default(source_models)
    except OSError:
        source_models = ""
        source_default_catboost = None
    source_details = controller_source_details(series_dir)
    seen_modes: set[tuple[str, str]] = set()
    for task_dir in task_dirs:
        task, mode = task_mode_from_dir(task_dir)
        seen_modes.add((task, mode))
        experiments = read_json(task_dir / "experiments.json")
        summary = read_json(task_dir / "summary.json")
        frozen = read_json(task_dir / "frozen_selection.json") or {}
        validation = read_json(task_dir / "validation.json") or {}
        audit_file = read_json(task_dir / "audit.json")
        audit = audit_file if isinstance(audit_file, list) else (summary or {}).get("audit", [])
        run = runs_by_result.get(str(task_dir.relative_to(root)))
        if not isinstance(experiments, list):
            missing.append(f"{series}/{task_dir.name}: experiments.json missing or invalid")
            experiments = []

        metric = (summary or {}).get("metric") or next((e.get("metric") for e in experiments if isinstance(e, dict) and e.get("metric")), None)
        inner_candidate = (summary or {}).get("search_selected") or (summary or {}).get("selected")
        inner_score = (summary or {}).get("selected_search_score")
        control_inner_score, control_candidate, control_run_id, control_status = frozen_control(experiments, frozen, metric)
        outer_score, outer_status = score_for_candidate(audit, inner_candidate, metric)
        selected_experiment = _experiment_for_candidate(experiments, inner_candidate, metric)
        control_experiment = _experiment_for_candidate(experiments, control_candidate, metric)
        aggregation = validation.get("search_aggregation", "single")
        fold_descriptions = validation.get("search_folds")
        if outer_score is None and isinstance(summary, dict) and summary.get("audit_status"):
            outer_status = f"{summary['audit_status']}; {outer_status}"
        if summary is not None and not audit:
            outer_status = "skipped/not-recorded"

        comparisons.append({
            "series": series, "task": task, "mode": mode, "metric": metric,
            "direction": metric_direction(metric),
            "audit_key": validation.get("audit_key"),
            "split": validation.get("audit"),
            "data_hashes": validation.get("data_hashes"),
            "inner_score": inner_score,
            "inner_score_aggregation": aggregation,
            "search_folds": fold_descriptions,
            "inner_selected_fold_results": _fold_results(selected_experiment),
            "inner_control_score": control_inner_score,
            "inner_control_aggregation": aggregation,
            "inner_control_fold_results": _fold_results(control_experiment),
            "inner_score_comparison_rule": "Only compare inner scores with matching aggregation and fold keys.",
            "inner_control_run_id": control_run_id,
            "inner_control_status": control_status,
            "outer_score_for_inner_selected": outer_score,
            "outer_status": outer_status,
            "selected_candidate": candidate_config(inner_candidate),
            "submission_seconds": (summary or {}).get("first_submission_seconds"),
            "run_status": (summary or {}).get("status"),
            "refs": [str((task_dir / "summary.json").relative_to(root)), str((task_dir / "validation.json").relative_to(root))],
        })

        if mode == "adaptive" and summary is None:
            missing.append(f"{series}/{task}: adaptive summary pending")
        for ordinal, item in enumerate(experiments, 1):
            if not isinstance(item, dict):
                continue
            candidate = item.get("candidate")
            novelty_candidate = dict(candidate) if isinstance(candidate, dict) else None
            if novelty_candidate is not None:
                novelty_candidate.setdefault("reason", item.get("reason"))
                novelty_candidate.setdefault("hypothesis", item.get("hypothesis"))
            kind, mechanisms = novelty(novelty_candidate)
            process = item.get("process") if isinstance(item.get("process"), dict) else {}
            run_id = item.get("id") or f"RUN-{ordinal:03d}"
            records.append({
                "id": f"{series}:{task}:{mode}:{run_id}", "series": series, "task": task, "mode": mode,
                "run_id": run_id, "candidate_key": item.get("candidate_key"),
                "hypothesis": (candidate or {}).get("hypothesis") if isinstance(candidate, dict) else None,
                "reason": ((candidate or {}).get("reason") or item.get("reason")) if isinstance(candidate, dict) else item.get("reason"),
                "config": candidate_config(candidate),
                "data_hashes": validation.get("data_hashes"),
                "split": {"search": validation.get("search"), "audit": validation.get("audit"),
                          "search_key": validation.get("search_key"), "audit_key": validation.get("audit_key"),
                          "search_folds": fold_descriptions,
                          "search_aggregation": aggregation},
                "validation_aggregation": item.get("validation_aggregation", aggregation),
                "fold_results": _fold_results(item),
                "metric": item.get("metric") or metric,
                "metric_direction": metric_direction(item.get("metric") or metric),
                "inner_control_score": control_inner_score,
                "inner_control_candidate": candidate_config(control_candidate),
                "inner_control_run_id": control_run_id,
                "inner_control_status": control_status,
                "inner_new_score": item.get("score") if item.get("status") == "completed" else None,
                "partial_fold_score": item.get("score") if item.get("status") != "completed" else None,
                "selected_candidate_outer_score": outer_score if candidate == inner_candidate else None,
                "outer_score_status": outer_status if candidate == inner_candidate else "not inner-selected",
                "seconds": item.get("seconds", process.get("seconds")),
                "fit_seconds": item.get("fit_seconds"),
                "completed_model_fits": _completed_fit_count(item, source_details["probability_cache_supported_in_source"]),
                "training_reused": bool(item.get("training_reused")),
                "evaluation_kind": _evaluation_kind(item),
                "effective_catboost_encodings": catboost_encodings(candidate, source_default_catboost),
                "peak_rss_mb": process.get("peak_rss_mb", item.get("peak_rss_mb")),
                "status": item.get("status", process.get("status", "unknown")),
                "novelty_type": kind, "novelty_mechanisms": mechanisms,
                "conclusion": ((item.get("conclusion") or score_conclusion(
                    item.get("score"), control_inner_score, metric_direction(item.get("metric") or metric)
                ) or item.get("result") or item.get("status")) if item.get("status") == "completed"
                    else f"incomplete ({item.get('status', 'unknown')}): {item.get('error', 'not compared')}; partial score is diagnostic only"),
                "refs": [str((task_dir / "experiments.json").relative_to(root)),
                         str((task_dir / "summary.json").relative_to(root)),
                         str((task_dir / "validation.json").relative_to(root))],
            })

    # runs.json is the authoritative manifest for expected tasks/modes, including pending ones.
    for run in runs:
        if not isinstance(run, dict):
            continue
        key = (str(run.get("task", "unknown")), str(run.get("mode", "unknown")))
        if key not in seen_modes:
            missing.append(f"{series}/{key[0]}-{key[1]}: artifacts pending")
    for task in ("regression", "multiclass"):
        for mode in ("baseline", "adaptive"):
            if (task, mode) not in seen_modes:
                missing.append(f"{series}/{task}-{mode}: artifacts pending")
    return records, comparisons, missing


def _experiment_for_candidate(experiments: Any, candidate: dict[str, Any] | None,
                              metric: str | None) -> dict[str, Any] | None:
    if not isinstance(experiments, list) or not isinstance(candidate, dict):
        return None
    target = candidate_config(candidate)
    for item in experiments:
        if (isinstance(item, dict) and item.get("status") == "completed"
                and (metric is None or item.get("metric") == metric)
                and candidate_config(item.get("candidate")) == target):
            return item
    return None


def _fold_results(experiment: Any) -> list[dict[str, Any]] | None:
    if not isinstance(experiment, dict):
        return None
    folds = experiment.get("fold_results")
    if not isinstance(folds, list):
        return [{"score": experiment.get("score"), "status": experiment.get("status"),
                 "fit_seconds": experiment.get("fit_seconds"),
                 "seconds": experiment.get("seconds"),
                 "completed_model_fits": _completed_fit_count(experiment, False),
                 "peak_rss_mb": (experiment.get("process") or {}).get("peak_rss_mb"),
                 "training_reused": bool(experiment.get("training_reused"))}]
    result = []
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        process = fold.get("process") if isinstance(fold.get("process"), dict) else {}
        result.append({"score": fold.get("score"), "status": fold.get("status"),
                       "fit_seconds": fold.get("fit_seconds"),
                       "seconds": fold.get("seconds", process.get("seconds")),
                       "completed_model_fits": _completed_fit_count(fold, False,
                           experiment.get("candidate")),
                       "peak_rss_mb": process.get("peak_rss_mb"),
                       "training_reused": bool(fold.get("training_reused"))})
    return result


def load_historical(root: Path) -> list[dict[str, Any]]:
    path = root / "docs" / "sources" / "sergey_branch--experiment_registry.json"
    source = read_json(path)
    if not isinstance(source, dict):
        return []
    records = []
    for row in source.get("records", []):
        if not isinstance(row, dict):
            continue
        records.append({
            "id": row.get("id"), "task": row.get("task"), "metric": row.get("metric"),
            "value": row.get("value"), "split": row.get("split"), "seed": row.get("seed"),
            "input_sha256": row.get("input_sha256"), "source_report": row.get("source_report"),
            "source": str(path.relative_to(root)),
        })
    return records


def historical_matches(comparison: dict[str, Any], historical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task, metric, key = comparison.get("task"), comparison.get("metric"), comparison.get("audit_key")
    expected = HISTORICAL_OUTER_KEYS.get(task)
    if not expected or not key or key != expected:
        return []
    hashes = comparison.get("data_hashes") or {}
    train_hash, test_hash = hashes.get("train"), hashes.get("test")
    matches = []
    for row in historical:
        input_hashes = row.get("input_sha256") or {}
        if row.get("task") != task or row.get("metric") != metric:
            continue
        if train_hash and input_hashes.get("train.csv") != train_hash:
            continue
        if test_hash and input_hashes.get("test.csv") != test_hash:
            continue
        matches.append(row)
    return matches


def best_score(rows: list[dict[str, Any]], field: str, direction: str | None) -> Any:
    values = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
    if not values:
        return None
    return min(values) if direction == "minimize" else max(values)


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, int)):
        return f"{value:.9f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(registry: dict[str, Any]) -> str:
    lines = ["# Experiment registry", "", "Generated from public-v1 through public-v5 result artifacts. Pending results remain explicitly marked.", "",
             "Outer-fold values are public and historically exposed; they are paired audit diagnostics, not an unbiased hidden-test estimate. Adaptive candidates are chosen by inner search score; this report never selects the largest outer score.", "",
             "Inner scores from a single search window are not comparable with a weighted mean across two rolling temporal windows (newest 0.65, previous 0.35). Fold descriptions, keys, and aggregation are recorded per series.", "",
             "Synthetic/API mock tests do not demonstrate a live Qwen API call. The public series used no live Qwen API.", "", "## Baseline vs adaptive: paired outer audit", "",
             "| Series | Task | Metric (direction) | Baseline inner (aggregation) | Baseline outer | Adaptive control inner (aggregation) | Adaptive selected inner (aggregation) | Adaptive selected outer | Adaptive audit status | Audit key |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in registry["comparisons"]:
        pairs.setdefault((row["series"], row["task"]), {})[row["mode"]] = row
    for (series, task), modes in sorted(pairs.items()):
        base, adaptive = modes.get("baseline", {}), modes.get("adaptive", {})
        metric = adaptive.get("metric") or base.get("metric")
        def score_with_aggregation(row: dict[str, Any], field: str) -> str:
            score = row.get(field)
            aggregation = row.get("inner_score_aggregation") or "unknown"
            folds = row.get("search_folds")
            count = len(folds) if isinstance(folds, list) else 1
            suffix = f"{aggregation}; {count} fold{'s' if count != 1 else ''}"
            return f"{fmt(score)} ({suffix})" if score is not None else "—"
        lines.append("| " + " | ".join(map(fmt, [series, task, f"{metric} ({metric_direction(metric)})" if metric else None,
            score_with_aggregation(base, "inner_score"), base.get("outer_score_for_inner_selected"),
            score_with_aggregation(adaptive, "inner_control_score"), score_with_aggregation(adaptive, "inner_score"),
            adaptive.get("outer_score_for_inner_selected"), adaptive.get("outer_status"), adaptive.get("audit_key")])) + " |")
    lines += ["", "An outer audit score is shown only for the candidate selected using inner validation. If the run timed out or did not audit that candidate, the cell remains empty and status explains why.", "", "## Historical comparability", "",
              "Historical rows are included only when the exact known outer split key and data hashes match (regression: `68ef…`; multiclass: `2392…`)."]
    seen_history: set[tuple[str, str, str]] = set()
    for row in registry["comparisons"]:
        history_key = (row.get("series"), row.get("task"), row.get("audit_key"))
        if history_key in seen_history:
            continue
        seen_history.add(history_key)
        matches = historical_matches(row, registry["historical_records"])
        if not matches:
            continue
        best = best_score(matches, "value", row.get("direction"))
        best_matches = [m for m in matches if isinstance(m.get("value"), (int, float)) and abs(m["value"] - best) <= 1e-12]
        match_ids = ", ".join("`" + str(m.get("id")) + "`" for m in best_matches)
        lines.append(f"- {row['series']} / {row['task']}: best historical {row['metric']} = `{fmt(best)}` from {match_ids}; sources `docs/sources/sergey_branch--experiment_registry.json`.")
    lines += ["", "## Candidate records", "", "| ID | Hypothesis / reason | Family | CatBoost encoding (source snapshot) | Novelty mechanism(s) | Inner score | Partial fold score (diagnostic only) | Conclusion | Seconds | Model fits / eval kind | Peak RSS MB | Status | Source |", "|---|---|---|---|---|---:|---:|---|---:|---|---:|---|---|"]
    for row in registry["records"]:
        config = row.get("config") or {}
        encoding = ", ".join(f"{e['path']}={e['encoding']}" for e in row.get("effective_catboost_encodings", []))
        folds = (row.get("split") or {}).get("search_folds")
        aggregation = row.get("validation_aggregation") or "single"
        fold_count = len(folds) if isinstance(folds, list) else 1
        score_label = (f"{fmt(row.get('inner_new_score'))} ({aggregation}; {fold_count} fold"
                       f"{'s' if fold_count != 1 else ''})" if row.get("inner_new_score") is not None else "—")
        lines.append("| " + " | ".join(map(fmt, [row.get("id"), row.get("hypothesis") or row.get("reason"), config.get("family"),
            encoding or "—", ", ".join(row.get("novelty_mechanisms", [])) or row.get("novelty_type"), score_label,
            row.get("partial_fold_score"), row.get("conclusion"), row.get("seconds"),
            f"{row.get('completed_model_fits', 0)} / {row.get('evaluation_kind', 'unknown')}",
            row.get("peak_rss_mb"), row.get("status"), ", ".join(row.get("refs", [])[:1])])) + " |")
    lines += ["", "## Novel mechanisms and distinct candidates", "", "| Mechanism | Distinct candidate records |", "|---|---:|"]
    mechanism_counts: dict[str, int] = {}
    for row in registry["records"]:
        for mechanism in row.get("novelty_mechanisms", []):
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
    for mechanism, count in sorted(mechanism_counts.items()):
        lines.append(f"| {mechanism} | {count} |")
    lines += ["", "Decision-power values such as 0.15/0.3/0.5 are configurations of one mechanism, not three mechanisms. Ensemble blending and train-only frequency features are known mechanisms from the historical registry. A softened class-weight power (for example 0.35) is a new weighting-strategy configuration relative to the historical balanced-weight baseline. Auto-generated numeric interactions and covariate-shift drop-feature proposals are separate new mechanisms; candidate rows remain distinct records. Per-value group experts with a global fallback are a separate candidate mechanism. Rolling temporal window selection is counted as an agent/controller mechanism in series metadata, not as an ML candidate hypothesis.", "", "## Runtime and controller notes", "",
              "The v1/v2 runtime comparison uses observed experiment process seconds/RSS and effective ET settings; v1's omitted max_depth resolves to the historical ExtraTrees default None (unbounded). It does not impute timings.", ""]
    if registry["runtime_comparisons"]:
        lines += ["| Series | Task/mode/run | Family | Parameters | Seconds | Peak RSS MB |", "|---|---|---|---|---:|---:|"]
        for row in registry["runtime_comparisons"]:
            task_mode_run = f"{row.get('task')}/{row.get('mode')}/{row.get('run_id')}"
            params = json.dumps(row.get("parameters"), ensure_ascii=False, sort_keys=True)
            lines.append("| " + " | ".join(map(fmt, [row.get("series"), task_mode_run, row.get("family"), params, row.get("seconds"), row.get("peak_rss_mb")])) + " |")
    else:
        lines.append("ET runtime comparison: pending ET experiment artifacts; no timing was inferred.")
    for series, data in registry["series_metadata"].items():
        visible_settings = {
            "logged": data["controller_settings"],
            "source_archive_sha256": data["source_archive_sha256"],
            "trial_configuration_from_source": data["trial_configuration_from_source"],
            "observed_trial_counts": data["observed_trial_counts"],
            "per_trial_time_cap_fraction_from_source": data["per_trial_time_cap_fraction_from_source"],
            "audit_used_for_selection_observed": data["audit_used_for_selection_observed"],
            "controller_source_details": data["controller_source_details"],
            "agent_mechanisms": data["agent_mechanisms"],
            "candidate_execution_counts": data["candidate_execution_counts"],
        }
        settings = json.dumps(visible_settings, ensure_ascii=False, sort_keys=True)
        lines.append(f"- {series}: selection policy `{data['selection_policy_label']}`; controller settings `{settings}`.")
    lines += ["", "## Completeness", "", f"Registry status: **{registry['status']}**."]
    lines.extend(f"- Pending/missing: {m}" for m in registry["missing"])
    if not registry["missing"]:
        lines.append("- No expected series artifacts are currently missing.")
    return "\n".join(lines) + "\n"


def build_registry(root: Path = ROOT, series_names: tuple[str, ...] = SERIES) -> dict[str, Any]:
    records, comparisons, missing = [], [], []
    for series in series_names:
        found, paired, pending = load_series(series, root)
        records.extend(found)
        comparisons.extend(paired)
        missing.extend(pending)
    historical = load_historical(root)
    # Preserve source-manifest hashes and run-level controller configuration when present.
    source_manifests = {s: read_json(root / "results" / s / "source-manifest.json") for s in series_names}
    run_metadata = {}
    raw_run_metadata = {}
    series_metadata = {}
    runtime_comparisons = []
    for series in series_names:
        raw_run_metadata[series] = read_json(root / "results" / series / "runs.json") or []
        run_metadata[series] = public_run_metadata(raw_run_metadata[series])
        source_details = controller_source_details(root / "results" / series)
        manifest = source_manifests.get(series)
        report = read_json(root / "results" / series / "report.json") or {}
        archive_hashes = list(dict.fromkeys(
            r.get("archive_sha256") for r in raw_run_metadata[series]
            if isinstance(r, dict) and r.get("archive_sha256")
        ))
        source_sha = (manifest.get("sha256") if isinstance(manifest, dict) else None)
        source_sha = source_sha or ((report.get("archive") or {}).get("sha256") if isinstance(report, dict) else None)
        source_sha = source_sha or (archive_hashes[0] if archive_hashes else None)
        source_text = ""
        source_path = root / "results" / series / "source" / "automl_engine.py"
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except OSError:
            pass
        trial_match = re.search(r"max_experiments\s*=\s*min\((\d+),\s*max\(0,\s*int\(os\.getenv\('AUTOML_MAX_EXPERIMENTS',\s*'?(\d+)'?", source_text)
        time_cap_match = re.search(r"budget\.seconds\s*\*\s*([0-9]*\.?[0-9]+)", source_text)
        configured_trials = None
        if trial_match:
            configured_trials = {"environment": "AUTOML_MAX_EXPERIMENTS", "default": int(trial_match.group(2)), "hard_cap": int(trial_match.group(1))}
        observed_trials = [r.get("summary", {}).get("experiments") for r in run_metadata[series]
                           if isinstance(r.get("summary"), dict)]
        selection_flags = [r["summary"].get("audit_used_for_selection") for r in run_metadata[series]
                           if isinstance(r, dict) and isinstance(r.get("summary"), dict)
                           and "audit_used_for_selection" in r["summary"]]
        if series == "public-v1":
            selection_label = "v1 audit gate"
        elif selection_flags and not any(selection_flags):
            selection_label = "audit-pure / inner-selected (reported)"
        else:
            selection_label = "selection policy not reported yet"
        if source_details["rolling_temporal_windows_supported_in_source"]:
            selection_label += ("; temporal regression uses up to two weighted rolling windows when budget >= "
                                + str(source_details["rolling_window_budget_threshold_seconds"]) + "s")
        series_records = [row for row in records if row.get("series") == series]
        distinct_keys = {row.get("candidate_key") for row in series_records if row.get("candidate_key")}
        series_metadata[series] = {
            "selection_policy_label": selection_label,
            "source_archive_sha256": source_sha,
            "source_archive_sha256_observed": archive_hashes,
            "controller_settings": find_controller_fields(run_metadata[series]),
            "controller_source_details": source_details,
            "agent_mechanisms": (([ "rolling_temporal_window_selection" ]
                                  if source_details["rolling_temporal_windows_supported_in_source"] else [])
                                 + (["group_expert_candidates"]
                                    if source_details.get("group_experts_supported_in_source") else [])),
            "catboost_effective_encodings_are_interpreted_from_each_source_snapshot": True,
            "candidate_execution_counts": {
                "distinct_candidate_keys": len(distinct_keys),
                "completed_model_fits_from_experiment_records": sum(
                    int(row.get("completed_model_fits") or 0) for row in series_records
                    if row.get("status") == "completed"),
                "cached_decision_evaluations": sum(bool(row.get("training_reused")) for row in series_records),
            },
            "trial_configuration_from_source": configured_trials,
            "observed_trial_counts": observed_trials,
            "per_trial_time_cap_fraction_from_source": float(time_cap_match.group(1)) if time_cap_match else None,
            "audit_used_for_selection_observed": selection_flags,
        }
    for row in records:
        config = row.get("config") or {}
        family = str(config.get("family", "")).lower()
        if family in {"et", "extra_trees", "extratrees", "random_forest", "rf"}:
            parameters = dict(config.get("parameters", {}))
            if row.get("series") == "public-v1" and family in {"et", "extra_trees", "extratrees"}:
                # v1's logged candidate omitted these estimator defaults; the observed slow run used them.
                v1_et_defaults = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
                                  "min_samples_split": 2, "max_features": 1.0}
                v1_et_defaults.update(parameters)
                parameters = v1_et_defaults
            runtime_comparisons.append({
                "series": row.get("series"), "task": row.get("task"), "mode": row.get("mode"),
                "run_id": row.get("run_id"), "family": family,
                "parameters": parameters, "seconds": row.get("seconds"),
                "peak_rss_mb": row.get("peak_rss_mb"), "status": row.get("status"), "refs": row.get("refs"),
            })
    for row in records + comparisons:
        row["source_archive_sha256"] = series_metadata[row["series"]]["source_archive_sha256"]
    return {
        "schema_version": 2,
        "status": "partial" if missing else "complete",
        "series": list(series_names),
        "records": records,
        "comparisons": comparisons,
        "historical_records": historical,
        "source_manifests": source_manifests,
        "run_metadata": run_metadata,
        "series_metadata": series_metadata,
        "runtime_comparisons": runtime_comparisons,
        "known_historical_outer_keys": HISTORICAL_OUTER_KEYS,
        "limitations": ["Outer public folds have historical exposure; audit is not a hidden/unbiased estimate.",
                        "Synthetic/API mock tests are not live Qwen calls; these public series did not use a live Qwen API.",
                        "No training data or labels are copied into this registry."],
        "missing": sorted(set(missing)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--series", nargs="+", choices=SERIES, default=SERIES)
    args = parser.parse_args()
    registry = build_registry(args.root.resolve(), tuple(args.series))
    json_path = args.root / "results" / "research_registry.json"
    md_path = args.root / "docs" / "EXPERIMENTS.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(registry), encoding="utf-8")
    print(json.dumps({"status": registry["status"], "records": len(registry["records"]),
                      "comparisons": len(registry["comparisons"]), "missing": registry["missing"],
                      "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
