"""Independently verify saved experiments; never fit models or import the runner.

Usage: python -m experiments.verify_results --date 2026-10-23
       --tag preprocessing-v1 [--task regression|multiclass|both]
Only verification JSON files are written. Recorded code hashes are checked
against the series' source snapshots, not the current working-tree runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
MSK = timezone(timedelta(hours=3))
RTOL = 1e-9


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_key(config):
    return json.dumps(config, sort_keys=True)


def oracle_scores(actual, prediction, task):
    """Use sklearn directly, without the experiment's metric implementation."""
    if task == "regression":
        return {"rmse": float(np.sqrt(mean_squared_error(actual, prediction)))}
    return {
        "macro_f1": float(f1_score(actual, prediction, average="macro",
                                    labels=sorted(set(actual)), zero_division=0)),
        "accuracy": float(accuracy_score(actual, prediction)),
        "weighted_f1": float(f1_score(actual, prediction, average="weighted", zero_division=0)),
    }


class ResultVerifier:
    def __init__(self, task, date, tag, root=ROOT):
        self.root = Path(root)
        self.task = task
        self.target = "site_eui" if task == "regression" else "attack_cat"
        self.date_dir = self.root / "results" / date
        self.suite = self.date_dir / f"series-{task}-{tag}"
        self.checks = []
        self.evidence = {}

    def check(self, name, condition, detail=None):
        row = {"name": name, "passed": bool(condition)}
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)
        return row["passed"]

    def equal(self, name, actual, expected):
        return self.check(name, actual == expected,
                          None if actual == expected else {"actual": actual, "expected": expected})

    def arrays_equal(self, name, actual, expected):
        return self.check(name, np.array_equal(np.asarray(actual), np.asarray(expected)))

    def close(self, name, actual, expected, rtol=RTOL, atol=0.0):
        return self.check(name, np.allclose(actual, expected, rtol=rtol, atol=atol),
                          {"relative_tolerance": rtol, "absolute_tolerance": atol})

    def guarded(self, name, callback):
        try:
            callback()
        except Exception as exc:
            self.check(name + "/exception", False,
                       {"error": repr(exc), "traceback": traceback.format_exc()})

    def utility(self, record):
        return -record["scores"]["rmse"] if self.task == "regression" else record["scores"]["macro_f1"]

    def verify_sources(self):
        actual_data = {name: sha256(self.root / "data" / self.task / name)
                       for name in self.environment["data_sha256"]}
        self.evidence["data_sha256"] = actual_data
        self.equal("sources/data_hashes", actual_data, self.environment["data_sha256"])
        snapshots = {}
        for relative, expected in self.environment["code_sha256"].items():
            # Windows paths are retained in the original metadata even when
            # this verifier is later run on another operating system.
            portable = Path(relative.replace("\\", "/"))
            candidates = [self.suite / "source" / portable, self.suite / "source" / portable.name]
            source = next((p for p in candidates if p.is_file()), candidates[0])
            self.check(f"sources/snapshot_exists/{relative}", source.is_file())
            actual = sha256(source) if source.is_file() else None
            snapshots[relative] = {"path": str(source.relative_to(self.root)),
                                   "expected_sha256": expected, "actual_sha256": actual}
            self.equal(f"sources/snapshot_hash/{relative}", actual, expected)
        self.evidence["source_snapshots"] = snapshots
        # Resolve the class embedded in the pickle from the verified snapshot;
        # later edits to the working-tree preprocessor must not change replay.
        prep_snapshot = next((entry for relative, entry in snapshots.items()
                              if Path(relative.replace("\\", "/")).name == "preprocessing.py"), None)
        if prep_snapshot and prep_snapshot["actual_sha256"] == prep_snapshot["expected_sha256"]:
            spec = importlib.util.spec_from_file_location("experiments.preprocessing", self.root / prep_snapshot["path"])
            module = importlib.util.module_from_spec(spec)
            sys.modules["experiments.preprocessing"] = module
            spec.loader.exec_module(module)
            self.evidence["pipeline_replay_preprocessing_source"] = prep_snapshot["path"]

    def verify_splits(self):
        with np.load(self.suite / "splits.npz", allow_pickle=False) as archive:
            self.splits = {key: archive[key].copy() for key in archive.files}
        sizes = read_json(self.suite / "splits.json")["sizes"]
        for key, values in self.splits.items():
            self.check(f"splits/{key}/integer", np.issubdtype(values.dtype, np.integer))
            self.check(f"splits/{key}/bounds", len(values) > 0 and np.all((values >= 0) & (values < len(self.train))))
            self.equal(f"splits/{key}/size", len(values), sizes[key])
            self.equal(f"splits/{key}/unique", len(np.unique(values)), len(values))
        raw_features = self.train.drop(columns=["id", self.target])
        groups = pd.util.hash_pandas_object(raw_features, index=False).to_numpy()
        leak_counts = {}
        for stage in ("inner", "outer"):
            fit, valid = self.splits[stage + "_fit"], self.splits[stage + "_valid"]
            self.check(f"splits/{stage}/row_disjoint", not np.intersect1d(fit, valid).size)
            leak_counts[stage] = len(set(groups[fit]).intersection(groups[valid]))
            self.equal(f"splits/{stage}/exact_feature_group_leakage", leak_counts[stage], 0)
            if self.task == "regression":
                self.check(f"splits/{stage}/temporal_order",
                           self.train.Year_Factor.iloc[fit].max() < self.train.Year_Factor.iloc[valid].min())
            else:
                classes = sorted(self.train[self.target].unique())
                for name, indices in (("fit", fit), ("valid", valid)):
                    self.equal(f"splits/{stage}/{name}/all_classes",
                               sorted(self.train[self.target].iloc[indices].unique()), classes)
        outer_fit, outer_valid = self.splits["outer_fit"], self.splits["outer_valid"]
        self.arrays_equal("splits/outer_partition", np.sort(np.r_[outer_fit, outer_valid]), np.arange(len(self.train)))
        self.arrays_equal("splits/inner_partition_of_outer_fit",
                          np.sort(np.r_[self.splits["inner_fit"], self.splits["inner_valid"]]), np.sort(outer_fit))
        if self.task == "regression":
            years = sorted(self.train.Year_Factor.unique())
            expected = {"inner_fit": np.flatnonzero(self.train.Year_Factor < years[-2]),
                        "inner_valid": np.flatnonzero(self.train.Year_Factor == years[-2]),
                        "outer_fit": np.flatnonzero(self.train.Year_Factor < years[-1]),
                        "outer_valid": np.flatnonzero(self.train.Year_Factor == years[-1])}
            for key, positions in expected.items():
                self.arrays_equal(f"splits/{key}/actual_year_indices", self.splits[key], positions)
        self.evidence["exact_feature_group_overlap_counts"] = leak_counts

    def verify_record(self, record):
        name = "runs/" + record["run_id"]
        directory = self.root / Path(record["path"].replace("\\", "/"))
        for filename in ("report.md", "metrics.json", "preprocessing_audit.json", "validation_predictions.csv"):
            self.check(name + "/exists/" + filename, (directory / filename).is_file())
        metrics = read_json(directory / "metrics.json")
        self.equal(name + "/summary_matches_metrics", record, metrics)
        self.equal(name + "/status", metrics["status"], "completed")
        self.equal(name + "/environment", metrics["environment"], self.environment)
        audit = read_json(directory / "preprocessing_audit.json")
        self.equal(name + "/audit_width", len(audit["feature_names"]), metrics["n_features"])
        predictions = pd.read_csv(directory / "validation_predictions.csv")
        positions = self.splits[record["stage"] + "_valid"]
        self.equal(name + "/n_validation", len(predictions), metrics["n_validation"])
        self.equal(name + "/n_fit", len(self.splits[record["stage"] + "_fit"]), metrics["n_fit"])
        self.arrays_equal(name + "/row_position", predictions.row_position, positions)
        self.arrays_equal(name + "/id_order", predictions.id, self.train.id.iloc[positions])
        if self.task == "regression":
            self.close(name + "/actual_from_source", predictions.actual, self.train[self.target].iloc[positions], rtol=1e-12)
            self.check(name + "/finite_prediction", np.isfinite(predictions.prediction).all())
        else:
            self.arrays_equal(name + "/actual_from_source", predictions.actual, self.train[self.target].iloc[positions])
            self.check(name + "/valid_classes", set(predictions.prediction).issubset(set(self.train[self.target])))
        raw_oracle = oracle_scores(predictions.actual, predictions.prediction, self.task)
        # sklearn 1.5.2 Lasso/Ridge fit on float32 arrays retain float32
        # coefficients and predict float32. pandas serializes such predictions
        # as the shortest decimal that round-trips to float32. Reading these
        # strings as float64 changes the represented numbers. Restore the
        # known producing dtype, never choose whichever dtype matches a score.
        float32_prediction = (self.task == "regression" and record["config"]["family"] in {"l1", "l2"}
                              and self.environment["sklearn"] == "1.5.2")
        restored_prediction = (predictions.prediction.to_numpy(dtype=np.float32)
                               if float32_prediction else predictions.prediction)
        oracle = oracle_scores(predictions.actual, restored_prediction, self.task)
        for metric, value in oracle.items():
            self.close(name + "/metric_oracle/" + metric, metrics["scores"][metric], value)
        self.evidence.setdefault("recomputed_scores", {})[record["run_id"]] = oracle
        self.evidence.setdefault("prediction_serialization", {})[record["run_id"]] = {
            "expected_original_dtype": "float32" if float32_prediction else (
                "float64" if self.task == "regression" else "class_label"),
            "basis": "sklearn 1.5.2 Lasso/Ridge on saved float32 design arrays" if float32_prediction
                     else "tree regression float64 or categorical classification predictions",
            "raw_csv_inferred_dtype_scores": raw_oracle,
            "original_dtype_restored_scores": oracle,
        }
        reference = self.by_id.get(record.get("reference"))
        if record.get("reference"):
            self.check(name + "/reference_exists", reference is not None)
        if reference is not None and "gain" in record:
            self.close(name + "/gain", record["gain"], self.utility(record) - self.utility(reference), atol=1e-12)
        if record["config"]["family"] == "catboost" and "strength" in record["config"]:
            self.check(name + "/native_l2_default_reference", reference is not None
                       and reference["stage"] == record["stage"]
                       and reference["config"] == {"family": "catboost", "recipe": record["config"]["recipe"]}
                       and reference["model_parameters"]["l2_leaf_reg"] == 3)
        if record["config"]["recipe"] in {"vif", "scaled", "gaussian", "engineered"}:
            self.check(name + "/vif_threshold", all(isinstance(value, (int, float)) and np.isfinite(value)
                       and value <= 10 + 1e-9 for value in audit["numeric_vif_after"].values()))

    def verify_frozen(self):
        frozen = read_json(self.suite / "frozen_selection.json")
        self.frozen = frozen
        self.equal("frozen/summary_matches_file", self.summary["frozen"], frozen)
        inner = [r for r in self.records if r["stage"] == "inner"]
        outer = [r for r in self.records if r["stage"] == "outer"]
        frozen_at = datetime.fromisoformat(frozen["frozen_at"])
        self.check("frozen/after_all_inner", all(datetime.fromisoformat(r["finished"]) <= frozen_at for r in inner))
        self.check("frozen/before_all_outer", all(frozen_at <= datetime.fromisoformat(r["started"]) for r in outer))
        self.equal("frozen/outer_configs_exactly_panel", [r["config"] for r in outer], frozen["outer_panel"])
        expected_tree = {}
        for family in ("catboost", "forest"):
            candidates = [r for r in inner if r["config"]["family"] == family and "strength" not in r["config"]]
            self.equal(f"protocol/{family}/all_six_recipes", sorted(r["config"]["recipe"] for r in candidates),
                       sorted(["baseline", "correlation", "vif", "scaled", "gaussian", "engineered"]))
            best = max(candidates, key=self.utility)
            expected_tree[family] = best
            self.equal(f"frozen/{family}/preprocessing_inner_winner", frozen["preprocessing"][family], best["config"])
            self.equal(f"frozen/{family}/selected_run", self.summary["inner_selected"][family], best["run_id"])
        native_candidates = [r for r in inner if r["config"]["family"] == "catboost"
                             and r["config"]["recipe"] == expected_tree["catboost"]["config"]["recipe"]]
        native = max(native_candidates, key=self.utility)
        self.equal("protocol/native_l2/all_strengths", sorted(r["config"].get("strength", 3) for r in native_candidates),
                   [1, 3, 10, 30])
        self.equal("frozen/native_l2_inner_winner", frozen["catboost_native_l2"], native["config"])
        for family in ("l1", "l2"):
            candidates = [r for r in inner if r["config"]["family"] == family]
            expected_strengths = ([.1, 1, 10] if self.task == "regression" and family == "l2" else [.01, .1, 1])
            self.equal(f"protocol/{family}/all_strengths", sorted(r["config"]["strength"] for r in candidates), expected_strengths)
            self.check(f"protocol/{family}/engineered_recipe", all(r["config"]["recipe"] == "engineered" for r in candidates))
            converged = [r for r in candidates if r["converged"]]
            best = max(converged or candidates, key=self.utility)
            self.equal(f"frozen/{family}/inner_config_including_strength", frozen["linear"][family], best["config"])
            self.equal(f"frozen/{family}/convergence", frozen["linear_inner_converged"][family], best["converged"])
        best_tree = max([expected_tree["forest"], native], key=self.utility)
        self.equal("frozen/final_inner_winner", frozen["best_tree_for_final_refit"], best_tree["config"])
        panel = []
        for family in ("catboost", "forest"):
            panel.extend([{"family": family, "recipe": "baseline"}, {"family": family, "recipe": "engineered"},
                          expected_tree[family]["config"]])
        panel.extend([native["config"], frozen["linear"]["l1"], frozen["linear"]["l2"]])
        expected_panel = list({config_key(config): config for config in panel}.values())
        self.equal("frozen/panel_reconstructed_from_inner_only", frozen["outer_panel"], expected_panel)

    def verify_arrays(self, directory, audit, name):
        # All files are artifacts created by this workspace. Only the owned
        # y_train vector may require pickle because class labels are objects.
        with np.load(directory / "preprocessed_data.npz", allow_pickle=False) as arrays:
            for key, rows in (("X_train", len(self.train)), ("X_test", len(self.test))):
                matrix = arrays[key]
                self.equal(name + "/" + key + "/shape", list(matrix.shape), [rows, len(audit["feature_names"])])
                self.equal(name + "/" + key + "/dtype", str(matrix.dtype), "float32")
                self.check(name + "/" + key + "/finite", np.isfinite(matrix).all())
            self.arrays_equal(name + "/train_id_order", arrays["train_id"], self.train.id)
            self.arrays_equal(name + "/test_id_order", arrays["test_id"], self.test.id)
            self.arrays_equal(name + "/feature_names", arrays["feature_names"], audit["feature_names"])
            # Retain only a small comparison window for the inference checks.
            saved_train = arrays["X_train"][:32].copy()
            saved_test = arrays["X_test"][:32].copy()
        with np.load(directory / "preprocessed_data.npz", allow_pickle=True) as arrays:
            target = arrays["y_train"]
            self.equal(name + "/y_train_shape", list(target.shape), [len(self.train)])
            self.arrays_equal(name + "/target_order", target, self.train[self.target])
        return saved_train, saved_test

    def verify_final(self):
        final = self.summary["final"]
        directory = self.root / Path(final["path"].replace("\\", "/"))
        name = "final"
        for filename in ("report.md", "metrics.json", "preprocessing_audit.json", "model.joblib",
                         "preprocessed_data.npz", "submission.csv"):
            self.check(name + "/exists/" + filename, (directory / filename).is_file())
        metrics = read_json(directory / "metrics.json")
        self.equal("final/summary_matches_metrics", final, metrics)
        self.equal("final/status", metrics["status"], "completed")
        self.equal("final/environment", metrics["environment"], self.environment)
        self.equal("final/config_frozen", metrics["config"], self.frozen["best_tree_for_final_refit"])
        self.check("final/started_after_freeze", datetime.fromisoformat(metrics["started"])
                   >= datetime.fromisoformat(self.frozen["frozen_at"]))
        audit = read_json(directory / "preprocessing_audit.json")
        self.equal("final/feature_width", len(audit["feature_names"]), metrics["n_features"])
        saved_train, saved_test = self.verify_arrays(directory, audit, name)
        submission = pd.read_csv(directory / "submission.csv")
        self.equal("final/submission_columns", list(submission), list(self.sample))
        self.equal("final/submission_rows", len(submission), len(self.test))
        self.arrays_equal("final/submission_id_order", submission.id, self.test.id)
        self.arrays_equal("final/template_id_order", submission.id, self.sample.id)
        self.check("final/submission_unique_ids", submission.id.is_unique)
        self.check("final/submission_no_missing", not submission.isna().any().any())
        if self.task == "regression":
            self.check("final/submission_finite", np.isfinite(submission[self.target]).all())
        else:
            self.check("final/submission_valid_classes", set(submission[self.target]).issubset(set(self.train[self.target])))
        self.equal("final/submission_sha256", sha256(directory / "submission.csv"), metrics["submission_sha256"])
        model = joblib.load(directory / "model.joblib")
        prep = model.named_steps["preprocess"]
        self.arrays_equal("final/model_feature_names", prep.get_feature_names_out(), audit["feature_names"])
        self.close("final/npz_matches_train_transform_first32", prep.transform(self.train.drop(columns=["id", self.target]).iloc[:32]), saved_train, rtol=1e-6, atol=1e-6)
        raw_test = self.test.drop(columns="id").iloc[:32]
        self.close("final/npz_matches_test_transform_first32", prep.transform(raw_test), saved_test, rtol=1e-6, atol=1e-6)
        with threadpool_limits(limits=2):
            prediction = np.asarray(model.predict(raw_test)).reshape(-1)
        if self.task == "regression":
            self.close("final/reloaded_predictions_first32", prediction, submission[self.target].iloc[:32], rtol=1e-10, atol=1e-10)
        else:
            self.arrays_equal("final/reloaded_predictions_first32", prediction, submission[self.target].iloc[:32])
        self.evidence["final_directory"] = str(directory.relative_to(self.root))

    def verify_requested_export(self):
        directory = self.date_dir / ("requested-preprocessing-" + self.task)
        for filename in ("manifest.json", "README.md", "preprocessing_audit.json", "preprocessor.joblib", "preprocessed_data.npz"):
            self.check("requested/exists/" + filename, (directory / filename).is_file())
        manifest = read_json(directory / "manifest.json")
        audit = read_json(directory / "preprocessing_audit.json")
        self.equal("requested/task", manifest["task"], self.task)
        self.equal("requested/dtype", manifest["dtype"], "float32")
        self.equal("requested/train_shape", manifest["shape_train"], [len(self.train), len(audit["feature_names"])])
        self.equal("requested/test_shape", manifest["shape_test"], [len(self.test), len(audit["feature_names"])])
        self.equal("requested/data_hashes", manifest["original_data_sha256"],
                   {key: self.environment["data_sha256"][key] for key in ("train.csv", "test.csv")})
        expected_prep_hash = next(value for key, value in self.environment["code_sha256"].items()
                                  if Path(key.replace("\\", "/")).name == "preprocessing.py")
        self.equal("requested/preprocessor_source_hash", manifest["preprocessor_sha256"], expected_prep_hash)
        saved_train, saved_test = self.verify_arrays(directory, audit, "requested")
        prep = joblib.load(directory / "preprocessor.joblib")
        params = prep.get_params()
        self.check("requested/all_requested_steps_enabled", all(params[key] for key in
                   ("drop_correlated", "drop_vif", "scale", "gaussian", "engineer")))
        self.arrays_equal("requested/feature_names", prep.get_feature_names_out(), audit["feature_names"])
        self.close("requested/train_transform_first32", prep.transform(self.train.drop(columns=["id", self.target]).iloc[:32]), saved_train, rtol=1e-6, atol=1e-6)
        self.close("requested/test_transform_first32", prep.transform(self.test.drop(columns="id").iloc[:32]), saved_test, rtol=1e-6, atol=1e-6)

    def run(self):
        def load():
            self.summary = read_json(self.suite / "summary.json")
            self.environment = read_json(self.suite / "environment.json")
            self.records = self.summary["records"]
            self.by_id = {record["run_id"]: record for record in self.records}
            self.train, self.test, self.sample = [pd.read_csv(self.root / "data" / self.task / filename)
                                                for filename in ("train.csv", "test.csv", "sample_submission.csv")]
            self.equal("series/task", self.summary["task"], self.task)
            self.equal("series/records_snapshot", self.records, read_json(self.suite / "records.json"))
            self.equal("series/unique_run_ids", len(self.by_id), len(self.records))
            self.evidence["status_counts"] = dict(Counter(r["status"] for r in self.records))
            self.evidence["convergence_counts"] = dict(Counter(str(r["converged"]) for r in self.records))
            self.check("series/all_records_completed", all(r["status"] == "completed" for r in self.records))
        self.guarded("load", load)
        if hasattr(self, "train"):
            self.guarded("sources", self.verify_sources)
            self.guarded("splits", self.verify_splits)
            for record in self.records:
                self.guarded("runs/" + record["run_id"], lambda record=record: self.verify_record(record))
            self.guarded("frozen", self.verify_frozen)
            self.guarded("final", self.verify_final)
            self.guarded("requested", self.verify_requested_export)
        failures = [check for check in self.checks if not check["passed"]]
        result = {"task": self.task, "series": str(self.suite.relative_to(self.root)),
                  "verified_at": datetime.now(MSK).isoformat(timespec="seconds"),
                  "status": "passed" if not failures else "failed", "metric_relative_tolerance": RTOL,
                  "check_count": len(self.checks), "failure_count": len(failures),
                  "failures": failures, "evidence": self.evidence, "checks": self.checks}
        if self.suite.is_dir():
            (self.suite / "independent-verification.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--tag", default="preprocessing-v1")
    parser.add_argument("--task", choices=("regression", "multiclass", "both"), default="both")
    args = parser.parse_args(argv)
    datetime.strptime(args.date, "%Y-%m-%d")
    tasks = ("regression", "multiclass") if args.task == "both" else (args.task,)
    results = [ResultVerifier(task, args.date, args.tag).run() for task in tasks]
    passed = all(result["status"] == "passed" for result in results)
    if args.task == "both":
        combined = {"verified_at": datetime.now(MSK).isoformat(timespec="seconds"),
                    "status": "passed" if passed else "failed", "tasks": results}
        directory = ROOT / "results" / args.date
        if directory.is_dir():
            (directory / "VERIFICATION.json").write_text(
                json.dumps(combined, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for result in results:
        print(f'{result["task"]}: {result["status"]}; checks={result["check_count"]}; failures={result["failure_count"]}')
        for failure in result["failures"]:
            print(json.dumps(failure, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
