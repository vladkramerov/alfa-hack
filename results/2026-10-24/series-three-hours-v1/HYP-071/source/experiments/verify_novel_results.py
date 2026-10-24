"""Independent SPEC-004 artifact verification; never imports or fits a model.

Usage: .venv-ml/Scripts/python.exe -m experiments.verify_novel_results --date 2026-10-23
Omit --date for the current Moscow date; --date latest selects the latest series.
Only the selected series' independent-verification.json is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
MSK = timezone(timedelta(hours=3))
CONTROL_DATE = "2026-10-23"
EXPECTED = {("HYP-011", "regression"), ("HYP-011", "multiclass")}
EXPECTED |= {(f"HYP-{n:03d}", "regression") for n in range(12, 16)}
EXPECTED |= {(f"HYP-{n:03d}", "multiclass") for n in range(16, 21)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_hash(fit, valid):
    return hashlib.sha256(np.asarray(fit, dtype="<i8").tobytes() + b"|"
                          + np.asarray(valid, dtype="<i8").tobytes()).hexdigest()


def identity(candidate):
    return candidate["hypothesis"], candidate["task"]


class Verifier:
    def __init__(self, date, root=ROOT):
        self.root = Path(root)
        self.suite = self.root / "results" / date / "series-ten-hypotheses-v1"
        self.checks = []
        self.recomputed = []
        self.datasets = {}

    def check(self, name, condition, detail=None):
        row = {"name": name, "passed": bool(condition)}
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)

    def equal(self, name, actual, expected):
        self.check(name, actual == expected)

    def arrays(self, name, actual, expected):
        self.check(name, np.array_equal(actual, expected))

    def close(self, name, actual, expected, atol=1e-12):
        self.check(name, np.allclose(actual, expected, rtol=1e-10, atol=atol),
                   {"rtol": 1e-10, "atol": atol})

    def guarded(self, name, callback):
        try:
            callback()
        except Exception as exc:
            self.check(name + "/exception", False,
                       {"error": repr(exc), "traceback": traceback.format_exc()})

    def data(self, task):
        if task in self.datasets:
            return self.datasets[task]
        base = self.root / "data" / task
        frame = pd.read_csv(base / "train.csv")
        target = "site_eui" if task == "regression" else "attack_cat"
        X, y = frame.drop(columns=["id", target]), frame[target]
        historical = read_json(self.root / "results" / CONTROL_DATE / "new-family-records.json")
        control = next(r for r in historical if r["task"] == task)
        hashes = {name: file_hash(base / name) for name in control["hashes"]}
        with np.load(self.root / "results" / CONTROL_DATE
                     / f"series-{task}-preprocessing-v1" / "splits.npz", allow_pickle=False) as saved:
            fit, valid = saved["outer_fit"].copy(), saved["outer_valid"].copy()
        key = split_hash(fit, valid)
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        prefix = "data/" + task
        self.equal(prefix + "/hashes_match_control", hashes, control["hashes"])
        self.equal(prefix + "/split_key_matches_control", key, control["split_key"])
        self.equal(prefix + "/split_sizes", (len(fit), len(valid)),
                   (35000, 18308) if task == "regression" else (70529, 17141))
        self.check(prefix + "/split_partition", np.array_equal(
            np.sort(np.concatenate([fit, valid])), np.arange(len(frame))))
        self.check(prefix + "/raw_X_groups_disjoint", not np.intersect1d(groups[fit], groups[valid]).size)
        if task == "regression":
            self.check(prefix + "/temporal_split", bool((X.iloc[fit].Year_Factor < 5).all()
                       and (X.iloc[valid].Year_Factor == 5).all()))
        result = dict(X=X, y=y, fit=fit, valid=valid, groups=groups, hashes=hashes, key=key)
        self.datasets[task] = result
        return result

    def sources(self):
        manifest = read_json(self.suite / "source-manifest.json")
        self.equal("source/frozen_plan_manifest", self.plan["source_sha256"], manifest)
        self.check("source/manifest_nonempty", bool(manifest))
        self.check("source/unique_snapshot_basenames",
                   len({Path(p.replace("\\", "/")).name for p in manifest}) == len(manifest))
        for relative, expected in manifest.items():
            name = Path(relative.replace("\\", "/")).name
            self.equal("source/snapshot/" + relative, file_hash(self.suite / "source" / name), expected)
        self.manifest = manifest

    def registry(self):
        rows = read_json(self.root / "results" / "spec004-attempts.json")
        fingerprints = [r["fingerprint"] for r in self.records]
        self.equal("registry/unique_series_fingerprints", len(set(fingerprints)), 11)
        self.check("registry/no_duplicate_fingerprints", len({r["fingerprint"] for r in rows}) == len(rows))
        selected = [r for r in rows if r["fingerprint"] in fingerprints]
        self.equal("registry/eleven_claims", len(selected), 11)
        self.equal("registry/identities", {identity(r["candidate"]) for r in selected}, EXPECTED)
        for record in self.records:
            matches = [r for r in rows if r["fingerprint"] == record["fingerprint"]]
            prefix = "/".join(identity(record["candidate"])) + "/registry"
            self.equal(prefix + "/one_claim", len(matches), 1)
            if not matches:
                continue
            row = matches[0]
            for field in ["candidate", "hashes", "split_key", "ledger_sha256_before_fit", "started", "status", "score", "directory"]:
                self.equal(prefix + "/" + field, row.get(field), record.get(field))
            self.check(prefix + "/ledger_hash_recorded", bool(re.fullmatch(
                r"[0-9a-f]{64}", row.get("ledger_sha256_before_fit", ""))))
            self.check(prefix + "/started_before_finished", datetime.fromisoformat(row["started"])
                       <= datetime.fromisoformat(row["finished"]))
        # Recorded hashes attest the saved claim; no historical ledger snapshot exists
        # to independently prove the contents read at that earlier instant.

    def target_folds(self, directory, audit, data, prefix):
        feature = audit["new_features"]
        fit = data["fit"]
        y = data["y"].iloc[fit].to_numpy(float)
        groups = data["groups"][fit]
        self.equal(prefix + "/oof/n_fit", feature["n_fit"], len(fit))
        self.equal(prefix + "/oof/n_groups", feature["n_groups"], len(np.unique(groups)))
        self.equal(prefix + "/oof/smoothing", feature["smoothing"], 20.)
        self.equal(prefix + "/oof/n_splits", feature["n_splits"], 5)
        self.check(prefix + "/oof/audit_no_leakage", feature["no_group_overlap"] is True
                   and feature["no_self_target_leakage"] is True)
        coverage = np.zeros(len(fit), dtype=int)
        with np.load(directory / "encoding-folds.npz", allow_pickle=False) as folds:
            self.equal(prefix + "/oof/artifact_keys", set(folds.files),
                       {f"fold{i}_{part}" for i in range(5) for part in ["fit", "valid"]})
            self.equal(prefix + "/oof/audit_folds", len(feature["folds"]), 5)
            for i in range(5):
                a, b = folds[f"fold{i}_fit"], folds[f"fold{i}_valid"]
                name = prefix + f"/oof/fold{i}"
                self.arrays(name + "/complete_partition", np.sort(np.concatenate([a, b])), np.arange(len(fit)))
                self.check(name + "/groups_disjoint", not np.intersect1d(groups[a], groups[b]).size)
                np.add.at(coverage, b, 1)
                saved = feature["folds"][i]
                self.equal(name + "/fold_number", saved["fold"], i)
                for key, actual in [("train_rows", len(a)), ("valid_rows", len(b)),
                                    ("train_groups", len(np.unique(groups[a]))),
                                    ("valid_groups", len(np.unique(groups[b]))),
                                    ("group_overlap", 0), ("row_overlap", 0)]:
                    self.equal(name + "/" + key, saved[key], actual)
                self.close(name + "/fold_fit_prior", saved["prior"], y[a].mean())
        self.arrays(prefix + "/oof/coverage_once", coverage, np.ones(len(fit), dtype=int))

    def weights(self, directory, audit, data, prefix):
        groups = pd.Series(data["groups"][data["fit"]])
        expected = (1. / groups.map(groups.value_counts())).to_numpy(float)
        with np.load(directory / "training-weights.npz", allow_pickle=False) as saved:
            self.arrays(prefix + "/weights/positions", saved["positions"], data["fit"])
            self.arrays(prefix + "/weights/exact_inverse_group_frequency", saved["weights"], expected)
            self.close(prefix + "/weights/sum_equals_groups", saved["weights"].sum(), groups.nunique())
        self.close(prefix + "/weights/audit_sum", audit["weights_sum"], expected.sum())
        self.close(prefix + "/weights/audit_min", audit["weight_min"], expected.min())
        self.close(prefix + "/weights/audit_max", audit["weight_max"], expected.max())
        self.equal(prefix + "/weights/not_normalized", audit["weights_normalized"], False)

    def run_record(self, record):
        candidate = record["candidate"]
        hypothesis, task = identity(candidate)
        prefix = hypothesis + "/" + task
        data = self.data(task)
        directory = (self.root / record["directory"]).resolve()
        self.check(prefix + "/artifact_within_results", directory.is_relative_to((self.root / "results").resolve()))
        evidence = read_json(directory / "evidence.json")
        metrics = read_json(directory / "metrics.json")
        audit = read_json(directory / "preprocessing-audit.json")
        request = read_json(directory / "request.json")
        self.equal(prefix + "/evidence_equals_record", evidence, record)
        self.equal(prefix + "/request_candidate", request["candidate"], candidate)
        self.equal(prefix + "/status", metrics["status"], "completed")
        self.equal(prefix + "/data_hashes", record["hashes"], data["hashes"])
        self.equal(prefix + "/split_key", record["split_key"], data["key"])
        self.equal(prefix + "/source_manifest", read_json(directory / "source-manifest.json"), self.manifest)
        self.equal(prefix + "/fit_rows", metrics["fit_rows"], len(data["fit"]))
        self.equal(prefix + "/valid_rows", metrics["valid_rows"], len(data["valid"]))
        self.equal(prefix + "/input_features", audit["input_features"], list(data["X"]))
        self.check(prefix + "/id_target_excluded", audit["identifier_excluded"] is True
                   and audit["target_excluded"] is True)
        with np.load(directory / "splits.npz", allow_pickle=False) as saved:
            self.arrays(prefix + "/saved_fit_positions", saved["fit"], data["fit"])
            self.arrays(prefix + "/saved_valid_positions", saved["valid"], data["valid"])
        # The trusted local worker stores classification y as a pandas object
        # array; other archives contain no objects and remain pickle-disabled.
        with np.load(directory / "validation_predictions.npz", allow_pickle=True) as saved:
            actual = data["y"].iloc[data["valid"]].to_numpy()
            prediction = saved["prediction"]
            self.arrays(prefix + "/prediction_positions", saved["positions"], data["valid"])
            self.arrays(prefix + "/actual_target", saved["y"], actual)
            self.equal(prefix + "/prediction_shape", prediction.shape, (len(actual),))
            if task == "regression":
                self.check(prefix + "/finite_predictions", np.isfinite(prediction).all())
                values = {"rmse": float(np.sqrt(mean_squared_error(actual, prediction)))}
            else:
                classes, probabilities = saved["classes"], saved["probabilities"]
                expected_classes = np.sort(data["y"].iloc[data["fit"]].unique())
                self.arrays(prefix + "/class_order", classes, expected_classes)
                self.arrays(prefix + "/metrics_class_order", classes, metrics["classes"])
                self.equal(prefix + "/probability_shape", probabilities.shape, (len(actual), len(classes)))
                self.check(prefix + "/finite_probabilities", np.isfinite(probabilities).all())
                self.check(prefix + "/probability_bounds", ((probabilities >= 0) & (probabilities <= 1)).all())
                self.arrays(prefix + "/argmax_prediction", prediction, classes[probabilities.argmax(axis=1)])
                if hypothesis != "HYP-018":
                    self.close(prefix + "/probability_mass", probabilities.sum(axis=1), 1., atol=1e-6)
                self.close(prefix + "/probability_row_sum_range", metrics["probability_row_sum_range"],
                           [probabilities.sum(axis=1).min(), probabilities.sum(axis=1).max()])
                values = {"macro_f1": float(f1_score(actual, prediction, average="macro", labels=expected_classes, zero_division=0)),
                          "accuracy": float(accuracy_score(actual, prediction)),
                          "weighted_f1": float(f1_score(actual, prediction, average="weighted", zero_division=0))}
                self.close(prefix + "/accuracy", metrics["accuracy"], values["accuracy"])
                report = read_json(directory / "classification-report.json")
                self.close(prefix + "/weighted_f1", report["weighted avg"]["f1-score"], values["weighted_f1"])
            metric = "rmse" if task == "regression" else "macro_f1"
            self.equal(prefix + "/metric_name", metrics["metric"], metric)
            self.close(prefix + "/independent_score", metrics["score"], values[metric])
            self.close(prefix + "/record_score", record["score"], values[metric])
            self.recomputed.append({"hypothesis": hypothesis, "task": task, **values})
        if hypothesis == "HYP-015":
            self.target_folds(directory, audit, data, prefix)
        elif hypothesis == "HYP-020":
            self.weights(directory, audit, data, prefix)
        elif hypothesis == "HYP-019":
            classes = np.sort(data["y"].unique()).tolist()
            self.equal(prefix + "/hierarchy/normal", audit["normal_class"], "Normal")
            self.equal(prefix + "/hierarchy/global_classes", audit["global_classes"], classes)
            self.equal(prefix + "/hierarchy/attack_classes", audit["attack_classes"], [c for c in classes if c != "Normal"])
            self.equal(prefix + "/hierarchy/binary_rows", audit["binary_fit_rows"], len(data["fit"]))
            self.equal(prefix + "/hierarchy/attack_rows", audit["attack_fit_rows"],
                       int((data["y"].iloc[data["fit"]] != "Normal").sum()))
            self.equal(prefix + "/hierarchy/separate_preprocessing", audit["separate_fitted_preprocessing"], True)

    def verify(self):
        summary = read_json(self.suite / "summary.json")
        self.records = read_json(self.suite / "records.json")
        self.plan = read_json(self.suite / "frozen-plan.json")
        self.equal("series/status", summary["status"], "completed")
        self.equal("series/eleven_records", len(self.records), 11)
        self.equal("series/eleven_planned", len(self.plan["candidates"]), 11)
        self.equal("series/identities", {identity(r["candidate"]) for r in self.records}, EXPECTED)
        self.equal("series/planned_identities", {identity(c) for c in self.plan["candidates"]}, EXPECTED)
        self.equal("series/summary_records", summary["records"], self.records)
        self.equal("series/planned_evaluations", summary["planned_evaluations"], 11)
        self.equal("series/zero_baseline_refits", summary["baseline_refits"], 0)
        self.equal("series/twelve_planned_component_fits", sum(r["planned_model_fit_count"] for r in self.records), 12)
        self.equal("series/reference_rmse", self.plan["reference_regression_rmse"], 43.0046386479306)
        self.equal("series/reference_macro_f1", self.plan["reference_grouped_macro_f1"], .5838559593680208)
        frozen = {identity(c): c for c in self.plan["candidates"]}
        for record in self.records:
            prefix = "/".join(identity(record["candidate"]))
            self.equal(prefix + "/completed", record["status"], "completed")
            self.equal(prefix + "/frozen_config", record["candidate"], frozen.get(identity(record["candidate"])))
        self.guarded("sources", self.sources)
        self.guarded("registry", self.registry)
        for record in self.records:
            self.guarded("/".join(identity(record["candidate"])), lambda r=record: self.run_record(r))

    def result(self):
        failed = sum(not row["passed"] for row in self.checks)
        return {"status": "passed" if not failed else "failed", "created_at_moscow": datetime.now(MSK).isoformat(),
                "series": self.suite.relative_to(self.root).as_posix(),
                "passed_checks": len(self.checks) - failed, "failed_checks": failed,
                "checked_runs": len(self.recomputed), "independent_metrics": self.recomputed, "checks": self.checks,
                "limitations": ["Historical ledger hashes and separate hierarchy preprocessing are checked against saved claims/audits; artifacts alone cannot prove historical execution."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(MSK).strftime("%Y-%m-%d"),
                        help="Moscow date YYYY-MM-DD, or latest")
    args = parser.parse_args()
    date = args.date
    if date == "latest":
        candidates = sorted((ROOT / "results").glob("????-??-??/series-ten-hypotheses-v1"))
        if not candidates:
            parser.error("No SPEC-004 series exists")
        date = candidates[-1].parent.name
    verifier = Verifier(date)
    verifier.guarded("verification", verifier.verify)
    result = verifier.result()
    if verifier.suite.is_dir():
        destination = verifier.suite / "independent-verification.json"
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "passed_checks", "failed_checks", "checked_runs"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
