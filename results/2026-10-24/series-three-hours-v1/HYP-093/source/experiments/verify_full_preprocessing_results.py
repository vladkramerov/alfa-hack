"""Independently verify HYP-021 artifacts without importing or fitting models.

Run after both records complete:
  .venv-ml/Scripts/python.exe -m experiments.verify_full_preprocessing_results --date latest
Only full-preprocessing-verification.json and FULL_PREPROCESSING_REPORT.md are written.
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
REFERENCE_DATE = "2026-10-23"
REFERENCES = {"regression": 42.19670002719412, "multiclass": .5914513709191194}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_key(fit, valid):
    return hashlib.sha256(np.asarray(fit, dtype="<i8").tobytes() + b"|"
                          + np.asarray(valid, dtype="<i8").tobytes()).hexdigest()


class Verifier:
    def __init__(self, date, root=ROOT):
        self.root = Path(root)
        self.directory = self.root / "results" / date
        self.checks = []
        self.rows = []

    def check(self, name, condition, detail=None):
        check = {"name": name, "passed": bool(condition)}
        if detail is not None:
            check["detail"] = detail
        self.checks.append(check)

    def equal(self, name, actual, expected):
        self.check(name, actual == expected)

    def arrays(self, name, actual, expected):
        self.check(name, np.array_equal(actual, expected))

    def close(self, name, actual, expected, atol=1e-12):
        self.check(name, np.allclose(actual, expected, rtol=1e-10, atol=atol))

    def guarded(self, name, callback):
        try:
            callback()
        except Exception as exc:
            self.check(name + "/exception", False,
                       {"error": repr(exc), "traceback": traceback.format_exc()})

    def verify_preprocessing(self, audit, X, kept_positions, name):
        retained = X.iloc[kept_positions]
        pre = audit["preprocessing"]
        self.equal(name + "/policy", pre["policy"], "full-v1")
        self.equal(name + "/fit_retained_only", pre["n_fit"], len(retained))
        self.equal(name + "/mandatory_normalization", pre["normalization_enabled"], True)
        numeric = retained.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
        expected_missing = {c: int(numeric[c].isna().sum()) for c in numeric}
        self.equal(name + "/train_missing_counts", pre["missing_counts"], expected_missing)
        kept_numeric = pre["numeric_columns"]
        self.check(name + "/numeric_subset", set(kept_numeric).issubset(numeric))
        expected_continuous = []
        for column in kept_numeric:
            values = numeric[column].fillna(numeric[column].median())
            counts = values.value_counts(dropna=False)
            if len(counts) >= 20 and counts.max() / len(retained) <= .8:
                expected_continuous.append(column)
        self.equal(name + "/continuous_eligibility", pre["continuous_columns"], expected_continuous)
        parameters = pre["quantile_parameters"]
        if expected_continuous:
            self.equal(name + "/quantile_settings", parameters,
                       {"output_distribution": "normal", "n_quantiles": min(512, len(retained)),
                        "subsample": None, "random_state": 42})
        else:
            self.equal(name + "/no_continuous_quantiles", parameters, None)
        mean_error = audit["normalized_training_mean_abs_max"]
        std_error = audit["normalized_training_std_error_max"]
        self.check(name + "/training_mean_error", np.isfinite(mean_error) and 0 <= mean_error < 1e-4)
        self.check(name + "/training_std_error", np.isfinite(std_error) and 0 <= std_error < 1e-4)
        means, scales = np.asarray(pre["scaler_mean"]), np.asarray(pre["scaler_scale"])
        self.equal(name + "/scaler_feature_count", len(means), len(pre["numeric_output_columns"]))
        self.equal(name + "/scaler_scale_count", len(scales), len(means))
        self.check(name + "/finite_scaler_statistics", np.isfinite(means).all()
                   and np.isfinite(scales).all() and (scales > 0).all())
        dropped = {d["column"]: d for d in pre["dropped_columns"]}
        self.equal(name + "/unique_dropped_columns", len(dropped), len(pre["dropped_columns"]))
        for column, detail in dropped.items():
            values = numeric[column] if column in numeric else retained[column]
            reason = detail["reason"]
            prefix = name + "/dropped/" + column
            if reason == "all_null":
                self.check(prefix, values.isna().all())
            elif reason == "constant":
                self.check(prefix, values.dropna().nunique() <= 1)
            elif reason == "exact_duplicate":
                other = detail["duplicate_of"]
                if column in numeric:
                    self.arrays(prefix, values.to_numpy(), numeric[other].to_numpy()) if not values.isna().any() else self.check(
                        prefix, np.array_equal(values.to_numpy(), numeric[other].to_numpy(), equal_nan=True))
                else:
                    self.check(prefix, values.equals(retained[other]))
            else:
                self.check(prefix + "/known_reason", False, reason)
        self.equal(name + "/numeric_columns_accounted",
                   set(numeric), set(kept_numeric) | (set(dropped) & set(numeric)))
        categorical = [c for c in retained if c not in numeric]
        self.equal(name + "/categorical_columns_accounted", set(categorical),
                   set(pre["categorical_columns"]) | (set(dropped) & set(categorical)))
        flags = [c for c in numeric if 0 < expected_missing[c] < len(retained)
                 and dropped.get(c, {}).get("reason") != "exact_duplicate"]
        self.equal(name + "/missing_flags", pre["missing_flag_columns"], flags)

    def run_record(self, record):
        task = record["task"]
        name = "HYP-021/" + task
        directory = (self.root / record["directory"]).resolve()
        self.check(name + "/results_path", directory.is_relative_to((self.root / "results").resolve()))
        self.equal(name + "/completed", record["status"], "completed")
        self.equal(name + "/evidence", read_json(directory / "evidence.json"), record)
        request = read_json(directory / "request.json")
        metrics = read_json(directory / "metrics.json")
        audit = read_json(directory / "preprocessing-audit.json")
        self.equal(name + "/request_candidate", request["candidate"], record["candidate"])
        self.equal(name + "/evaluate_only", request["mode"], "evaluate")
        self.equal(name + "/mandatory_candidate_policy", record["candidate"]["preprocessing"], "full-v1")
        self.equal(name + "/outlier_fraction", record["candidate"]["max_outlier_fraction"], .05)
        self.equal(name + "/candidate_family", record["candidate"]["family"],
                   "extra_trees" if task == "regression" else "xgboost")
        self.equal(name + "/target_transform", record["candidate"]["target_transform"],
                   "log1p" if task == "regression" else None)
        for source, expected in record["code_sha256"].items():
            base = Path(source.replace("\\", "/")).name
            self.equal(name + "/source/" + source, sha256(directory / "source" / base), expected)
        self.check(name + "/ledger_hash_recorded", bool(re.fullmatch(r"[0-9a-f]{64}", record.get("ledger_sha256_before_fit", ""))))
        self.check(name + "/started_before_finished", datetime.fromisoformat(record["started"])
                   <= datetime.fromisoformat(record["finished"]))
        controls = read_json(self.root / "results" / REFERENCE_DATE / "new-family-records.json")
        control = next(row for row in controls if row["task"] == task)
        hashes = {filename: sha256(self.root / "data" / task / filename) for filename in control["hashes"]}
        self.equal(name + "/actual_data_hashes", hashes, control["hashes"])
        self.equal(name + "/record_data_hashes", record["hashes"], hashes)
        frame = pd.read_csv(self.root / "data" / task / "train.csv")
        target = "site_eui" if task == "regression" else "attack_cat"
        X, y = frame.drop(columns=["id", target]), frame[target]
        with np.load(self.root / "results" / REFERENCE_DATE
                     / f"series-{task}-preprocessing-v1" / "splits.npz", allow_pickle=False) as old:
            fit, valid = old["outer_fit"], old["outer_valid"]
        self.equal(name + "/control_split_key", split_key(fit, valid), control["split_key"])
        self.equal(name + "/record_split_key", record["split_key"], control["split_key"])
        self.arrays(name + "/split_partition", np.sort(np.concatenate([fit, valid])), np.arange(len(frame)))
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        self.check(name + "/raw_X_groups_disjoint", not np.intersect1d(groups[fit], groups[valid]).size)
        with np.load(directory / "splits.npz", allow_pickle=False) as saved:
            self.arrays(name + "/fit_positions", saved["fit"], fit)
            self.arrays(name + "/valid_positions", saved["valid"], valid)
        with np.load(directory / "training-filter.npz", allow_pickle=False) as saved:
            keep, removed = saved["keep_mask"], saved["removed_positions"]
        self.equal(name + "/keep_shape", keep.shape, (len(fit),))
        self.equal(name + "/keep_boolean_dtype", keep.dtype, np.dtype(bool))
        self.equal(name + "/removed_one_dimensional", removed.ndim, 1)
        self.check(name + "/removed_integral", np.issubdtype(removed.dtype, np.integer))
        self.equal(name + "/removed_unique", len(np.unique(removed)), len(removed))
        self.arrays(name + "/removed_positions_match_mask", np.sort(removed), np.flatnonzero(~keep))
        self.check(name + "/removed_positions_in_fit", ((removed >= 0) & (removed < len(fit))).all())
        retained = np.flatnonzero(keep)
        self.check(name + "/no_validation_removed", not np.intersect1d(fit[removed], valid).size)
        self.check(name + "/global_removal_cap", len(removed) <= int(np.floor(.05 * len(fit))))
        outliers = audit["outliers"]
        self.equal(name + "/outer_fit_original_rows", audit["fit_original_rows"], len(fit))
        self.equal(name + "/outer_fit_retained_rows", audit["fit_retained_rows"], len(retained))
        for field, expected in [("n_input", len(fit)), ("n_retained", len(retained)),
                                ("n_removed", len(removed)), ("max_removed", int(np.floor(.05 * len(fit)))),
                                ("max_fraction", .05), ("classification", task == "multiclass")]:
            self.equal(name + "/outliers/" + field, outliers[field], expected)
        self.close(name + "/outliers/fraction", outliers["removed_fraction"], len(removed) / len(fit))
        self.equal(name + "/target_not_in_detector_audit", outliers["target_used_in_detector"], False)
        numeric = X.select_dtypes(include="number").columns.tolist()
        self.equal(name + "/detector_numeric_features", outliers["detector_numeric_columns"], numeric)
        self.check(name + "/detector_excludes_id_target", target not in outliers["detector_numeric_columns"]
                   and "id" not in outliers["detector_numeric_columns"])
        self.equal(name + "/detector_settings", outliers["detector_parameters"],
                   {"n_estimators": 100, "max_samples": min(256, len(fit)), "contamination": "auto", "random_state": 42, "n_jobs": 4})
        classes_audit = outliers["classes"]
        if task == "multiclass":
            fit_y = y.iloc[fit].reset_index(drop=True)
            counts = fit_y.value_counts()
            self.equal(name + "/class_caps/class_names", set(classes_audit), set(counts.index))
            for label, count in counts.items():
                n_removed = int((fit_y.iloc[removed] == label).sum())
                cap = min(int(np.floor(.05 * count)), int(count - min(50, count)))
                prefix = name + "/class_caps/" + str(label)
                self.check(prefix + "/removal_cap", n_removed <= int(np.floor(.05 * count)))
                self.check(prefix + "/rare_class_floor", count - n_removed >= min(50, count))
                for field, expected in [("before", int(count)), ("removed", n_removed),
                                        ("retained", int(count - n_removed)), ("max_removed", cap)]:
                    self.equal(prefix + "/audit_" + field, classes_audit[str(label)][field], expected)
        else:
            self.equal(name + "/no_regression_y_caps", classes_audit, {})
        self.verify_preprocessing(audit, X.iloc[fit], retained, name + "/preprocessing")
        # These are trusted local worker artifacts; pandas label arrays have
        # object dtype. Other archives are opened with pickle disabled.
        with np.load(directory / "validation_predictions.npz", allow_pickle=True) as saved:
            actual, prediction = y.iloc[valid].to_numpy(), saved["prediction"]
            self.arrays(name + "/all_validation_positions", saved["positions"], valid)
            self.arrays(name + "/validation_target", saved["y"], actual)
            self.equal(name + "/prediction_shape", prediction.shape, (len(valid),))
            if task == "regression":
                self.check(name + "/finite_predictions", np.isfinite(prediction).all())
                score = float(np.sqrt(mean_squared_error(actual, prediction)))
                metric = "rmse"
                extra = {}
            else:
                classes, probabilities = saved["classes"], saved["probabilities"]
                self.arrays(name + "/class_order", classes, np.sort(y.iloc[fit].unique()))
                self.equal(name + "/probability_shape", probabilities.shape, (len(valid), len(classes)))
                self.check(name + "/finite_probabilities", np.isfinite(probabilities).all())
                self.check(name + "/probability_bounds", ((probabilities >= 0) & (probabilities <= 1)).all())
                self.close(name + "/probability_mass", probabilities.sum(axis=1), 1., atol=1e-6)
                self.arrays(name + "/argmax_prediction", prediction, classes[probabilities.argmax(axis=1)])
                score = float(f1_score(actual, prediction, average="macro", labels=classes, zero_division=0))
                metric = "macro_f1"
                extra = {"accuracy": float(accuracy_score(actual, prediction)),
                         "weighted_f1": float(f1_score(actual, prediction, average="weighted", zero_division=0))}
        self.equal(name + "/metric", metrics["metric"], metric)
        self.equal(name + "/metric_status", metrics["status"], "completed")
        self.equal(name + "/validation_count", metrics["valid_rows"], len(valid))
        self.equal(name + "/original_fit_count", metrics["fit_rows"], len(fit))
        self.equal(name + "/retained_fit_count", metrics["fit_retained_rows"], len(retained))
        self.close(name + "/independent_score", metrics["score"], score)
        self.close(name + "/record_score", record["score"], score)
        self.equal(name + "/reference", record["reference"], REFERENCES[task])
        previous = read_json(self.root / "results" / REFERENCE_DATE / "series-ten-hypotheses-v1" / "independent-verification.json")
        prior = next(r for r in previous["independent_metrics"] if r["task"] == task
                     and r["hypothesis"] == ("HYP-012" if task == "regression" else "HYP-011"))
        self.equal(name + "/reference_verified", previous["status"], "passed")
        self.equal(name + "/reference_from_SPEC004", prior[metric], REFERENCES[task])
        improved = score < REFERENCES[task] if task == "regression" else score > REFERENCES[task]
        conclusion = "Подтверждена в проверенных условиях" if improved else "Не подтверждена по заданному критерию"
        self.equal(name + "/honest_conclusion", record["conclusion"], conclusion)
        self.rows.append({"task": task, "metric": metric, "reference": REFERENCES[task], "score": score,
                          "difference_new_minus_old": score - REFERENCES[task], "improved": improved,
                          "n_input": len(fit), "n_removed": len(removed), "n_retained": len(retained),
                          "preprocessing": audit["preprocessing"], "classes": classes_audit,
                          "directory": record["directory"], **extra})

    def verify(self):
        records = read_json(self.directory / "full-preprocessing-records.json")
        self.equal("series/two_records", len(records), 2)
        self.equal("series/two_task_identities", {r["task"] for r in records}, set(REFERENCES))
        self.check("series/all_completed", all(r["status"] == "completed" for r in records))
        self.equal("series/shared_source_snapshot", len({json.dumps(r["code_sha256"], sort_keys=True) for r in records}), 1)
        registry = read_json(self.root / "results" / "spec005-attempts.json")
        for record in records:
            self.equal("registry/" + record["task"] + "/one_claim", len([r for r in registry
                       if r["task"] == record["task"] and r["candidate_key"] == record["candidate_key"]]), 1)
            row = next(r for r in registry if r["task"] == record["task"] and r["candidate_key"] == record["candidate_key"])
            self.equal("registry/" + record["task"] + "/evidence", row, record)
            self.guarded("HYP-021/" + record["task"], lambda r=record: self.run_record(r))

    def result(self):
        failed = sum(not check["passed"] for check in self.checks)
        return {"status": "passed" if failed == 0 else "failed", "created_at_moscow": datetime.now(MSK).isoformat(),
                "passed_checks": len(self.checks) - failed, "failed_checks": failed,
                "checked_runs": len(self.rows), "comparisons": self.rows, "checks": self.checks,
                "limitations": ["Normalization moments and historical detector inputs are verified against recorded audit and source snapshots; transformed training matrices/detector traces were not saved.",
                                "No model or preprocessing fits are performed by this verifier.",
                                "Repeated outer validation is exploratory; this is not a hidden-platform score."]}

    def report(self, result):
        lines = ["# HYP-021: обязательная полная предобработка", "",
                 f"Независимая сверка: **{result['status']}**, проверок {result['passed_checks']} успешно / {result['failed_checks']} с ошибками; проверено запусков {result['checked_runs']}/2.", "",
                 "| Задача | Метрика | Сохранённый SPEC-004 | Full-v1 | Новое − прежнее | Вывод |",
                 "| --- | --- | ---: | ---: | ---: | --- |"]
        for task in REFERENCES:
            row = next((r for r in self.rows if r["task"] == task), None)
            if row is None:
                lines.append(f"| {task} | — | {REFERENCES[task]:.12f} | — | — | Не удалось независимо проверить |")
                continue
            outcome = "улучшение" if row["improved"] else "улучшение не подтверждено"
            lines.append(f"| {task} | {row['metric']} | {row['reference']:.12f} | {row['score']:.12f} | {row['difference_new_minus_old']:+.12f} | {outcome} |")
        lines += ["", "RMSE уменьшают, Macro F1 увеличивают. Прежние контрольные оценки переиспользованы из готовой независимой сверки; baseline заново не обучался этим верификатором. Проверено составное изменение: отдельный вклад выбросов, imputation, quantiles или scaling из этой пары опытов определить нельзя.", ""]
        for row in self.rows:
            pre = row["preprocessing"]
            lines += [f"## {row['task']}", "",
                      f"Training: {row['n_input']} → {row['n_retained']}; удалено {row['n_removed']} ({row['n_removed']/row['n_input']:.2%}), только из outer-fit. Все исходные validation positions и таргеты сохранены; validation/test не фильтруются.", "",
                      f"Gaussian quantile-преобразование применено к {len(pre['continuous_columns'])} непрерывным признакам: " + ", ".join(pre["continuous_columns"]) + ".",
                      f"Числовая imputation: median по retained train; {len(pre['missing_flag_columns'])} missing indicators. Пропуски retained train: " + json.dumps({k:v for k,v in pre["missing_counts"].items() if v}, ensure_ascii=False) + ".",
                      "Удалённые признаки: " + (json.dumps(pre["dropped_columns"], ensure_ascii=False) if pre["dropped_columns"] else "нет") + ".",
                      "Категории: отдельное представление пропусков и обработка неизвестных значений; категории не объявляются Gaussian.",
                      f"[Исходные артефакты]({Path(row['directory']).name}/report.md).", ""]
            if row["classes"]:
                lines += ["Удаление по классам (исходно / удалено / осталось): " + "; ".join(
                    f"{label}: {values['before']}/{values['removed']}/{values['retained']}" for label,values in row["classes"].items()) + ".", ""]
        lines += ["## Границы вывода", "",
                  "Нормализация обязательна по поручению пользователя и при отрицательной метрике; факт улучшения из этого требования не следует. Проверены train-only границы fit, лимит удаления ≤5% и защита редких классов. Средние и стандартные отклонения проверены по сохранённым audit-диагностикам (<1e-4), а не по отдельно сохранённой преобразованной матрице; аудит detector и замороженный код указывают, что target в detector не поступал.", "",
                  "QuantileTransformer(normal) приближает training marginals к нормальным; ties, дискретность и сдвиг распределения не позволяют обещать точную нормальность всех признаков либо будущего test. Категориальные метки классов к нормальному распределению не приводятся.", "",
                  "Outer validation уже использовался: вывод разведочный, без независимого test. Реальный LLM API платформы не проверен; это локальные метрики. Число leaderboard 579.658 и лидер 29.3946 не сопоставляются с RMSE/Macro F1 этой таблицы без подтверждённого определения показателя, состава задач и версии пакета; улучшение в определённое число раз не обещается.", "",
                  "[Машиночитаемая сверка](full-preprocessing-verification.json)."]
        if result["failed_checks"]:
            lines += ["", "Проверка не пройдена: выводы о корректности предобработки до устранения ошибок не подтверждены; подробности failed checks приведены в JSON."]
        return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(MSK).strftime("%Y-%m-%d"), help="Moscow YYYY-MM-DD or latest")
    args = parser.parse_args()
    date = args.date
    if date == "latest":
        matches = sorted((ROOT / "results").glob("????-??-??/full-preprocessing-records.json"))
        if not matches:
            parser.error("No full-preprocessing records found")
        date = matches[-1].parent.name
    verifier = Verifier(date)
    verifier.guarded("verification", verifier.verify)
    result = verifier.result()
    if verifier.directory.is_dir():
        (verifier.directory / "full-preprocessing-verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (verifier.directory / "FULL_PREPROCESSING_REPORT.md").write_text(verifier.report(result), encoding="utf-8")
    print(json.dumps({key:result[key] for key in ["status", "passed_checks", "failed_checks", "checked_runs"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
