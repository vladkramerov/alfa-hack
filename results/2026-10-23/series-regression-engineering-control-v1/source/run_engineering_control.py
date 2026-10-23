"""Exploratory HYP-008: retain engineered predictors and reuse fixed baselines.

No primary-series selection or final model is changed. Only engineer=True is
enabled; the four model/stage comparisons are frozen before the first fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from experiments.preprocessing import DataPreprocessor
from experiments.run_suite import (ExperimentSuite, RECIPES, ROOT, build_model,
                                   now, save_json)


TAG = "engineering-control-v1"
ORIGINAL_TAG = "preprocessing-v1"
POLL_SECONDS = 30
NOTICE = ("Exploratory analysis after inspecting primary-series inner diagnostics and part of regression outer; "
          "reused validation splits, not an independent confirmatory test; primary frozen winner is unchanged.")


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def smoke_check():
    """Verify the control actually retains new features without fitting models."""
    toy = pd.DataFrame({"sbytes": [0, 10, 20], "dbytes": [5, 10, 15],
                        "spkts": [0, 1, 2], "dpkts": [1, 2, 3]})
    prep = DataPreprocessor("multiclass", engineer=True)
    values = prep.fit_transform(toy)
    expected = {"src_bytes_per_packet", "dst_bytes_per_packet", "sbytes_log1p",
                "dbytes_log1p", "spkts_log1p", "dpkts_log1p"}
    if not expected.issubset(set(prep.get_feature_names_out())) or not np.isfinite(values).all():
        raise AssertionError("Engineering control does not preserve its new finite features")
    if prep.audit_["dropped_columns"] or prep.audit_["vif_dropped_columns"]:
        raise AssertionError("Selection must be disabled in engineering-only control")
    return {"status": "passed", "new_feature_names": sorted(expected), "shape": list(values.shape),
            "model_fit_performed": False}


class EngineeringControlSuite(ExperimentSuite):
    def write_report(self, directory, record):
        gain = record.get("gain")
        successful = record.get("status") == "completed" and record.get("converged", True)
        conclusion = ("Подтверждена в проверенных условиях" if successful and gain is not None and gain > 0
                      else "Не подтверждена по заданному критерию" if successful and gain is not None
                      else "Недостаточно данных")
        text = (f'# {record["run_id"]}\n\n'
                f'- Начало: {record["started"]}; окончание: {record.get("finished")}.\n'
                f'- Статус: {record["status"]}; вывод: {conclusion}.\n'
                '- Гипотеза: [HYP-008](../../../docs/hypotheses/HYP-008.md).\n'
                '- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md).\n'
                f'- Задача: {self.task}; stage: {record["stage"]}.\n'
                f'- Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_engineering_control '
                f'--task {self.task} --threads {self.threads}`.\n'
                f'- Контроль: {record.get("reference")}; выигрыш utility: {gain}.\n\n'
                'Единственное семейство изменений относительно исходного baseline — engineer=True; '
                'correlation/VIF/scaling/Yeo–Johnson выключены, новые признаки сохранены.\n'
                'Критерий до запуска: RMSE меньше baseline / Macro F1 больше baseline на тех же строках.\n\n'
                'Разведочный анализ после просмотра основной inner-диагностики и части regression outer; '
                'splits используются повторно, независимого подтверждения нет. Основной frozen winner не меняется.\n\n'
                '[Метаданные и параметры](metrics.json), [аудит признаков](preprocessing_audit.json), '
                '[предсказания validation](validation_predictions.csv).\n')
        (directory / "report.md").write_text(text, encoding="utf-8")


def snapshot_sources(suite):
    destination = suite.directory / "source"
    destination.mkdir()
    hashes = {}
    for relative, expected in suite.environment["code_sha256"].items():
        path = ROOT / Path(relative.replace("\\", "/"))
        output = destination / path.name
        shutil.copyfile(path, output)
        actual = hashlib.sha256(output.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source changed before snapshot: {relative}")
        hashes[relative] = actual
    return hashes


def load_reference(original, stage, family):
    """Read only existing baselines; a partially written records file is retried."""
    try:
        records = json.loads(read_json(original / "records.json"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    matches = [record for record in records if record["stage"] == stage and record["config"] ==
               {"family": family, "recipe": "baseline"}]
    if matches:
        if matches[0]["status"] != "completed":
            raise RuntimeError(f"Original baseline did not complete: {matches[0]['run_id']}")
        if len(matches) != 1:
            raise RuntimeError("Ambiguous original baseline references")
        return matches[0]
    if (original / "summary.json").is_file():
        raise RuntimeError(f"Completed original series lacks {stage}/{family} baseline")
    return None


def verify_reference(suite, reference, stage, family):
    if reference["environment"]["data_sha256"] != suite.environment["data_sha256"]:
        raise RuntimeError("Control and baseline source data differ")
    if reference["model_parameters"] != build_model(suite.task, {"family": family}, suite.threads).get_params():
        raise RuntimeError("Control and original baseline model parameters differ")
    if reference["n_fit"] != len(suite.splits[stage + "_fit"]) or reference["n_validation"] != len(suite.splits[stage + "_valid"]):
        raise RuntimeError("Control and original baseline sample counts differ")


def run_control(task, threads):
    smoke = smoke_check()
    print(f"{now()} smoke: {smoke}", flush=True)
    RECIPES["engineering_only"] = {"engineer": True}
    suite = EngineeringControlSuite(task, threads, TAG)
    original = suite.root / f"series-{task}-{ORIGINAL_TAG}"
    try:
        snapshots = snapshot_sources(suite)
        with np.load(original / "splits.npz", allow_pickle=False) as arrays:
            for key in ("inner_fit", "inner_valid", "outer_fit", "outer_valid"):
                if not np.array_equal(suite.splits[key], arrays[key]):
                    raise RuntimeError(f"Control split differs from original: {key}")
        panel = [{"stage": stage, "config": {"family": family, "recipe": "engineering_only"},
                  "reference_selector": {"stage": stage, "config": {"family": family, "recipe": "baseline"}}}
                 for stage in ("inner", "outer") for family in ("catboost", "forest")]
        frozen = {"frozen_at": now(), "hypothesis": "HYP-008", "notice": NOTICE,
                  "recipe": {"engineer": True}, "panel": panel,
                  "original_series": str(original.relative_to(ROOT)),
                  "primary_winner_changes_allowed": False, "smoke": smoke, "source_sha256": snapshots}
        save_json(suite.directory / "frozen_engineering_control.json", frozen)
        supplemental = None
        for panel_index, item in enumerate(panel):
            if panel_index == 2:
                baseline_inner = [load_reference(original, "inner", family) for family in ("catboost", "forest")]
                if any(record is None for record in baseline_inner):
                    raise RuntimeError("Original inner baselines disappeared before supplemental freeze")
                candidates = baseline_inner + list(suite.records)
                best = max(candidates, key=suite.utility)
                supplemental = {"frozen_at": now(), "notice": NOTICE,
                                "selection": "inner only across original baseline and engineering-only for both families",
                                "candidates": [{key: record[key] for key in ("run_id", "config", "scores")}
                                               for record in candidates],
                                "selected_run_id": best["run_id"], "selected_config": best["config"],
                                "primary_winner_changed": False}
                save_json(suite.directory / "frozen_supplemental_selection.json", supplemental)
                print(f"{now()} {task} supplemental inner-only selection frozen: {best['config']} {best['scores']}", flush=True)
            stage, family = item["stage"], item["config"]["family"]
            reference = load_reference(original, stage, family)
            while reference is None:
                print(f"{now()} {task}: awaiting original {stage}/{family} baseline; no fit started", flush=True)
                time.sleep(POLL_SECONDS)
                reference = load_reference(original, stage, family)
            verify_reference(suite, reference, stage, family)
            record = suite.evaluate(item["config"], stage, hypothesis="HYP-008", reference=reference)
            prep = suite.cache[(stage, "engineering_only")][0]
            new_numeric = [column for column in prep.audit_["numeric_columns"] if column not in suite.X.columns]
            if not new_numeric:
                raise RuntimeError("Engineering-only produced no newly retained numeric features")
            save_json(suite.directory / f"{stage}-new-feature-names.json", new_numeric)
            print(f"{now()} {task}/{stage}/{family}: retained {len(new_numeric)} new numeric features {new_numeric}", flush=True)
        final = suite.refit_final(supplemental["selected_config"])
        final_directory = ROOT / Path(final["path"].replace("\\", "/"))
        save_json(final_directory / "supplemental-provenance.json", {
            "series": str(suite.directory.relative_to(ROOT)), "notice": NOTICE,
            "frozen_selection": supplemental, "primary_winner_changed": False,
            "technical_refit_not_new_validation": True})
        with (final_directory / "report.md").open("a", encoding="utf-8") as stream:
            stream.write("\nДополнение: это дополнительный разведочный кандидат HYP-008; основной frozen winner сохранён. "
                         "Выбор сделан по повторно используемому inner до новых outer-оценок; "
                         "качество hidden test неизвестно. [Происхождение](supplemental-provenance.json).\n")
        summary = {"task": task, "status": "completed", "completed_at": now(),
                   "directory": str(suite.directory.relative_to(ROOT)), "frozen": frozen,
                   "supplemental_selection": supplemental, "final": final,
                   "records": suite.records, "primary_winner_changed": False, "final_refit_performed": True,
                   "limitation": NOTICE}
        save_json(suite.directory / "engineering-control-summary.json", summary)
        print(f"{now()} {task} ENGINEERING CONTROL completed: {suite.directory}", flush=True)
        return summary
    except BaseException as exc:
        save_json(suite.directory / "engineering-control-status.json",
                  {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                   "finished": now(), "error": repr(exc), "records": suite.records, "notice": NOTICE})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("regression", "multiclass"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads != 2:
        parser.error("HYP-008 protocol fixes --threads 2 to match the original model parameters")
    run_control(args.task, args.threads)


if __name__ == "__main__":
    main()
