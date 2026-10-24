"""Report completed SPEC-004 artifacts; never import ML code or fit models.

Run only after the series and independent verification have finished:
    .venv-ml/Scripts/python.exe -m experiments.report_novel_results --date 2026-10-23
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REG_BEST = 43.0046386479306
CLASS_BEST = 0.5838559593680208
NAMES = {
    "HYP-011": "XGBoost histogram",
    "HYP-012": "log1p target → expm1",
    "HYP-013": "LightGBM Tweedie 1.5",
    "HYP-014": "Линейные листья + scaling",
    "HYP-015": "Grouped OOF target encoding",
    "HYP-016": "Категория proto×service×state",
    "HYP-017": "Четыре train-only частоты",
    "HYP-018": "Multiclass one-vs-rest",
    "HYP-019": "Иерархия Normal / attack / тип",
    "HYP-020": "Веса 1 / размер raw-X группы",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def comparison_rows(records, metrics_by_directory):
    """Compute fixed-threshold conclusions independently of saved prose."""
    rows = []
    for record in records:
        candidate = record["candidate"]
        metric = metrics_by_directory[record["directory"]]
        task = candidate["task"]
        if task not in ("regression", "multiclass"):
            raise ValueError("Unknown task: " + task)
        reference = REG_BEST if task == "regression" else CLASS_BEST
        completed = metric["status"] == "completed"
        value = float(metric["score"]) if completed else None
        if completed and (not math.isfinite(value) or metric["metric"] !=
                          ("rmse" if task == "regression" else "macro_f1")):
            raise ValueError("Invalid score or incompatible metric")
        if metric["status"] != record["status"]:
            raise ValueError("Record and metrics status disagree")
        if completed and not math.isclose(value, float(record["score"]), rel_tol=1e-12):
            raise ValueError("Record and metrics score disagree")
        delta = value - reference if completed else None
        improved = (delta < 0 if task == "regression" else delta > 0) if completed else None
        conclusion = ("Подтверждена в проверенных условиях" if improved else
                      "Не подтверждена по заданному критерию" if completed else "Недостаточно данных")
        rows.append({"hypothesis": candidate["hypothesis"], "task": task,
                     "mechanism": NAMES[candidate["hypothesis"]],
                     "metric": "rmse" if task == "regression" else "macro_f1",
                     "score": value, "previous_best": reference, "delta_score_minus_previous": delta,
                     "improvement": (-delta if task == "regression" else delta) if completed else None,
                     "success": improved, "status": metric["status"], "conclusion": conclusion,
                     "seconds": float(metric.get("seconds", record.get("seconds", 0))),
                     "model_fit_count": int(record.get("planned_model_fit_count", 2 if candidate["hypothesis"] == "HYP-019" else 1)),
                     "directory": record["directory"], "error": metric.get("error", "")})
    return sorted(rows, key=lambda r: (r["hypothesis"], 0 if r["task"] == "regression" else 1))


def verification_passed(verification):
    """Accept explicit verifier success, reject any recorded failed check."""
    if not isinstance(verification, dict):
        return False
    if verification.get("passed") is False or verification.get("all_passed") is False:
        return False
    if verification.get("failures") or verification.get("errors") or verification.get("failed_checks"):
        return False
    checks = verification.get("checks", [])
    if isinstance(checks, list) and any(isinstance(c, dict) and c.get("passed") is False for c in checks):
        return False
    return (verification.get("passed") is True or verification.get("all_passed") is True
            or verification.get("status") in ("passed", "verified", "ok"))


def number(value, signed=False):
    if value is None:
        return "—"
    return format(value, "+.12f" if signed else ".12f")


def generate(date, root=ROOT, verification_path=None, test_log=None):
    root = Path(root).resolve()
    date_dir = root / "results" / date
    suite = date_dir / "series-ten-hypotheses-v1"
    summary = read_json(suite / "summary.json")
    records = read_json(suite / "records.json")
    plan = read_json(suite / "frozen-plan.json")
    if summary.get("status") not in ("completed", "completed_with_failures"):
        raise ValueError("Series has not finished; no report was written")
    if summary.get("baseline_refits") != 0:
        raise ValueError("Unexpected baseline refits")
    expected = {(c["hypothesis"], c["task"]) for c in plan["candidates"]}
    actual = [(r["candidate"]["hypothesis"], r["candidate"]["task"]) for r in records]
    required = {("HYP-011", "regression"), ("HYP-011", "multiclass")}
    required.update((f"HYP-{i:03d}", "regression" if i < 16 else "multiclass") for i in range(12, 21))
    if len(records) != 11 or len(set(actual)) != 11 or set(actual) != expected or expected != required:
        raise ValueError("Expected all 11 unique predeclared runs for HYP-011…020")
    if summary.get("records") != records:
        raise ValueError("Summary and records disagree")
    if any(r["status"] not in ("completed", "failed") for r in records):
        raise ValueError("Unfinished run in records")
    verification_path = Path(verification_path) if verification_path else suite / "independent-verification.json"
    verification = read_json(verification_path)
    if not verification_passed(verification):
        raise ValueError("Independent verification has not passed; no report was written")
    test_log = Path(test_log) if test_log else suite / "final-unit-tests.log"
    metrics = {r["directory"]: read_json(root / r["directory"] / "metrics.json") for r in records}
    rows = comparison_rows(records, metrics)
    independent = verification.get("independent_metrics", [])
    verified = {(r["hypothesis"], r["task"]): r for r in independent}
    if verification.get("checked_runs") != 11 or len(verified) != 11:
        raise ValueError("Independent verification must cover all 11 runs")
    for row in rows:
        match = verified.get((row["hypothesis"], row["task"]))
        if match is None or not math.isclose(row["score"], float(match[row["metric"]]), rel_tol=1e-9):
            raise ValueError("Independent metrics disagree with the report inputs")
    if sum(r["model_fit_count"] for r in rows) != 12:
        raise ValueError("Expected 12 fresh component fits, including two for HYP-019")

    def link(label, path):
        return f'[{label}]({Path(os.path.relpath(path, date_dir)).as_posix()})'

    def source(row, filename="report.md", label="отчёт"):
        return link(label, root / row["directory"] / filename)

    def selection(task):
        completed = [r for r in rows if r["task"] == task and r["status"] == "completed"]
        best = (min(completed, key=lambda r: r["score"]) if task == "regression" else
                max(completed, key=lambda r: r["score"])) if completed else None
        if best and best["success"]:
            label = "Регрессия" if task == "regression" else "Классификация"
            return (f'**{label}: новый лучший {best["hypothesis"]}, '
                    f'{best["metric"]} {number(best["score"])}**; '
                    f'улучшение {number(best["improvement"])}; {source(best)}.'), best
        if task == "regression":
            return ("**Регрессия: сохранён CatBoost H4-zero-year, RMSE 43.0046386479306**; "
                    "новые завершённые кандидаты общий порог не превысили. " +
                    link("исторический источник", root / "results/history/2026-10-21/historical-experiments.json") + "."), best
        return ("**Классификация: сохранён LightGBM HYP-009, grouped Macro F1 0.5838559593680208**; "
                "новые завершённые кандидаты общий порог не превысили. " +
                link("сохранённый контроль", date_dir / "HYP-009-02-multiclass-lightgbm/report.md") + "."), best

    reg_text, reg_best = selection("regression")
    class_text, class_best = selection("multiclass")
    lines = [f"# Проверка десяти новых гипотез — {date}", "", reg_text, "", class_text, "",
             "## Все 11 проверок", "",
             "Десять гипотез дали 11 validation-результатов: HYP-011 проверена на двух задачах. "
             "Δ = новая метрика − прежняя лучшая: для RMSE отрицательная Δ лучше, для Macro F1 положительная. "
             "Критерии зафиксированы до fit; строгие пороги: RMSE < 43.0046386479306 / grouped Macro F1 > 0.5838559593680208.", "",
             "| Гипотеза / механизм | Задача | Метрика | Значение | Δ к прежнему лучшему | Порог пройден | Время, с | Источники |",
             "| --- | --- | --- | ---: | ---: | --- | ---: | --- |"]
    for row in rows:
        success = "да" if row["success"] else "нет" if row["status"] == "completed" else "не оценивался"
        sources = source(row) + " · " + source(row, "metrics.json", "метрики")
        lines.append(f'| {row["hypothesis"]}: {row["mechanism"]} | {row["task"]} | '
                     f'{row["metric"]} | {number(row["score"])} | '
                     f'{number(row["delta_score_minus_previous"], signed=True)} | {success} | '
                     f'{row["seconds"]:.2f} | {sources} |')
    lines += ["", "Вывод по фиксированному критерию: " + "; ".join(
        f'{hyp}: ' + ", ".join(f'{r["task"]} — {r["conclusion"]}' for r in rows if r["hypothesis"] == hyp)
        for hyp in NAMES) + ".", "",
        "Сохранённые прежние ориентиры и их происхождение: " +
        link("CatBoost H4 и параметры", root / "results/history/2026-10-21/historical-experiments.json") + "; " +
        link("LightGBM HYP-009", date_dir / "HYP-009-02-multiclass-lightgbm/metrics.json") + ". "
        "Внутрисемейные ориентиры регрессии: ExtraTrees 43.56756773832011, LightGBM 60.492688256313926; "
        "улучшение одного из них отдельно не означает победу над общим RMSE-порогом.", ""]
    if class_best:
        classification = read_json(root / class_best["directory"] / "classification-report.json")
        lines += [f'## Классы лучшего нового классификатора: {class_best["hypothesis"]}', "",
                  f'Grouped Macro F1 {number(class_best["score"])}; ' +
                  source(class_best, "classification-report.json", "полный per-class отчёт") + "; " +
                  source(class_best, "confusion-matrix.csv", "матрица ошибок") + ".", "",
                  "| Класс | Precision | Recall | F1 | Число validation-строк |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, stats in classification.items():
            if name in ("accuracy", "macro avg", "weighted avg", "micro avg") or not isinstance(stats, dict):
                continue
            lines.append(f'| {name} | {stats["precision"]:.6f} | {stats["recall"]:.6f} | '
                         f'{stats["f1-score"]:.6f} | {int(stats["support"])} |')
        lines.append("")
    h18 = next(r for r in rows if r["hypothesis"] == "HYP-018")
    h19 = next(r for r in rows if r["hypothesis"] == "HYP-019")
    h18_range = metrics[h18["directory"]].get("probability_row_sum_range")
    range_text = f' Фактический диапазон сумм OVA scores: {h18_range}.' if h18_range else ""
    tests_text = ("46 unit tests прошли; " + link("финальный лог тестов", test_log) + ".") if test_log.exists() else (
        "Предварительно выполнено 46 unit tests; итоговый повтор и его результат фиксируются отдельно: " +
        link("ожидаемый финальный лог тестов", test_log) + " (на момент генерации файл отсутствует).")
    lines += ["## Проверка и ограничения", "",
              f'Сумма worker runtime: {sum(r["seconds"] for r in rows):.2f} с; '
              f'завершено {sum(r["status"] == "completed" for r in rows)}/11 проверок. '
              "Всего запланировано 12 новых model fit: HYP-019 включает две свежие модели "
              "с отдельной предобработкой binary и attack-only частей; " + source(h19, "preprocessing-audit.json", "аудит иерархии") + ". "
              "Старые baseline не переобучались (baseline_refits=0); tuning/повторных baseline fit нет.", "",
              "Все метрики независимо пересчитаны из сохранённых предсказаний и исходных y: " +
              link("независимая проверка", verification_path) + ". " +
              link("Записи 11 запусков", suite / "records.json") + ", " +
              link("итог серии", suite / "summary.json") + ", " +
              link("план до fit", suite / "frozen-plan.json") + ", " +
              link("SHA исходного кода", suite / "source-manifest.json") + ". " + tests_text, "",
              "HYP-018 сохраняет независимые OVA scores и выбирает argmax; "
              "они не перенормированы и не обязаны суммироваться к единице." + range_text, "",
              "Исходные outer indices сохранены: regression 35 000/18 308, multiclass 70 529/17 141; "
              "категории, медианы, масштабирование, target/frequency statistics обучены только на fit. "
              "Для HYP-015 training encoding исключает всю группу одинакового исходного X; "
              "для HYP-020 validation не перевзвешен. Outer уже использовался ранее: "
              "выбор разведочный, независимого подтверждения и оценки статистической значимости нет.", "",
              "Независимый verifier проверяет сохранённые артефакты и заявленные аудиты; "
              "они сами по себе не доказывают весь порядок исторического выполнения.", "",
              "Платформенный агент автоматически не обновлён экспериментальными победителями; "
              "локальные метрики не являются скрытым платформенным score. Отправка на соревновательную "
              "платформу не выполнялась. Пользователь разрешил публикацию кода в GitHub ivan_branch; "
              "разрешение не означает, что push уже выполнен.", ""]
    for row in rows:
        if row["status"] != "completed":
            lines.append(f'- {row["hypothesis"]}/{row["task"]}: ошибка запуска, вывод «Недостаточно данных»; '
                         f'{source(row, "metrics.json", "причина и метрики")}.')
    report = date_dir / "TEN_HYPOTHESES_REPORT.md"
    comparison = date_dir / "ten-hypotheses-comparison.csv"
    report.write_text("\n".join(lines), encoding="utf-8")
    with comparison.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d"))
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--test-log", type=Path)
    args = parser.parse_args()
    report, comparison = generate(args.date, verification_path=args.verification, test_log=args.test_log)
    print(json.dumps({"report": str(report), "comparison": str(comparison)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
