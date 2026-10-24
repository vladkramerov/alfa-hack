"""Выгрузка итогового решения как самостоятельного кода: папка solution_code/ и архив solution_code.zip
в рабочей папке агента.

Внутри: копия ML-ядра агента (agent/ml, schemas, finalize — без LLM и цикла исследования),
solution.json с найденным решением (признаки, чистка, сгенерированные признаки, код признаков от LLM,
модели, параметры, веса ансамбля, постобработка, схема валидации) и train_predict.py, который на
новых train/test той же структуры заново обучает именно это решение и пишет предсказания.
Так решение можно воспроизвести без агента, без LLM и без 3 часов поиска, перенести на похожую
задачу (те же колонки) или разобрать и доработать руками."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

import ag_ml_metrics as M

log = logging.getLogger("agent.export")

ROOT = Path(__file__).resolve().parent
COPY = sorted(p.name for p in ROOT.glob("ag_*.py")
              if p.name in ("ag_schemas.py", "ag_finalize.py") or p.name.startswith("ag_ml_"))

RUNNER = '''"""Повторяет решение, найденное агентом, на новых данных той же структуры.

    python train_predict.py --train train.csv --test test.csv --out submission.csv [--sample sample_submission.csv]

Решение описано в solution.json: его можно править руками (параметры моделей, состав ансамбля,
блоки признаков) и запускать снова. Зависимости: pandas, numpy, scikit-learn, lightgbm / xgboost /
catboost (те, что использованы в solution.json), optuna не нужна."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ag_finalize as F  # noqa: E402
import ag_ml_dataset as DS, ag_ml_metrics as M, ag_ml_structure as structure  # noqa: E402
from ag_ml_pipeline import run_pipeline  # noqa: E402
from ag_ml_validation import fold_groups, make_folds  # noqa: E402
from ag_schemas import PipelineSpec, TaskSpec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--sample", default=None, help="sample_submission.csv: порядок строк и колонок")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--solution", default=str(HERE / "solution.json"))
    a = ap.parse_args()

    sol = json.loads(Path(a.solution).read_text(encoding="utf-8"))
    task = TaskSpec(**sol["task"])
    if sol.get("cost_matrix"):
        M.register_cost(sol["cost_matrix"], task.metric)
    if not M.ensure(task.metric):   # своя метрика пользователя не выгружается — считаем стандартную
        task.metric = M.default_metric(task.task_type)
    M.configure(task)
    train = pd.read_csv(a.train, low_memory=False)
    test = pd.read_csv(a.test, low_memory=False)
    train = train[train[task.targets or [task.target]].notna().all(axis=1)].reset_index(drop=True)
    missing = [c for c in sol["feature_cols"] if c not in train.columns or c not in test.columns]
    if missing:
        sys.exit(f"в данных нет колонок решения: {missing[:10]} — это другая структура, запустите агента заново")
    ds = DS.build(task, train, test, sol["feature_cols"])
    y = ds.y
    st = sol["structure"]
    folds = None
    if st.get("period"):
        is_dt = st["period"] in ds.datetime_cols
        ds.period = np.concatenate([structure.period_values(train, st["period"], is_dt),
                                    structure.period_values(test, st["period"], is_dt)])
        if st.get("entity_key"):
            both = pd.concat([train[st["entity_key"]], test[st["entity_key"]]], ignore_index=True)
            ds.entity = pd.factorize(both.astype(str).agg("|".join, axis=1))[0]
        folds = structure.time_folds(ds.period[:len(y)], sol["n_folds"],
                                     test_share=len(test) / max(len(y), 1) if ds.entity is not None else None,
                                     rare=structure.rare_class_mask(y, task.is_classification))
        if len(folds) >= 2:
            ds.refit = True
        else:
            folds = None
    if folds is None and st.get("group_time"):
        gt = st["group_time"]
        per = structure.period_values(train, gt["time"], gt["time"] in ds.datetime_cols)
        folds = structure.group_time_folds(per, pd.factorize(train[gt["group"]])[0], gt["ratio"], sol["n_folds"])
        if len(folds) >= 2:
            ds.refit = True
        else:
            folds = None
    if folds is None:
        folds = make_folds(task, y, sol["n_folds"], sol["seed"], fold_groups(task, ds))

    oof, pred = 0.0, 0.0
    for m in sol["members"]:
        spec = PipelineSpec.from_dict(m["spec"])
        out = run_pipeline(ds, task, spec, folds, a.n_jobs, sol["seed"])
        print(f"{spec.model}: CV {task.metric} = {out['cv_mean']:.5f}, вес {m['weight']}")
        oof = oof + m["weight"] * out["oof"]
        pred = pred + m["weight"] * np.asarray(out["test_pred"])
    covered = np.concatenate([va for _, va in folds])
    print(f"итог: CV {task.metric} = {M.score(M.METRICS[task.metric], y[covered], oof[covered], sol['post']):.5f}")
    sample = pd.read_csv(a.sample) if a.sample else None
    df = F.predictions_frame(task, ds.test_ids, pred, sol["post"], len(test))
    F.write_submission(df, Path(a.out), sample, task)
    print("предсказания:", a.out)


if __name__ == "__main__":
    main()
'''


def _plain(v):
    """numpy-типы -> обычные Python, чтобы классы и параметры корректно легли в JSON."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def export_solution(work_dir: Path, task, solution, memory, structure: dict, feature_cols: list[str],
                    datetime_cols: list[str], n_folds: int, seed: int) -> Path:
    out = work_dir / "solution_code"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, out / "agent" / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, out / name)
    members = []
    for exp_id, w in solution.members.items():
        r = memory.get(int(exp_id))
        spec = r.hypothesis.spec.to_dict()
        spec.update(tune_trials=0, tune_seconds=0.0, feature_search=None, search_seconds=0.0)
        members.append({"exp_id": int(exp_id), "weight": float(w), "hypothesis": r.hypothesis.text,
                        "cv": float(r.cv_mean), "spec": spec})
    t = {k: getattr(task, k) for k in ("task_type", "target", "id_col", "metric", "submission_columns",
                                       "proba_output", "classes", "targets", "group_col", "ordinal",
                                       "positive_class", "column_types")}
    sol = {"task": t, "feature_cols": list(feature_cols), "datetime_cols": list(datetime_cols),
           "cost_matrix": M.METRICS[task.metric].args.get("cost_matrix") if task.metric in M.METRICS else None,
           "structure": {"period": structure.get("period"), "entity_key": structure.get("entity_key")},
           "n_folds": n_folds, "seed": seed, "members": members, "post": solution.post,
           "cv": float(solution.cv), "name": solution.name}
    (out / "solution.json").write_text(json.dumps(_plain(sol), ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "train_predict.py").write_text(RUNNER, encoding="utf-8")
    (out / "README.md").write_text(
        "# Решение, найденное агентом\n\n"
        f"Задача: {task.task_type}, таргет `{task.target}`, метрика {task.metric}. "
        f"Итог: {solution.name}, CV = {solution.cv:.5f}.\n\n"
        "```bash\npython train_predict.py --train train.csv --test test.csv --out submission.csv\n```\n\n"
        "- `solution.json` — всё решение: признаки, чистка, сгенерированные признаки (`gen_features`), "
        "код признаков от LLM (`feature_code`), модели и их параметры, веса ансамбля, постобработка, схема валидации. "
        "Его можно править руками.\n"
        "- `ag_*.py` — ML-ядро агента (предобработка, признаки, модели, валидация), без LLM и цикла поиска.\n"
        "- Подходит для данных с теми же колонками (новый период, другая выборка). Для задачи с другой "
        "структурой нужно запустить агента заново: он найдёт решение под неё.\n", encoding="utf-8")
    zip_path = Path(shutil.make_archive(str(work_dir / "solution_code"), "zip", root_dir=out))
    log.info("код решения выгружен: %s", zip_path)
    return zip_path
