"""Единственное место, которое знает интерфейс платформы.

Интерфейс взят из baseline.py организаторов:
    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... uv run solution.py task.md
  - единственный аргумент — путь к Markdown-файлу с условием задачи;
  - относительные пути в условии отсчитываются от каталога task.md;
  - LLM — OpenAI-совместимый API из переменных OPENAI_*.
Куда писать предсказания, говорит само условие (task.md), поэтому путь выхода ищем в нём.

Порядок поиска:
  условие:  argv[1] | $TASK_PATH | ./task.md
  данные:   --data | каталог task.md (и вложенные на 1-2 уровня) | $DATA_DIR | ./data | cwd
  выход:    --output | $OUTPUT_PATH | путь .csv из task.md | <каталог задачи>/submission.csv
  рабочая:  --work-dir | $WORK_DIR | <каталог задачи>/.agent_work (если можно писать) | ./workspace
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_FILES = ("train.csv", "test.csv")
INPUT_FILES = {"train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"}
DEFAULT_OUTPUT_NAME = "submission.csv"


@dataclass
class TaskPaths:
    task_md: Path | None
    task_dir: Path
    data_dir: Path
    train: Path
    test: Path
    data_dictionary: Path | None
    sample_submission: Path | None
    description_text: str
    output_path: Path
    output_source: str            # откуда взят путь выхода: cli | env | task.md | default
    work_dir: Path
    extra_outputs: list[Path] = field(default_factory=list)


@dataclass
class LLMSettings:
    base_url: str | None
    api_key: str | None
    model: str


def _first_env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _locate(root: Path) -> Path | None:
    """Папка, где лежат train.csv и test.csv: сам root или вложенная на 1-2 уровня."""
    if not root.exists() or not root.is_dir():
        return None
    if all((root / f).exists() for f in REQUIRED_FILES):
        return root
    for pattern in ("*", "*/*"):
        for d in sorted(root.glob(pattern)):
            if d.is_dir() and all((d / f).exists() for f in REQUIRED_FILES):
                return d
    return None


_CSV_RE = re.compile(r"[`'\"«]?((?:[\w.\-]+/)*[\w.\-]+\.csv)[`'\"»]?", re.I)
_OUTPUT_HINT = re.compile(r"сохран|запиш|записа|выгруз|сформир|output|save|write|предсказани|prediction|submission|сабмит|ответ", re.I)


def find_output_in_task(text: str) -> str | None:
    """Путь файла предсказаний из условия: .csv, который не является входным файлом и
    упомянут рядом со словами «сохранить/записать/предсказания/submission»."""
    best: tuple[int, str] | None = None
    for line in text.splitlines():
        for m in _CSV_RE.finditer(line):
            name = m.group(1)
            if Path(name).name.lower() in INPUT_FILES:
                continue
            score = 2 if _OUTPUT_HINT.search(line) else 1
            if best is None or score > best[0]:
                best = (score, name)
    return best[1] if best else None


def find_data_paths_in_task(text: str) -> list[str]:
    """Каталоги входных файлов, если условие указывает их явно (например data/train.csv)."""
    dirs = []
    for m in _CSV_RE.finditer(text):
        p = Path(m.group(1))
        if p.name.lower() in INPUT_FILES and str(p.parent) not in (".", ""):
            dirs.append(str(p.parent))
    return dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AutoML-агент: решает ML-задачу end-to-end")
    ap.add_argument("task", nargs="?", help="путь к task.md (как в baseline.py)")
    ap.add_argument("--data", help="папка с train.csv/test.csv (если не рядом с task.md)")
    ap.add_argument("--output", help="куда записать файл предсказаний")
    ap.add_argument("--work-dir", help="рабочая папка для журналов")
    args, _unknown = ap.parse_known_args(argv)  # неизвестные флаги платформы не роняют запуск
    return args


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_probe"
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


def resolve_task(argv: list[str] | None = None) -> TaskPaths:
    args = parse_args(argv)

    # --- условие ---
    task_arg = args.task or _first_env("TASK_PATH")
    task_md: Path | None = None
    if task_arg:
        p = Path(task_arg)
        if p.is_dir():
            cands = sorted(p.glob("*.md"))
            task_md = cands[0] if cands else None
            task_dir = p
        else:
            task_md = p
            task_dir = p.parent
    elif Path("task.md").exists():
        task_md = Path("task.md")
        task_dir = Path(".")
    else:
        task_dir = Path(args.data or _first_env("DATA_DIR") or ".")
    task_dir = task_dir.resolve()
    description = task_md.read_text(encoding="utf-8", errors="ignore") if task_md and task_md.exists() else ""
    if not description:
        description = _first_env("TASK_DESCRIPTION") or ""

    # --- данные ---
    data_dir = None
    roots = [args.data] + [str(task_dir / d) for d in find_data_paths_in_task(description)] + \
            [str(task_dir), _first_env("DATA_DIR", "INPUT_DIR"), "data", "."]
    for r in roots:
        if r:
            data_dir = _locate(Path(r) if Path(r).is_absolute() else Path(r).resolve())
            if data_dir:
                break
    if data_dir is None:
        raise FileNotFoundError(f"не найдены train.csv и test.csv рядом с {task_dir} (укажите --data)")

    # --- выход ---
    extra: list[Path] = []
    if args.output:
        out, src = Path(args.output), "cli"
    elif _first_env("OUTPUT_PATH", "SUBMISSION_PATH"):
        out, src = Path(_first_env("OUTPUT_PATH", "SUBMISSION_PATH")), "env"
    else:
        from_md = find_output_in_task(description)
        if from_md:
            out, src = (Path(from_md) if Path(from_md).is_absolute() else task_dir / from_md), "task.md"
        else:
            out, src = task_dir / DEFAULT_OUTPUT_NAME, "default"
            # условие не назвало файл: дублируем под самыми частыми именами
            extra = [task_dir / "predictions.csv"]
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- рабочая папка ---
    if args.work_dir or os.environ.get("WORK_DIR"):
        work = Path(args.work_dir or os.environ["WORK_DIR"])
    elif _writable(task_dir / ".agent_work"):
        work = task_dir / ".agent_work"
    else:
        work = Path("workspace")
    work.mkdir(parents=True, exist_ok=True)

    dd = data_dir / "data_dictionary.csv"
    ss = data_dir / "sample_submission.csv"
    return TaskPaths(
        task_md=task_md, task_dir=task_dir, data_dir=data_dir,
        train=data_dir / "train.csv", test=data_dir / "test.csv",
        data_dictionary=dd if dd.exists() else None,
        sample_submission=ss if ss.exists() else None,
        description_text=description, output_path=out, output_source=src, work_dir=work, extra_outputs=extra,
    )


def llm_settings() -> LLMSettings:
    base = _first_env("OPENAI_BASE_URL", "LLM_BASE_URL", "OPENAI_API_BASE")
    return LLMSettings(
        base_url=base.rstrip("/") if base else None,
        api_key=_first_env("OPENAI_API_KEY", "LLM_API_KEY"),
        model=_first_env("OPENAI_MODEL", "LLM_MODEL") or "qwen/qwen3.8-27b",   # имя из инструкции организаторов; прокси всё равно подставит модель задачи
    )
