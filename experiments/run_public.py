"""Reproduce fixed baseline vs adaptive agent on both supplied public tasks."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from package_solution import build, ROOT


def run(data_root, work_root, output, budget=600, experiments=10):
    manifest = build()
    data_root, work_root = Path(data_root), Path(work_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task, folder in [('regression', 'Регрессия'), ('multiclass', 'Многоклассовая классификация')]:
        for mode, trials in [('baseline', 1), ('adaptive', experiments)]:
            directory = work_root / f'{task}-{mode}'
            if directory.exists():
                raise ValueError(f'Refusing to overwrite an experiment: {directory}')
            directory.mkdir(parents=True)
            for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
                shutil.copy2(data_root / folder / name, directory / name)
            with zipfile.ZipFile(ROOT / 'dist' / 'alfa_autoresearch_submission.zip') as archive:
                archive.extractall(directory)
            env = {k: v for k, v in os.environ.items() if not k.startswith('OPENAI_')}
            env.update(AUTOML_TIME_LIMIT=str(budget), AUTOML_MAX_EXPERIMENTS=str(trials),
                       AUTOML_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                       MPLCONFIGDIR=str(work_root / 'matplotlib'))
            started = time.monotonic()
            with (directory / 'cli.log').open('w') as log:
                process = subprocess.run([sys.executable, 'solution.py', 'task.md'], cwd=directory,
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=budget + 20)
            runs = sorted((directory / 'agent_runs').glob('*'))
            if process.returncode or not runs or not (runs[-1] / 'summary.json').is_file():
                raise RuntimeError(f'Public run failed: {directory}; inspect cli.log')
            source = runs[-1]
            copied = output / f'{task}-{mode}'
            copied.mkdir()
            for name in ('summary.json', 'experiments.json', 'validation.json', 'audit.json', 'llm.json', 'frozen_selection.json'):
                if (source / name).is_file():
                    shutil.copy2(source / name, copied / name)
            record = {'task': task, 'mode': mode, 'process_returncode': process.returncode,
                'wall_seconds': time.monotonic() - started, 'archive_sha256': manifest['sha256'],
                'summary': json.loads((source / 'summary.json').read_text()),
                'workspace': str(directory), 'result_directory': str(copied.relative_to(ROOT))}
            records.append(record)
            (output / 'runs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'task': task, 'mode': mode, 'seconds': record['wall_seconds'],
                'search_score': record['summary'].get('selected_search_score'),
                'audit': [(r.get('candidate', {}).get('family'), r.get('score')) for r in record['summary'].get('audit', [])],
                'status': record['summary']['status']}, ensure_ascii=False), flush=True)
    report = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.version, 'archive': manifest, 'runs': records,
              'real_llm_api': False, 'validation': 'nested search; fixed baseline/selected audit gate',
              'limitation': 'Outer public folds have historical exposure. No hidden test labels or Boosters score.'}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--output', default=str(ROOT / 'results' / 'public-v1'))
    parser.add_argument('--budget', type=int, default=600)
    parser.add_argument('--experiments', type=int, default=10)
    args = parser.parse_args()
    run(args.data_root, args.work_root, args.output, args.budget, args.experiments)
