"""Copy small experiment records to reports/ for sharing through Git."""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*', type=Path, help='Run folders or metrics.json files')
    args = parser.parse_args()
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    for item in args.runs:
        source = item if item.is_file() else item / 'metrics.json'
        m = json.loads(source.read_text(encoding='utf-8'))
        allowed = ['task', 'model', 'split', 'seed', 'categorical', 'scores', 'seconds',
                   'python', 'sklearn', 'pandas', 'numpy', 'input_sha256', 'source_commit',
                   'script_sha256', 'model_family', 'model_parameters', 'hypothesis', 'catboost', 'preprocessing', 'optuna', 'tuning']
        record = {k: m[k] for k in allowed if k in m}
        record['run_id'] = source.parent.name
        record['kind'] = 'local_fixed_pipeline'
        (reports / f'{source.parent.name}.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        details = source.parent / 'classification_report.txt'
        if details.exists():
            shutil.copyfile(details, reports / f'{source.parent.name}-classes.txt')
        trials = source.parent / 'trials.csv'
        if trials.exists():
            shutil.copyfile(trials, reports / f'{source.parent.name}-trials.csv')
    from build_experiment_registry import main as rebuild_registry
    rebuild_registry()
    print(f'Updated {reports}; no Git commands or network requests executed.')


if __name__ == '__main__':
    main()
