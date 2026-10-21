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
                   'script_sha256', 'model_family', 'model_parameters', 'hypothesis']
        record = {k: m[k] for k in allowed if k in m}
        record['run_id'] = source.parent.name
        record['kind'] = 'local_fixed_pipeline'
        (reports / f'{source.parent.name}.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        details = source.parent / 'classification_report.txt'
        if details.exists():
            shutil.copyfile(details, reports / f'{source.parent.name}-classes.txt')
    lines = ['# Журнал экспериментов', '',
             'Метрики получены на локальной валидации, не на скрытом test. LLM не используется.', '',
             '| Запуск | Задача | Модель | RMSE ↓ / Macro F1 ↑ | Время, сек |',
             '|---|---|---|---:|---:|']
    for path in sorted(reports.glob('*.json')):
        m = json.loads(path.read_text(encoding='utf-8'))
        if m.get('kind') != 'local_fixed_pipeline':
            continue
        metric = 'rmse' if m['task'] == 'regression' else 'macro_f1'
        score = m['scores'][metric]
        lines.append(f'| [{m["run_id"]}]({path.name}) | {m["task"]} | {m["model"]} | {score:.6f} | {m["seconds"]:.2f} |')
    lines += ['', 'Сравнивайте одинаковые задачи, разделения и контрольные суммы данных.',
              'Время включает обучение на части train, валидацию, переобучение на всём train и прогноз test.',
              'Полные CSV остаются в runs/ и не включаются в Git. Для публикации reports/ нужен commit и push.', '']
    (reports / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Updated {reports}; no Git commands or network requests executed.')


if __name__ == '__main__':
    main()
