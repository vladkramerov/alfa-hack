"""Actual clean-flat-source SPEC010 CLI checks; synthetic fits only, no ZIP.

Run with .venv-ml/Scripts/python.exe -X utf8 experiments/verify_rewrite_cli.py.
All fixtures, copied sources, subprocess logs and independent checks are retained.
An existing output directory is refused to preserve evidence and avoid refits.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn import metrics


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_SHA256 = '9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def fixture(directory, kind):
    nested = directory / 'inputs' / 'unfamiliar'
    nested.mkdir(parents=True)
    n, m = 90, 12
    rng = np.random.default_rng(391)
    classes = ['001', 'NA'] if kind != 'multiclass_log_loss' else ['001', 'NA', 'zeta']
    count = len(classes)
    group = np.arange(n) % count
    # Mixed features with missing cells and unknown test categories; exact X
    # duplicates give classification a meaningful raw-X grouped split.
    train = pd.DataFrame({'opaque_key': [f'train-{i:03d}' for i in range(n)],
                          'signal': group * 10. + (np.arange(n) // count % 7) * .01,
                          'category': [classes[i] for i in group],
                          'partly_missing': np.where(np.arange(n) % 9 == 0, np.nan, group),
                          'empty_feature': [''] * n})
    if kind == 'regression_rmse':
        train['signal'] = rng.normal(size=n)
        train['outcome_strange'] = 4 * train['signal'] + .7 * train['signal'] ** 2
    else:
        train['outcome_strange'] = [classes[i] for i in group]
    identifiers = ['001', '+1', 'NA'] + [f'test-{i:03d}' for i in range(m - 3)]
    test_group = np.arange(m) % count
    test = pd.DataFrame({'opaque_key': identifiers, 'signal': test_group * 10. + .025,
                         'category': [classes[i] for i in test_group],
                         'partly_missing': test_group.astype(float), 'empty_feature': [''] * m})
    test.loc[m - 1, 'category'] = 'unseen_category'
    test.loc[0, 'partly_missing'] = np.nan
    if kind == 'regression_rmse':
        test['signal'] = rng.normal(size=m)
    if kind == 'multiclass_log_loss':
        # Deliberately differs from sorted learner class order.
        sample = pd.DataFrame({'opaque_key': identifiers, 'zeta': 0., '001': 0., 'NA': 0.})
    else:
        sample = pd.DataFrame({'opaque_key': identifiers, 'outcome_strange': 0 if kind != 'macro_f1' else '001'})
    train.to_csv(nested / 'learning_table.csv', index=False)
    test.to_csv(nested / 'unknown_rows.csv', index=False)
    sample.to_csv(nested / 'expected_format.csv', index=False)
    pd.DataFrame({'column': list(train.columns), 'description': list(train.columns)}).to_csv(nested / 'column_notes.csv', index=False)
    metric = 'RMSE' if kind == 'regression_rmse' else 'macro-F1' if kind == 'macro_f1' else 'log_loss'
    task = directory / 'task.md'
    task.write_text('\n'.join([
        'Task type: regression' if kind == 'regression_rmse' else 'Task type: classification',
        f'Primary metric: {metric}', 'Target column: outcome_strange', 'ID column: opaque_key',
        'Train: `inputs/unfamiliar/learning_table.csv`', 'Test: `inputs/unfamiliar/unknown_rows.csv`',
        'Sample: `inputs/unfamiliar/expected_format.csv`', 'Dictionary: `inputs/unfamiliar/column_notes.csv`',
    ]), encoding='utf8')
    return task, sample, classes


def verify_case(directory, sample, classes, kind):
    output = pd.read_csv(directory / 'submission.csv', dtype=str, keep_default_na=False)
    assert list(output) == list(sample), 'CSV schema or order changed'
    assert len(output) == 12, 'CSV test row count changed'
    assert output.opaque_key.tolist() == sample.opaque_key.tolist(), 'Lexical ID or order changed'
    assert output.opaque_key.tolist()[:3] == ['001', '+1', 'NA']
    run = directory / 'research_runs'
    history = json.loads((run / 'hypotheses.json').read_text(encoding='utf8'))
    assert len(history) == 2 and all(r['status'] == 'completed' for r in history), 'Expected actual evaluate and refit'
    assert {r['mode'] for r in history} == {'evaluate', 'refit'}
    assert all(r['ledger_sha256'] and r['fingerprint'] for r in history)
    assert all(r['recipe']['family'] == ('factorization_machine' if kind == 'regression_rmse' else 'mixed_neighbors') for r in history)
    final = json.loads((run / 'final.json').read_text(encoding='utf8'))
    assert final['status'] == 'completed' and final['train_rows'] == 90
    inner = json.loads((run / 'run-001' / 'metrics.json').read_text(encoding='utf8'))
    assert inner['status'] == 'completed' and inner['fit_rows'] + inner['valid_rows'] == 90
    diagnostic = inner['diagnostics']
    checks = {'schema_ids_row_count': True, 'two_completed_fit_claims': True,
              'full_refit_completed': True, 'inner_rows': inner['valid_rows']}
    with np.load(run / 'inner-split.npz', allow_pickle=False) as split:
        assert not set(split['fit']) & set(split['valid'])
        assert sorted([*split['fit'], *split['valid']]) == list(range(90))
        valid = split['valid'].copy()
    with np.load(run / 'run-001' / 'validation_predictions.npz', allow_pickle=False) as saved:
        y, prediction = saved['y'], saved['prediction']
        assert np.array_equal(saved['positions'], valid)
        assert diagnostic['rows'] == len(y)
        if kind == 'regression_rmse':
            measured = float(np.sqrt(metrics.mean_squared_error(y.astype(float), prediction)))
            residual = prediction.astype(float) - y.astype(float)
            assert np.allclose(diagnostic['residual_quantiles'], np.quantile(residual, [0, .1, .5, .9, 1]))
            assert np.isclose(diagnostic['bias'], residual.mean())
            assert sum(segment['rows'] for segment in diagnostic['target_segments']) == len(y)
            assert np.isfinite(output.outcome_strange.astype(float)).all()
        else:
            probabilities, order = saved['probabilities'], saved['classes'].astype(str)
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1)
            hard = order[probabilities.argmax(axis=1)]
            if kind == 'macro_f1':
                measured = metrics.f1_score(y, prediction, average='macro', zero_division=0)
                assert set(output.outcome_strange) <= set(classes)
                assert {'001', 'NA'} <= set(output.outcome_strange), 'Opaque labels not exercised in final output'
            else:
                measured = metrics.log_loss(y, prediction, labels=np.unique(y))
                values = output.drop(columns='opaque_key').astype(float).to_numpy()
                assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
                if kind == 'multiclass_log_loss':
                    assert list(output)[1:] == ['zeta', '001', 'NA']
                    assert np.allclose(values.sum(axis=1), 1)
                else:
                    assert values.shape == (12, 1)
            rows = diagnostic['per_class']
            labels = [row['class'] for row in rows]
            p, r, f, support = metrics.precision_recall_fscore_support(y, hard, labels=labels, zero_division=0)
            assert set(labels) == set(classes)
            for row, pp, rr, ff, ss in zip(rows, p, r, f, support):
                assert row['support'] == int(ss)
                assert np.allclose([row['precision'], row['recall'], row['f1']], [pp, rr, ff])
            confusion = {}
            for actual, predicted in zip(y, hard):
                if actual != predicted:
                    confusion[(str(actual), str(predicted))] = confusion.get((str(actual), str(predicted)), 0) + 1
            assert {(row['actual'], row['predicted']): row['count'] for row in diagnostic['confusion_pairs']} == confusion
        assert np.isfinite(measured) and np.isclose(measured, inner['score'], rtol=1e-12, atol=1e-12)
    checks.update(independent_inner_metric=float(measured), inner_diagnostics_recomputed=True,
                  history_fingerprints=[row['fingerprint'] for row in history],
                  submission_sha256=sha(directory / 'submission.csv'))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/2026-10-24/rewrite-clean-cli')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite or repeat fits: {destination}')
    destination.mkdir(parents=True)
    report = {'spec': 'SPEC010', 'status': 'started', 'synthetic_only': True, 'archive_created': False,
              'runtime': sys.version, 'cases': {}}
    write_json(destination / 'verification.json', report)
    (destination / 'report.md').write_text('# SPEC010 clean CLI verification\n\nStarted: synthetic fixtures only; no ZIP, no real-data fits.\n', encoding='utf8')
    try:
        source = destination / 'source'
        source.mkdir()
        files = [ROOT / 'rewrite_solution.py', *sorted(ROOT.glob('research_*.py'))]
        manifest = {}
        for original in files:
            name = 'solution.py' if original.name == 'rewrite_solution.py' else original.name
            tree = ast.parse(original.read_text(encoding='utf8'))
            for node in ast.walk(tree):
                imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                assert not any(part.startswith('automl_') for part in imports), f'Legacy dependency: {original}'
            shutil.copyfile(original, source / name)
            manifest[name] = {'original': str(original), 'sha256': sha(source / name)}
        requirements = ROOT / 'requirements.txt'
        assert sha(requirements) == REQUIREMENTS_SHA256, 'Provided requirements checksum mismatch'
        shutil.copyfile(requirements, source / 'requirements.txt')
        manifest['requirements.txt'] = {'original': str(requirements), 'sha256': sha(source / 'requirements.txt')}
        write_json(destination / 'source_manifest.json', manifest)
        environment = os.environ.copy()
        for key in list(environment):
            if 'API_KEY' in key.upper() or key.upper() in {'OPENAI_BASE_URL', 'OPENAI_MODEL', 'PYTHONPATH'}:
                environment.pop(key, None)
        temporary = destination / 'tmp'
        temporary.mkdir()
        environment.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary), PYTHONNOUSERSITE='1',
                           RESEARCH_LEDGER_PATH=str(ROOT / 'HYPOTHESES.md'), OMP_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', RESEARCH_THREADS='4')
        for kind in ('regression_rmse', 'macro_f1', 'binary_log_loss', 'multiclass_log_loss'):
            directory = destination / kind
            task, sample, classes = fixture(directory, kind)
            command = [sys.executable, '-X', 'utf8', str(source / 'solution.py'), str(task),
                       '--max-experiments', '1', '--time-budget', '45', '--hypothesis-budget', '15']
            write_json(directory / 'invocation.json', {'command': command, 'cwd': str(source), 'api_environment_cleared': True,
                       'input_sha256': {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}})
            started = time.monotonic()
            with (directory / 'cli.log').open('w', encoding='utf8') as stream:
                child = subprocess.run(command, cwd=source, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            assert child.returncode == 0, f'{kind}: CLI exit {child.returncode}'
            checks = verify_case(directory, sample, classes, kind)
            checks.update(exit_code=child.returncode, seconds=time.monotonic() - started, status='passed')
            report['cases'][kind] = checks
            write_json(directory / 'checks.json', checks)
            write_json(destination / 'verification.json', report)
        for name, item in manifest.items():
            assert sha(source / name) == item['sha256'], f'Copied source mutated: {name}'
        report.update(status='passed', cases_passed=len(report['cases']), total_completed_synthetic_fits=8,
                      requirements_sha256=REQUIREMENTS_SHA256, isolated_flat_source=True)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (destination / 'verification_traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
    write_json(destination / 'verification.json', report)
    (destination / 'report.md').write_text(
        '# SPEC010: проверка clean CLI\n\n'
        f"Статус: **{report['status']}**. Синтетических сценариев пройдено: {len(report['cases'])}/4.\n\n"
        'Источник требований: [SPEC010](../../../docs/specs/SPEC-010-clean-rewrite.md), REQ1001–1003, REQ1010–1012.\n'
        'Независимая копия корневых solution.py (из rewrite_solution.py), research_*.py и byte-exact requirements.txt находится в source/. '
        'Сценарии: RMSE, macro-F1, binary log_loss (вектор), multiclass log_loss (нелексический порядок столбцов классов). '
        'Каждый CLI выполняет один evaluate и один полный refit на новых синтетических 90 train / 12 test; '
        'вложенные CSV, переименованные target/ID, лексические 001/+1/NA, missing и unknown category.\n\n'
        'Проверены схема, ID/порядок/число строк, конечность, диапазон вероятностей/суммы строк, классы, два completed claims, '
        'completed final.json; inner metrics и diagnostics независимо пересчитаны из NPZ через sklearn/numpy. '
        'Все команды, fingerprints, input/source SHA256, worker/CLI logs и результаты сохранены рядом. '
        'API env очищено, ledger читается из корневого HYPOTHESES.md, temporary files остаются в этой папке.\n\n'
        'Ограничения: это техническая синтетическая проверка, не проверка качества на real/hidden данных; '
        'установка requirements в новом окружении и вызов внешнего API не выполнялись. ZIP не создавался. '
        f"Подробности: verification.json; source_manifest.json. Ошибка (если есть): {report.get('error', 'нет')}.\n",
        encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
