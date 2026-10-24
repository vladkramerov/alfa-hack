"""Clean-room CLI checks: only ZIP-root files plus newly generated task data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from package_solution import build, ROOT, MODULES, OFFICIAL_REQUIREMENTS_SHA256


def fixture(directory, metric, kind='regression', seed=1729, probability=False):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    nested = directory / 'material' / 'data'
    nested.mkdir(parents=True)
    n, m = 300, 47
    a, b = rng.normal(size=n + m), rng.normal(size=n + m)
    frame = pd.DataFrame({'reference_key': [f'item-{i:05d}' for i in range(n + m)],
                          'measurement': a, 'other': b, 'segment': rng.choice(['oak', 'pine', 'birch'], n + m)})
    if kind == 'regression':
        y = 5 + 2 * a - b + .1 * rng.normal(size=n + m)
    elif kind == 'binary':
        y = np.where(a + b > 0, 0, 7)
    else:
        y = np.where(a > .5, 'copper', np.where(b > .3, 'silver', 'gold'))
    train = frame.iloc[:n].copy()
    train['unknown_outcome'] = y[:n]
    test = frame.iloc[n:].copy()
    train.to_csv(nested / 'observations.csv', index=False)
    test.to_csv(nested / 'queries.csv', index=False)
    sample = pd.DataFrame({'reference_key': test['reference_key'].iloc[::-1].to_numpy()})
    if probability and kind == 'classification':
        for label in ['silver', 'copper', 'gold']:
            sample[label] = 0.
    else:
        sample['unknown_outcome'] = 0. if probability or kind == 'regression' else ''
    sample.to_csv(nested / 'answer_template.csv', index=False)
    description = (f'# Unseen {kind} task\nPrimary metric: {metric}\nTarget: unknown_outcome\n'
        'Identifier: reference_key\n'
        'Training data: `material/data/observations.csv`\n'
        'Test data: `material/data/queries.csv`\n'
        'Sample submission: `material/data/answer_template.csv`\n')
    if kind == 'binary' and probability:
        description += 'Positive class: 0\nPredict positive-class probability.\n'
    (directory / 'task.md').write_text(description, encoding='utf-8')
    # Holdout truth exists only in this parent-process metadata, never in CSV or child environment.
    return {'rows': m, 'kind': kind, 'metric': metric, 'probability': probability,
            'sample': sample, 'classes': tuple(sorted(map(str, np.unique(y)))),
            'train_truth': np.asarray(y[:n]).copy(),
            'test_truth': np.asarray(y[n:][::-1]).copy(), 'seed': seed,
            'positive_class': '0' if kind == 'binary' else None}


def _metric_from_submission(output, metadata):
    metric = metadata['metric']
    truth = np.asarray(metadata['test_truth'])
    if metadata['kind'] == 'regression':
        actual = truth.astype(float)
        predicted = output['unknown_outcome'].to_numpy(dtype=float)
        if metric == 'rmse':
            return float(np.sqrt(mean_squared_error(actual, predicted)))
        if metric == 'mae':
            return float(mean_absolute_error(actual, predicted))
        if metric == 'r2':
            return float(r2_score(actual, predicted))
        raise ValueError(f'Unsupported synthetic metric: {metric}')
    truth = np.asarray(truth, dtype=str)
    if not metadata['probability']:
        predicted = output['unknown_outcome'].astype(str).to_numpy()
        if metric == 'accuracy':
            return float(accuracy_score(truth, predicted))
        average = {'macro_f1': 'macro', 'micro_f1': 'micro'}[metric]
        return float(f1_score(truth, predicted, labels=list(metadata['classes']),
                              average=average, zero_division=0))
    classes = list(metadata['classes'])
    if metadata['kind'] == 'binary':
        positive = metadata['positive_class']
        positive_probability = output['unknown_outcome'].to_numpy(dtype=float)
        probabilities = np.column_stack([
            positive_probability if label == positive else 1 - positive_probability
            for label in classes])
        if metric == 'roc_auc':
            return float(roc_auc_score(truth == positive, positive_probability))
    else:
        # The template intentionally lists probability columns in reverse class order.
        probabilities = output.loc[:, classes].to_numpy(dtype=float)
    return float(log_loss(truth, probabilities, labels=classes))


def _constant_control(metadata):
    sample = metadata['sample'].copy()
    train = np.asarray(metadata['train_truth'])
    if metadata['kind'] == 'regression':
        sample['unknown_outcome'] = float(np.mean(train))
    else:
        labels, counts = np.unique(train.astype(str), return_counts=True)
        classes = list(metadata['classes'])
        priors = {label: count / len(train) for label, count in zip(labels, counts)}
        if not metadata['probability']:
            maximum = max(counts)
            sample['unknown_outcome'] = min(label for label, count in zip(labels, counts)
                                            if count == maximum)
        elif metadata['kind'] == 'binary':
            sample['unknown_outcome'] = priors[metadata['positive_class']]
        else:
            for label in classes:
                sample[label] = priors.get(label, 0.0)
    return sample


def check_output(directory, metadata):
    output = pd.read_csv(directory / 'submission.csv')
    sample = metadata['sample']
    assert list(output) == list(sample), 'Output columns differ from template'
    assert len(output) == metadata['rows'], 'Output row count mismatch'
    assert output['reference_key'].equals(sample['reference_key']), 'Output ID order mismatch'
    assert not output.isna().any().any(), 'Missing output values'
    values = output.drop(columns='reference_key')
    if metadata['kind'] == 'regression' or metadata['probability']:
        assert np.isfinite(values.to_numpy(dtype=float)).all(), 'Nonfinite values'
        if metadata['probability']:
            assert ((values >= 0) & (values <= 1)).all().all(), 'Invalid probability range'
            if values.shape[1] > 1:
                assert np.allclose(values.sum(axis=1), 1), 'Invalid probability sums'
    else:
        assert set(values.iloc[:, 0].astype(str)).issubset(metadata['classes']), 'Unknown classes'
    summaries = sorted((directory / 'agent_runs').glob('*/summary.json'))
    assert summaries, 'Missing execution diagnostics'
    summary = json.loads(summaries[-1].read_text())
    assert summary['status'] == 'completed', summary
    assert summary['final_refit']['status'] == 'completed', summary
    return summary, output


def _child_environment(workspace):
    # Keep runtime essentials while ensuring no source API credentials reach the clean-room child.
    allowed = ('PATH', 'HOME', 'TMPDIR', 'TEMP', 'TMP', 'SYSTEMROOT', 'WINDIR', 'LANG',
               'LC_ALL', 'DYLD_FALLBACK_LIBRARY_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(AUTOML_TIME_LIMIT='45', AUTOML_MAX_EXPERIMENTS='2', AUTOML_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(workspace / 'matplotlib'))
    return env


def verify(output, keep=None):
    manifest = build()
    archive_path = ROOT / 'dist' / 'alfa_autoresearch_submission.zip'
    records = []
    workspace = Path(keep) if keep else Path(tempfile.mkdtemp(prefix='alfa-cleanroom-'))
    workspace.mkdir(parents=True, exist_ok=True)
    scenarios = [('rmse', 'regression', False), ('mae', 'regression', False),
                 ('r2', 'regression', False), ('macro_f1', 'classification', False),
                 ('micro_f1', 'classification', False), ('accuracy', 'classification', False),
                 ('roc_auc', 'binary', True), ('log_loss', 'binary', True),
                 ('log_loss', 'classification', True)]
    for index, (metric, kind, probability) in enumerate(scenarios):
        directory = workspace / f'{index + 1:02d}-{metric}-{kind}'
        metadata = fixture(directory, metric, kind, probability=probability)
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.namelist() == list(MODULES)
            assert all('/' not in name and '\\' not in name for name in archive.namelist())
            archive.extractall(directory)
        assert hashlib.sha256((directory / 'requirements.txt').read_bytes()).hexdigest() == OFFICIAL_REQUIREMENTS_SHA256
        env = _child_environment(workspace)
        started = time.monotonic()
        process = subprocess.run([sys.executable, 'solution.py', 'task.md'], cwd=directory,
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=55)
        (directory / 'cli.log').write_text(process.stdout, encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'{directory.name}: CLI failed\n{process.stdout[-5000:]}')
        summary, submission = check_output(directory, metadata)
        test_score = _metric_from_submission(submission, metadata)
        control_score = _metric_from_submission(_constant_control(metadata), metadata)
        lower_is_better = metadata['metric'] in {'rmse', 'mae', 'log_loss'}
        improvement = control_score - test_score if lower_is_better else test_score - control_score
        assert improvement > 1e-12, (
            f'{directory.name}: synthetic holdout did not beat constant control '
            f'(candidate={test_score:.9g}, control={control_score:.9g})')
        records.append({'scenario': directory.name, 'status': 'passed', 'seconds': time.monotonic() - started,
            'experiments': summary['experiments'], 'first_submission_seconds': summary['first_submission_seconds'],
            'final_refit': summary['final_refit']['status'], 'llm_calls': summary['llm_calls'],
            'test_metric': test_score, 'constant_control_metric': control_score,
            'improvement_over_control': improvement, 'seed': metadata['seed']})
        print(json.dumps(records[-1]), flush=True)
    report = {'status': 'passed', 'synthetic_quality_checked': True,
              'python': sys.version, 'archive': manifest,
              'scenarios': records, 'workspace': str(workspace), 'real_llm_api': False,
              'limitation': 'Local macOS Python 3.12 runtime; not the organizer Docker container.'}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not keep:
        shutil.rmtree(workspace)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'results' / 'cleanroom.json'))
    parser.add_argument('--keep')
    args = parser.parse_args()
    verify(args.output, args.keep)
