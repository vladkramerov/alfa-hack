"""CLI checks for budget-gated rolling temporal validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _make_task(directory, periods=5, rows_per_period=60):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261024)
    rows = []
    for period in range(1, periods + 1):
        for row in range(rows_per_period):
            x = rng.normal()
            z = rng.normal()
            category = ('cedar', 'birch', 'maple')[row % 3]
            target = 1.5 * x - .7 * z + .25 * period + .12 * rng.normal()
            rows.append((f'train-{period}-{row}', period, x, z, category, target))
    train = pd.DataFrame(rows, columns=['record_id', 'cohort_period', 'measure_a',
                                        'measure_b', 'material', 'response'])
    future = []
    for row in range(rows_per_period):
        x, z = rng.normal(size=2)
        future.append((f'future-{row}', periods + 1, x, z,
                       ('cedar', 'birch', 'maple')[row % 3]))
    test = pd.DataFrame(future, columns=['record_id', 'cohort_period', 'measure_a',
                                         'measure_b', 'material'])
    sample = pd.DataFrame({'record_id': test.record_id.to_numpy()[::-1], 'response': 0.})
    train.to_csv(directory / 'observations.csv', index=False)
    test.to_csv(directory / 'future_observations.csv', index=False)
    sample.to_csv(directory / 'submission_template.csv', index=False)
    (directory / 'task.md').write_text(
        'Primary metric: RMSE\nTarget: response\nIdentifier: record_id\n'
        'Training data: observations.csv\nTest data: future_observations.csv\n'
        'Sample submission: submission_template.csv\n', encoding='utf-8')
    return directory / 'task.md'


def _run_cli(task, workdir, budget, max_experiments):
    env = os.environ.copy()
    for key in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL'):
        env.pop(key, None)
    env.update(AUTOML_TIME_LIMIT=str(budget),
               AUTOML_VALIDATION_POLICY='nested',
               AUTOML_MAX_EXPERIMENTS=str(max_experiments),
               AUTOML_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(Path(workdir) / 'mpl'))
    return subprocess.run([sys.executable, str(ROOT / 'solution.py'), str(task)],
        cwd=Path(task).parent, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=budget + 15)


def _latest_run(task):
    summaries = sorted((Path(task).parent / 'agent_runs').glob('*/summary.json'))
    if not summaries:
        raise AssertionError('Engine did not save a summary.json')
    return summaries[-1].parent


class TemporalResearchTests(unittest.TestCase):
    def test_two_rolling_windows_are_disjoint_past_only_and_weighted(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _make_task(Path(tmp) / 'rolling', periods=5, rows_per_period=60)
            process = _run_cli(task, tmp, budget=120, max_experiments=2)
            self.assertEqual(process.returncode, 0, process.stdout[-5000:])
            run_dir = _latest_run(task)
            validation = json.loads((run_dir / 'validation.json').read_text(encoding='utf-8'))
            summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
            profile = json.loads((run_dir / 'profile.json').read_text(encoding='utf-8'))
            train = pd.read_csv(Path(task).parent / 'observations.csv')
            periods = train['cohort_period'].to_numpy()

            self.assertEqual(len(validation['search_folds']), 2)
            self.assertEqual(summary['search_window_count'], 2)
            self.assertEqual(validation['search_aggregation'], 'weighted mean across rolling windows')
            self.assertAlmostEqual(validation['search_folds'][0]['weight'], .65)
            self.assertAlmostEqual(validation['search_folds'][1]['weight'], .35)
            self.assertTrue((run_dir / 'search_split-01.npz').is_file())
            self.assertTrue((run_dir / 'search_split-02.npz').is_file())

            with np.load(run_dir / 'search_split-01.npz') as archive:
                primary_fit, primary_valid = archive['fit'], archive['valid']
            with np.load(run_dir / 'search_split-02.npz') as archive:
                earlier_fit, earlier_valid = archive['fit'], archive['valid']
            with np.load(run_dir / 'audit_split.npz') as archive:
                outer_fit, outer_valid = archive['fit'], archive['valid']

            for window_fit, window_valid in ((primary_fit, primary_valid), (earlier_fit, earlier_valid)):
                self.assertTrue(set(window_fit).isdisjoint(outer_valid))
                self.assertTrue(set(window_valid).isdisjoint(outer_valid))
                self.assertTrue(set(window_fit).isdisjoint(window_valid))
                self.assertLess(periods[window_fit].max(), periods[window_valid].min())
            self.assertEqual(set(periods[primary_valid]), {4})
            self.assertEqual(set(periods[earlier_valid]), {3})
            self.assertEqual(set(periods[outer_valid]), {5})

            self.assertEqual(profile['shape'][0], len(earlier_fit))
            self.assertEqual(len(profile['validation_windows']), 2)
            self.assertTrue(profile['validation'].startswith('temporal:'))

            self.assertFalse(summary['audit_used_for_selection'])
            self.assertEqual(summary['final_refit']['status'], 'completed')
            records = json.loads((run_dir / 'experiments.json').read_text(encoding='utf-8'))
            selected_key = json.dumps(summary['selected'], sort_keys=True)
            selected_record = next(record for record in records
                if json.dumps(record['candidate'], sort_keys=True) == selected_key)
            scores = [fold['score'] for fold in selected_record['fold_results']]
            weights = [fold['weight'] for fold in validation['search_folds']]
            self.assertEqual(len(scores), 2)
            self.assertAlmostEqual(summary['selected_search_score'],
                                   float(np.dot(scores, weights)), places=10)
            self.assertEqual(selected_record['validation_aggregation'],
                             'weighted mean of rolling temporal windows')

    def test_short_budget_keeps_one_temporal_search_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = _make_task(Path(tmp) / 'short-budget', periods=5, rows_per_period=60)
            process = _run_cli(task, tmp, budget=45, max_experiments=1)
            self.assertEqual(process.returncode, 0, process.stdout[-5000:])
            run_dir = _latest_run(task)
            validation = json.loads((run_dir / 'validation.json').read_text(encoding='utf-8'))
            summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['search_window_count'], 1)
            self.assertEqual(validation['search_aggregation'], 'single window')
            self.assertEqual(len(validation['search_folds']), 1)
            self.assertTrue((run_dir / 'search_split-01.npz').is_file())
            self.assertFalse((run_dir / 'search_split-02.npz').exists())
            self.assertEqual(summary['final_refit']['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
