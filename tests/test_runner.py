import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


class RunnerTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.run_suite'), 'runner is not implemented')
        return importlib.import_module('experiments.run_suite')

    def test_temporal_splits_keep_outer_and_inner_future_out_of_fit(self):
        api = self.api()
        X = pd.DataFrame({'Year_Factor': np.repeat(np.arange(1, 6), 6), 'value': np.arange(30)})
        splits = api.build_splits(X, pd.Series(np.arange(30)), 'regression')
        self.assertEqual(set(X.iloc[splits['inner_fit']].Year_Factor), {1, 2, 3})
        self.assertEqual(set(X.iloc[splits['inner_valid']].Year_Factor), {4})
        self.assertEqual(set(X.iloc[splits['outer_valid']].Year_Factor), {5})
        self.assertTrue(set(splits['inner_fit']).issubset(splits['outer_fit']))
        self.assertFalse(set(splits['outer_fit']) & set(splits['outer_valid']))

    def test_duplicate_rows_never_cross_grouped_classification_split(self):
        api = self.api()
        X = pd.DataFrame({'value': np.repeat(np.arange(60), 3), 'cat': np.repeat(['a', 'b', 'c'], 60)})
        y = pd.Series(np.repeat(np.arange(60) % 3, 3).astype(str))
        splits = api.build_splits(X, y, 'multiclass')
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        for fit, valid in [('inner_fit', 'inner_valid'), ('outer_fit', 'outer_valid')]:
            self.assertFalse(set(groups[splits[fit]]) & set(groups[splits[valid]]))
            self.assertEqual(set(y.iloc[splits[fit]]), set(y))
            self.assertEqual(set(y.iloc[splits[valid]]), set(y))
        self.assertIn('group', splits['description'].lower())

    def test_submission_validation_rejects_reordering_and_invalid_predictions(self):
        api = self.api()
        test = pd.DataFrame({'id': [10, 20], 'x': [1, 2]})
        sample = pd.DataFrame({'id': [10, 20], 'site_eui': [0, 0]})
        valid = pd.DataFrame({'id': [10, 20], 'site_eui': [1.2, 3.4]})
        api.validate_submission(valid, test, sample, 'regression', None)
        with self.assertRaises(ValueError):
            api.validate_submission(valid.iloc[::-1], test, sample, 'regression', None)
        valid.loc[1, 'site_eui'] = np.inf
        with self.assertRaises(ValueError):
            api.validate_submission(valid, test, sample, 'regression', None)
        with self.assertRaises(ValueError):
            api.validate_submission(pd.DataFrame({'id': [10, 20], 'attack_cat': ['a', 'unknown']}), test,
                                    pd.DataFrame({'id': [10, 20], 'attack_cat': ['a', 'a']}), 'multiclass', ['a', 'b'])

    def test_macro_f1_uses_all_observed_classes(self):
        scores = self.api().metric_values(np.array(['a', 'a', 'b', 'b']), np.array(['a'] * 4), 'multiclass')
        self.assertAlmostEqual(scores['macro_f1'], 1 / 3)
        self.assertAlmostEqual(scores['accuracy'], 0.5)
        self.assertAlmostEqual(self.api().metric_values(np.array([0., 2.]), np.array([0., 0.]), 'regression')['rmse'], np.sqrt(2))

    def test_each_run_gets_a_new_directory_without_overwriting_history(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as folder:
            first = api.create_run_dir(Path(folder), 'HYP-001', 'regression-inner-test')
            (first / 'report.md').write_text('original', encoding='utf-8')
            second = api.create_run_dir(Path(folder), 'HYP-001', 'regression-inner-test')
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'report.md').read_text(encoding='utf-8'), 'original')

    def test_concurrent_tasks_receive_distinct_hypothesis_run_numbers(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as folder:
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(lambda i: api.create_run_dir(Path(folder), 'HYP-001', f'task-{i}'), range(16)))
            numbers = [p.name.split('-')[2] for p in paths]
            self.assertEqual(len(set(numbers)), 16)


if __name__ == '__main__':
    unittest.main()
