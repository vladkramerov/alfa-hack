"""Task-local cache tests for classification decision-power variants."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from automl_worker import run


class WorkerProbabilityCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(93)
        n_train, n_test = 150, 18
        labels = np.asarray([0] * 90 + [2] * 42 + [10] * 18)
        rng.shuffle(labels)
        x = labels / 5 + rng.normal(0, 1.1, n_train)
        z = rng.normal(size=n_train)
        train = pd.DataFrame({'row_id': [f'train-{i}' for i in range(n_train)],
                              'signal': x, 'noise': z, 'target': labels})
        test_ids = [f'test-{i}' for i in range(n_test)]
        test = pd.DataFrame({'row_id': test_ids,
                             'signal': rng.normal(size=n_test), 'noise': rng.normal(size=n_test)})
        sample = pd.DataFrame({'row_id': test_ids[::-1], 'target': 0})
        train.to_csv(self.root / 'train.csv', index=False)
        test.to_csv(self.root / 'test.csv', index=False)
        sample.to_csv(self.root / 'sample.csv', index=False)
        (self.root / 'task.md').write_text(
            'Primary metric: macro_f1\nTarget: target\nIdentifier: row_id\n'
            'Training data: train.csv\nTest data: test.csv\n'
            'Sample submission: sample.csv\n', encoding='utf-8')
        # Keep all classes represented in fit and fix exact row order for cache validation.
        fit = np.arange(0, 120, dtype=np.int64)
        valid = np.arange(120, n_train, dtype=np.int64)
        np.savez(self.root / 'split.npz', fit=fit, valid=valid)
        self.base_candidate = {'family': 'linear', 'parameters': {'C': .4}, 'decision_power': 0}
        status, baseline, metrics = self._run('baseline', self.base_candidate)
        if status != 0:
            raise AssertionError(f'Baseline worker failed: {metrics}')
        self.cache_path = baseline / 'probability_cache.npz'

    def tearDown(self):
        self.temp.cleanup()

    def _request(self, name, candidate, reuse=None, split=None):
        output = self.root / name
        request = {'task_path': str(self.root / 'task.md'),
                   'split_path': str(split or self.root / 'split.npz'),
                   'candidate': candidate, 'mode': 'evaluate', 'output': str(output), 'threads': 1}
        if reuse is not None:
            request['reuse_probabilities'] = str(reuse)
        manifest = self.root / f'{name}.json'
        manifest.write_text(json.dumps(request), encoding='utf-8')
        return manifest, output

    def _run(self, name, candidate, reuse=None, split=None):
        manifest, output = self._request(name, candidate, reuse, split)
        status = run(manifest)
        metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
        return status, output, metrics

    def test_cached_decision_power_matches_real_refit_and_sample_order(self):
        self.assertTrue(self.cache_path.exists())
        base_metrics = json.loads((self.root / 'baseline' / 'metrics.json').read_text())
        self.assertGreater(base_metrics['fit_seconds'], 0)

        adjusted = {**self.base_candidate, 'decision_power': .3}
        status, cached, cached_metrics = self._run('cached', adjusted, self.cache_path)
        self.assertEqual(status, 0, cached_metrics)
        self.assertEqual(cached_metrics['status'], 'completed')
        self.assertTrue(cached_metrics['training_reused'])
        self.assertEqual(cached_metrics['fit_seconds'], 0)

        status, refit, refit_metrics = self._run('refit', adjusted)
        self.assertEqual(status, 0, refit_metrics)
        self.assertFalse(refit_metrics['training_reused'])
        self.assertGreater(refit_metrics['fit_seconds'], 0)
        self.assertEqual(cached_metrics['score'], refit_metrics['score'])

        cached_valid = np.load(cached / 'validation_predictions.npz')
        refit_valid = np.load(refit / 'validation_predictions.npz')
        np.testing.assert_array_equal(cached_valid['positions'], refit_valid['positions'])
        np.testing.assert_array_equal(cached_valid['prediction'], refit_valid['prediction'])
        cached_submission = pd.read_csv(cached / 'candidate_submission.csv')
        refit_submission = pd.read_csv(refit / 'candidate_submission.csv')
        self.assertEqual(cached_submission['row_id'].tolist(),
                         pd.read_csv(self.root / 'sample.csv')['row_id'].tolist())
        pd.testing.assert_frame_equal(cached_submission, refit_submission)

    def test_cache_rejects_different_training_candidate(self):
        changed = {'family': 'linear', 'parameters': {'C': 2.0}, 'decision_power': .3}
        status, output, metrics = self._run('wrong-key', changed, self.cache_path)
        self.assertEqual(status, 1)
        self.assertIn('training_key mismatch', metrics['error'])
        self.assertFalse((output / 'candidate_submission.csv').exists())
        self.assertFalse((output / 'probability_cache.npz').exists())

    def test_cache_rejects_different_positions(self):
        split = self.root / 'other-split.npz'
        fit = np.arange(0, 119, dtype=np.int64).tolist() + [120]
        valid = np.arange(119, 150, dtype=np.int64)
        np.savez(split, fit=np.asarray(fit, dtype=np.int64), valid=valid)
        status, _, metrics = self._run('wrong-positions',
            {**self.base_candidate, 'decision_power': .3}, self.cache_path, split)
        self.assertEqual(status, 1)
        self.assertIn('fit_positions mismatch', metrics['error'])

    def test_cache_rejects_invalid_probabilities(self):
        bad_cache = self.root / 'baseline' / 'probability_cache.npz'
        with np.load(bad_cache, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        payload['valid_probabilities'] = payload['valid_probabilities'].copy()
        payload['valid_probabilities'][0, 0] = np.nan
        np.savez_compressed(bad_cache, **payload)
        status, output, metrics = self._run('invalid-probability',
            {**self.base_candidate, 'decision_power': .3}, bad_cache)
        self.assertEqual(status, 1)
        self.assertIn('Invalid cached valid probabilities', metrics['error'])
        self.assertFalse((output / 'candidate_submission.csv').exists())


if __name__ == '__main__':
    unittest.main()
