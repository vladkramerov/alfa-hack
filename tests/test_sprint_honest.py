"""Synthetic H88 cache/OOF tests: no scientific or real learner fits."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

AVAILABLE = (importlib.util.find_spec('sprint_honest') is not None
             and importlib.util.find_spec('experiments.sprint_crossfit_cache') is not None)
if AVAILABLE:
    from experiments import sprint_crossfit_cache as cache
    from sprint_honest import HonestContextCorrector


class FakeRaw:
    def __init__(self, component, parameters=None, threads=2):
        self.component, self.parameters, self.threads = component, parameters, threads

    def fit(self, X, y):
        raise AssertionError('Only the injected registration hook may fit a teacher')

    def predict_proba(self, X):
        return np.tile([.6, .1, .3], (len(X), 1))


class HonestTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'H88 implementation is absent')
        parent = Path(__file__).resolve().parents[1] / 'results/2026-10-24/series-three-hours-v1'
        self.temporary = tempfile.TemporaryDirectory(dir=parent, prefix='honest-test-')
        self.root = Path(self.temporary.name)
        self.output = self.root / 'results/run'
        X = pd.DataFrame({'group': np.repeat(np.arange(24), 2), 'proto': ['tcp'] * 48})
        y = pd.Series(np.repeat(np.array(['a', 'b', 'c'] * 8), 2))
        self.bundle = SimpleNamespace(X=X, y=y, hashes={'train.csv': 'synthetic-hash'})
        self.outer_fit, self.outer_valid = np.arange(36), np.arange(36, 48)
        classes = np.array(['a', 'b', 'c'])
        subset = X.iloc[self.outer_fit]
        groups = pd.util.hash_pandas_object(subset, index=False).to_numpy()
        folds = []
        directory = self.root / 'raw-cache'
        directory.mkdir()
        for number, (fit, valid) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(subset, y.iloc[self.outer_fit], groups)):
            fold = {'fit': fit, 'valid': valid, 'fit_positions': self.outer_fit[fit],
                    'valid_positions': self.outer_fit[valid], 'classes': classes,
                    'probabilities': np.tile([.2, .3, .5], (len(valid), 1))}
            np.savez(directory / f'fold-{number}.npz', **fold)
            folds.append(fold)
        self.raw = {'directory': directory, 'folds': folds, 'classes': classes,
                    'probabilities': np.tile([.2, .3, .5], (36, 1)),
                    'audit': {'model_fits': 0, 'key': 'synthetic-raw-cache'}}
        self.fit, self.valid = folds[0]['fit_positions'], folds[0]['valid_positions']
        source = self.root / 'results/2026-10-23/HYP-011-02-multiclass-none'
        source.mkdir(parents=True)
        (source / 'evidence.json').write_text(json.dumps({'hashes': self.bundle.hashes}), encoding='utf8')
        np.savez(source / 'splits.npz', fit=self.outer_fit, valid=self.outer_valid)
        np.savez(source / 'validation_predictions.npz', positions=self.outer_valid, classes=classes,
                 probabilities=np.tile([.1, .2, .7], (12, 1)), y=np.array(['unsafe'], dtype=object))

    def tearDown(self):
        if hasattr(self, 'temporary'):
            self.temporary.cleanup()

    def load(self, stage='inner', fit_hook=None, fit=None, valid=None):
        if fit is None:
            fit = self.fit if stage == 'inner' else self.outer_fit
        if valid is None:
            valid = self.valid if stage == 'inner' else self.outer_valid
        with patch.object(cache, '_original_context', return_value=(self.root, self.outer_fit, self.outer_valid)), \
             patch.object(cache, '_read_component', return_value=self.raw), \
             patch.object(cache, 'ClassBranch', FakeRaw):
            return cache.load_honest_teacher(self.bundle, stage, fit, valid,
                root=self.root, output=self.output, fit_hook=fit_hook)

    def hook(self, estimator, X, y, configuration):
        self.assertEqual(estimator.component, 'raw')
        self.assertEqual(estimator.parameters, {})
        self.assertEqual(estimator.threads, 2)
        self.assertEqual(configuration['encoder'], {'native': False, 'scale': False})
        self.assertTrue((self.output / 'honest-raw-oof' / f"fold-{configuration['nested_fold']}.started.json").exists())
        self.assertTrue(set(X.index).issubset(set(self.fit)))
        self.assertEqual(len(X), len(y))
        self.calls.append(configuration)
        estimator.classes_ = np.array(['c', 'a', 'b'])
        return estimator

    def test_inner_three_hooked_new_folds_cover_every_row_with_aligned_classes(self):
        self.calls = []
        teacher = self.load(fit_hook=self.hook)
        self.assertEqual(len(self.calls), 3)
        np.testing.assert_allclose(teacher['train_probabilities'], np.tile([.1, .3, .6], (len(self.fit), 1)))
        np.testing.assert_allclose(teacher['valid_probabilities'], np.tile([.2, .3, .5], (len(self.valid), 1)))
        np.testing.assert_array_equal(teacher['fit_positions'], self.fit)
        self.assertTrue(teacher['audit']['independent_training_oof'])
        self.assertEqual(teacher['audit']['new_teacher_fits'], 3)
        self.assertFalse(teacher['audit']['H48_meta_state_used'])
        for claim in self.calls:
            self.assertFalse(set(claim['fit_positions']) & set(claim['valid_positions']))
            groups = self.bundle.X['group'].to_numpy()
            self.assertFalse(set(groups[claim['fit_positions']]) & set(groups[claim['valid_positions']]))

    def test_completed_inner_cache_reused_without_any_hook_calls(self):
        self.calls = []
        original = self.load(fit_hook=self.hook)
        def forbidden(*args):
            raise AssertionError('Completed cache must not refit')
        cached = self.load(fit_hook=forbidden)
        np.testing.assert_array_equal(cached['train_probabilities'], original['train_probabilities'])
        self.assertEqual(cached['audit']['new_teacher_fits'], 0)

    def test_cached_fold_byte_mutation_rejected_before_any_refit(self):
        self.calls = []
        self.load(fit_hook=self.hook)
        self.calls = []
        with (self.output / 'honest-raw-oof/fold-0.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.load(fit_hook=self.hook)
        self.assertEqual(self.calls, [])

    def test_outer_reuses_existing_raw_cache_and_selective_nonpickle_query(self):
        def forbidden(*args):
            raise AssertionError('Outer teacher must never refit')
        teacher = self.load('outer', forbidden)
        np.testing.assert_allclose(teacher['train_probabilities'], self.raw['probabilities'])
        np.testing.assert_allclose(teacher['valid_probabilities'], np.tile([.1, .2, .7], (12, 1)))
        self.assertEqual(teacher['audit']['new_teacher_fits'], 0)
        self.assertFalse(self.output.exists())

    def test_wrong_original_positions_and_missing_hook_rejected(self):
        with self.assertRaises(ValueError):
            self.load('outer', fit=self.outer_fit[::-1])
        with self.assertRaises(ValueError):
            self.load(fit_hook=None)

    def test_interrupted_fold_claim_cannot_be_silently_refitted(self):
        self.calls = []
        def interrupted(estimator, X, y, configuration):
            raise RuntimeError('synthetic interruption')
        with self.assertRaises(RuntimeError):
            self.load(fit_hook=interrupted)
        with self.assertRaisesRegex(ValueError, 'attempted|unfinished|incomplete'):
            self.load(fit_hook=self.hook)
        self.assertEqual(self.calls, [])

    def test_wrapper_uses_unchanged_fixed_context_configuration(self):
        model = HonestContextCorrector()
        config = model.configuration()
        self.assertEqual(config['hypothesis'], 'HYP-088')
        self.assertEqual(config['training_teacher'], 'grouped_oof_raw_xgboost')
        self.assertEqual(config['parameters']['iterations'], 400)
        self.assertEqual(config['parameters']['depth'], 5)
        self.assertEqual(config['parameters']['l2_leaf_reg'], 10)
        self.assertEqual(config['parameters']['thread_count'], 2)

    def test_wrapper_zero_residual_reproduces_teacher_and_audits_h88(self):
        class ZeroResidual:
            def __init__(self, **parameters):
                self.classes = parameters['classes_count']
            def fit(self, pool):
                return self
            def predict(self, pool, prediction_type):
                return np.zeros((pool.num_row(), self.classes))
        teacher = np.tile([.2, .3, .5], (len(self.fit), 1))
        query_teacher = np.tile([.1, .2, .7], (len(self.valid), 1))
        with patch('sprint_contextual.CatBoostClassifier', ZeroResidual):
            model = HonestContextCorrector().fit(self.bundle.X.iloc[self.fit], self.bundle.y.iloc[self.fit], teacher)
            p = model.predict_proba(self.bundle.X.iloc[self.valid], query_teacher)
        np.testing.assert_allclose(p, query_teacher, rtol=1e-14, atol=1e-14)
        self.assertEqual(model.audit_['configuration']['hypothesis'], 'HYP-088')
        self.assertTrue(model.audit_['teacher_oof_provenance_is_callers_responsibility'])


if __name__ == '__main__':
    unittest.main()
