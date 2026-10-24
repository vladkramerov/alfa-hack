"""Synthetic H74 contracts; never reads public competition data."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd
from catboost import Pool
from scipy.special import softmax


class ContextualTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_contextual'), 'contextual corrector missing')
        self.module = importlib.import_module('sprint_contextual')

    def test_contextual_signal_improves_heldout_teacher_cross_entropy(self):
        rng = np.random.default_rng(17)
        train = pd.DataFrame({'context': rng.uniform(-2, 2, 360), 'proto': ['tcp'] * 360})
        query = pd.DataFrame({'context': rng.uniform(-2, 2, 140), 'proto': ['tcp'] * 140})
        y = np.where(train.context > 0, 'z', 'a')
        actual = (query.context.to_numpy() > 0).astype(int)
        teacher = np.tile([.6, .4], (len(train), 1))
        model = self.module.ContextualTeacherCorrector(iterations=70, depth=3, learning_rate=.15).fit(train, y, teacher)
        given = np.tile([.6, .4], (len(query), 1))
        probability = model.predict_proba(query, given)
        teacher_ce = -np.log(given[np.arange(len(query)), actual]).mean()
        self.assertLess(-np.log(probability[np.arange(len(query)), actual]).mean(), teacher_ce * .3)
        np.testing.assert_array_equal(model.classes_, ['a', 'z'])
        np.testing.assert_array_equal(model.predict(query, given), model.classes_[probability.argmax(1)])

    def test_baseline_is_added_exactly_once_and_query_prior_is_used(self):
        X = pd.DataFrame({'context': np.repeat([-1., 1.], 20)})
        y = np.tile(['a', 'b'], 20)
        p = np.tile([.7, .3], (40, 1))
        model = self.module.ContextualTeacherCorrector(iterations=3).fit(X, y, p)
        query_p = np.tile([.2, .8], (40, 1))
        frame = X.copy()
        frame['__teacher_logp_0'] = np.log(.2)
        frame['__teacher_logp_1'] = np.log(.8)
        baseline = np.log(query_p)
        raw = model.model_.predict(Pool(frame), prediction_type='RawFormulaVal')
        embedded = model.model_.predict(Pool(frame, baseline=baseline), prediction_type='RawFormulaVal')
        np.testing.assert_allclose(embedded, raw + baseline, atol=1e-12)
        np.testing.assert_allclose(model.predict_proba(X, query_p), softmax(raw + baseline, axis=1), atol=1e-12)

    def test_balanced_zero_gradient_keeps_teacher_and_weights_mean_one(self):
        X = pd.DataFrame({'x': np.repeat(np.arange(10), 3)})
        y = np.tile(['a', 'b', 'c'], 10)
        p = np.full((30, 3), 1 / 3)
        model = self.module.ContextualTeacherCorrector(iterations=2).fit(X, y, p)
        np.testing.assert_allclose(model.predict_proba(X, p), p, atol=1e-10)
        self.assertAlmostEqual(model.audit_['sample_weight_mean'], 1.)
        with self.assertRaises(ValueError):
            self.module.ContextualTeacherCorrector(iterations=0)

    def test_unknown_missing_and_serialization_are_finite_without_mutating_fit_schema(self):
        X = pd.DataFrame({'x': [0., np.inf, 2., -np.inf, 1., np.nan] * 4,
                          'proto': ['tcp', None, 'udp', '__MISSING__', 'tcp', 'udp'] * 4})
        y = np.tile(['z', 'a', 'm'], 8)
        p = pd.DataFrame(np.full((24, 3), 1 / 3), columns=['a', 'm', 'z'])
        model = self.module.ContextualTeacherCorrector(iterations=5).fit(X, y, p)
        query = pd.DataFrame({'x': [np.inf, np.nan], 'proto': ['unseen', None]})
        given = np.array([[0., 2., 0.], [1., 1., 1.]])
        before = model.audit_.copy()
        prediction = model.predict_proba(query, given)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(prediction.sum(axis=1), 1.)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict_proba(query, given), prediction)
        self.assertEqual(model.audit_, before)
        with self.assertRaises(ValueError):
            model.predict_proba(query[['proto', 'x']], given)

    def test_rejects_teacher_shape_class_order_missing_and_invalid_values(self):
        X = pd.DataFrame({'x': np.arange(12)})
        y = np.tile(['a', 'b', 'c'], 4)
        for bad in [None, np.ones((11, 3)), np.ones((12, 2)), np.zeros((12, 3)),
                    np.full((12, 3), np.nan), np.full((12, 3), -1.),
                    pd.DataFrame(np.ones((12, 3)), columns=['c', 'b', 'a'])]:
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                self.module.ContextualTeacherCorrector(iterations=1).fit(X, y, bad)
        model = self.module.ContextualTeacherCorrector(iterations=1).fit(X, y, np.ones((12, 3)))
        with self.assertRaises(ValueError):
            model.predict_proba(X, np.ones((12, 4)))


if __name__ == '__main__':
    unittest.main()
