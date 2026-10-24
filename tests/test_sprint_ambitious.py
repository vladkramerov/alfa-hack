"""Synthetic H77 contracts: no competition files or teacher fitting."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax


class LovaszTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_ambitious'), 'H77 module missing')
        self.module = importlib.import_module('sprint_ambitious')

    def test_rank_surrogate_has_hand_checked_value(self):
        # Class-0 descending weights [.5,.5], class-1 descending weights [1,0].
        z = np.log([[.8, .2], [.3, .7]])
        loss, gradient = self.module.lovasz_softmax_loss_and_gradient(z, np.array([0, 1]), ce_weight=0)
        self.assertAlmostEqual(loss, .275)
        np.testing.assert_allclose(gradient, [[-.04, .04], [.1575, -.1575]], atol=1e-12)

    def test_full_softmax_chain_gradient_matches_non_tied_finite_differences(self):
        z = np.array([[1.3, -.2, .7], [-.9, .4, .1], [.2, -.7, 1.1], [.6, 1.7, -.1]])
        y = np.array([0, 1, 2, 0])
        for ce_weight in (0., .5, 1.):
            loss, analytical = self.module.lovasz_softmax_loss_and_gradient(z, y, ce_weight=ce_weight)
            numeric = np.empty_like(z)
            for i, j in np.ndindex(z.shape):
                plus, minus = z.copy(), z.copy()
                plus[i, j] += 1e-6
                minus[i, j] -= 1e-6
                a = self.module.lovasz_softmax_loss_and_gradient(plus, y, ce_weight=ce_weight)[0]
                b = self.module.lovasz_softmax_loss_and_gradient(minus, y, ce_weight=ce_weight)[0]
                numeric[i, j] = (a - b) / 2e-6
            self.assertTrue(np.isfinite(loss))
            np.testing.assert_allclose(analytical, numeric, atol=2e-9, rtol=1e-6)
            np.testing.assert_allclose(analytical.sum(axis=1), 0, atol=1e-15)

    def test_absent_class_and_extreme_logits_are_finite(self):
        z = np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])
        loss, grad = self.module.lovasz_softmax_loss_and_gradient(z, np.array([0, 1]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(grad).all())
        for bad in (np.array([0, 3]), np.array([0., 1.]), np.array([0])):
            with self.assertRaises(ValueError):
                self.module.lovasz_softmax_loss_and_gradient(z, bad)

    def test_zero_rounds_preserves_explicit_query_teacher_without_fit(self):
        X = pd.DataFrame({'x': [0., 1., 2., 3.], 'kind': ['a', 'b', None, 'a']})
        y = ['z', 'a', 'z', 'a']
        model = self.module.LovaszTeacherBoost(iterations=0).fit(X, y, np.full((4, 2), .5))
        query = X.iloc[:2].copy()
        p = np.array([[1., 4.], [9., 1.]])
        np.testing.assert_allclose(model.predict_proba(query, p), [[.2, .8], [.9, .1]])
        self.assertEqual(model.audit_['learner_fits'], 0)
        self.assertEqual(model.predict(query, p).tolist(), ['z', 'a'])

    def test_real_boosting_learns_context_and_keeps_teacher_baseline_once(self):
        rng = np.random.default_rng(77)
        X = pd.DataFrame({'x': rng.uniform(-2, 2, 240), 'kind': ['tcp'] * 240})
        q = pd.DataFrame({'x': rng.uniform(-2, 2, 80), 'kind': ['unknown'] * 80})
        y = np.where(X.x > 0, 'z', 'a')
        truth = (q.x.to_numpy() > 0).astype(int)
        model = self.module.LovaszTeacherBoost(iterations=30, learning_rate=.15, num_leaves=7,
                                             min_child_samples=5).fit(X, y, np.tile([.6, .4], (240, 1)))
        p = np.tile([.6, .4], (80, 1))
        given = model.predict_proba(q, p)
        self.assertGreater(np.mean(given.argmax(1) == truth), .95)
        features = model.encoder_.transform(q)
        raw = model.model_.predict(features, raw_score=True)
        np.testing.assert_allclose(given, softmax(raw + np.log(p), axis=1), atol=1e-12)
        np.testing.assert_allclose(model.predict_proba(q.iloc[:1], p[:1]), given[:1], atol=1e-12)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict_proba(q, p), given, atol=1e-12)
        self.assertEqual(model.audit_['learner_fits'], 1)

    def test_positive_curvature_surrogate_and_scaled_gradient_contract(self):
        X = pd.DataFrame({'x': np.arange(12)})
        model = self.module.LovaszTeacherBoost(iterations=0).fit(
            X, ['a', 'b', 'c'] * 4, np.full((12, 3), 1 / 3))
        z = np.arange(36).reshape(12, 3) / 7
        y = np.tile(np.arange(3), 4)
        grad, hess = model._objective(y, z)
        expected = self.module.lovasz_softmax_loss_and_gradient(z, y)[1] * 12
        np.testing.assert_allclose(grad, expected)
        self.assertTrue(np.isfinite(hess).all())
        self.assertGreaterEqual(hess.min(), 1e-3)

    def test_teacher_and_schema_validation(self):
        X = pd.DataFrame({'x': [0., 1., 2., 3.]})
        y = ['a', 'b', 'a', 'b']
        for p in (None, np.zeros((4, 2)), np.full((4, 2), np.nan), np.ones((4, 3)),
                  np.full((4, 2), -1.), pd.DataFrame(np.ones((4, 2)), columns=['b', 'a'])):
            with self.subTest(teacher=type(p).__name__), self.assertRaises(ValueError):
                self.module.LovaszTeacherBoost(iterations=0).fit(X, y, p)
        model = self.module.LovaszTeacherBoost(iterations=0).fit(X, y, np.full((4, 2), .5))
        with self.assertRaises(ValueError):
            model.predict_proba(pd.DataFrame({'bad': [0.]}), [[.5, .5]])
        with self.assertRaises(ValueError):
            model.predict_proba(X, None)
        for arguments in ({'iterations': -1}, {'threads': 5}, {'ce_weight': 2.}):
            with self.assertRaises(ValueError):
                self.module.LovaszTeacherBoost(**arguments)


if __name__ == '__main__':
    unittest.main()
