"""H50 synthetic focal derivative and teacher-offset contracts."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax


class FocalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_focal'), 'H50 focal implementation missing')
        self.module = importlib.import_module('sprint_focal')

    def test_exact_multiclass_gradient_matches_independent_finite_differences(self):
        z = np.array([[1.1, -.2, .3], [-.7, .5, .1], [.1, -.4, .9], [.8, .2, -.5]])
        y = np.array([0, 1, 2, 1])
        for gamma in (0., 1., 2., 4.):
            loss, analytic = self.module.focal_loss_and_gradient(z, y, gamma=gamma)
            numeric = np.empty_like(z)
            for i, j in np.ndindex(z.shape):
                a, b = z.copy(), z.copy()
                a[i, j] += 1e-6
                b[i, j] -= 1e-6
                numeric[i, j] = (self.module.focal_loss_and_gradient(a, y, gamma)[0]
                                  - self.module.focal_loss_and_gradient(b, y, gamma)[0]) / 2e-6
            self.assertGreater(loss, 0.)
            np.testing.assert_allclose(analytic, numeric, atol=2e-9, rtol=1e-6)
            np.testing.assert_allclose(analytic.sum(1), 0., atol=1e-15)

    def test_gamma_zero_is_ce_and_gamma_two_downweights_easy_gradient(self):
        z = np.log([[.9, .1], [.5, .5]])
        y = np.array([0, 0])
        loss, ce_gradient = self.module.focal_loss_and_gradient(z, y, gamma=0)
        self.assertAlmostEqual(loss, (-np.log(.9) - np.log(.5)) / 2)
        np.testing.assert_allclose(ce_gradient, [[-.05, .05], [-.25, .25]])
        focal_loss, focal_gradient = self.module.focal_loss_and_gradient(z, y)
        self.assertAlmostEqual(focal_loss, (-.01 * np.log(.9) - .25 * np.log(.5)) / 2)
        self.assertLess(abs(focal_gradient[0, 0] / ce_gradient[0, 0]),
                        abs(focal_gradient[1, 0] / ce_gradient[1, 0]))

    def test_extreme_correct_and_wrong_logits_remain_finite(self):
        z = np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])
        loss, gradient = self.module.focal_loss_and_gradient(z, np.array([0, 0]))
        self.assertAlmostEqual(loss, 1000.)
        np.testing.assert_allclose(gradient, [[0., 0., 0.], [-.5, .5, 0.]])
        for bad in (np.array([0., 1.]), np.array([0, 3]), np.array([0])):
            with self.assertRaises(ValueError):
                self.module.focal_loss_and_gradient(z, bad)
        with self.assertRaises(ValueError):
            self.module.focal_loss_and_gradient(z, np.array([0, 1]), gamma=-1)

    def test_objective_is_exact_scaled_gradient_and_positive_surrogate(self):
        model = self.module.FocalTeacherBoost(iterations=0)
        model.loss_history_ = []
        z, y = np.array([[.3, -.7, .1], [-.2, .5, 1.]]), np.array([0, 2])
        grad, curvature = model._objective(y, z)
        np.testing.assert_allclose(grad, 2 * self.module.focal_loss_and_gradient(z, y)[1])
        self.assertGreaterEqual(curvature.min(), .001)
        self.assertTrue(np.isfinite(curvature).all())
        self.assertEqual(len(model.loss_history_), 1)

    def test_teacher_offset_is_added_once_and_model_pickles(self):
        X = pd.DataFrame({'x': np.linspace(-2, 2, 150)})
        y = np.where(X.x > 0, 'z', 'a')
        model = self.module.FocalTeacherBoost(iterations=25, learning_rate=.15,
            num_leaves=7, min_child_samples=5).fit(X, y, np.tile([.6, .4], (150, 1)))
        q = pd.DataFrame({'x': [-1.5, -.5, .5, 1.5]})
        p = np.tile([.4, .6], (4, 1))
        probability = model.predict_proba(q, p)
        residual = model.model_.predict(model.encoder_.transform(q), raw_score=True)
        np.testing.assert_allclose(probability, softmax(residual + np.log(p), axis=1), atol=1e-12)
        self.assertEqual(model.predict(q, p).tolist(), ['a', 'a', 'z', 'z'])
        np.testing.assert_allclose(probability[:1], model.predict_proba(q.iloc[:1], p[:1]))
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict_proba(q, p), probability)
        self.assertFalse(model.audit_['curvature_is_exact'])

    def test_zero_rounds_uses_query_teacher_without_any_learner_fit(self):
        X = pd.DataFrame({'x': [0., 1., 2., 3.]})
        model = self.module.FocalTeacherBoost(iterations=0).fit(X, ['a', 'z', 'a', 'z'],
                                                             np.full((4, 2), .5))
        given = [[.2, .8], [.9, .1]]
        np.testing.assert_allclose(model.predict_proba(X.iloc[:2], given), given)
        self.assertEqual(model.audit_['learner_fits'], 0)
        with self.assertRaises(ValueError):
            model.predict_proba(X, None)


if __name__ == '__main__':
    unittest.main()
