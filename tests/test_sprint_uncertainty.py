"""H85 synthetic lognormal reconstruction and installed CatBoost contracts."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_uncertainty'), 'H85 uncertainty module missing')
        self.module = importlib.import_module('sprint_uncertainty')

    def test_original_mean_includes_conditional_variance_not_only_exp_mu(self):
        raw = np.array([[np.log(2), np.log(.5)], [np.log(2), 0.], [np.log(3), -1000.]])
        result = self.module.conditional_lognormal_mean(raw)
        np.testing.assert_allclose(result, [2 * np.exp(.125) - 1, 2 * np.exp(.5) - 1, 2.])
        self.assertGreater(result[1], result[0])

    def test_variance_cap_numerical_guard_and_invalid_raw_parameters(self):
        raw = np.array([[0., 1000.], [0., .5 * np.log(2)], [-20., 1000.], [1000., -1000.]])
        result = self.module.conditional_lognormal_mean(raw)
        np.testing.assert_allclose(result[:3], [np.e - 1, np.e - 1, 0.])
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[-1], 1e300)
        for invalid in ([1., 2.], [[np.nan, 1.]], [[1., np.inf]], [[1., 2., 3.]]):
            with self.assertRaises(ValueError):
                self.module.conditional_lognormal_mean(invalid)

    def test_installed_uncertainty_is_variance_and_model_handles_unseen_missing_pickle(self):
        rng = np.random.default_rng(85)
        X = pd.DataFrame({'x': rng.uniform(-1, 1, 360), 'type': np.repeat(['low', 'high'], 180),
                          'floor_area': np.linspace(100, 500, 360)})
        sigma = np.repeat([.10, .60], 180)
        y = np.expm1(3 + .2 * X.x.to_numpy() + rng.normal(size=360) * sigma)
        original = X.copy(deep=True)
        class ForbiddenBase:
            def predict(self, X):
                raise AssertionError('Standalone H85 must never use or refit base')
        model = self.module.ConditionalLognormalRegressor().fit(X, y, base_predictor=ForbiddenBase())
        q = pd.DataFrame({'x': [0., 0., np.inf, np.nan], 'type': ['low', 'high', 'unseen', None],
                          'floor_area': [200., 400., 300., 350.]})
        prepared = model._prepare(model.feature_.transform(q))
        raw = model.model_.predict(prepared, prediction_type='RawFormulaVal')
        uncertainty = model.model_.predict(prepared, prediction_type='RMSEWithUncertainty')
        np.testing.assert_allclose(uncertainty[:, 0], raw[:, 0], atol=1e-12)
        np.testing.assert_allclose(uncertainty[:, 1], np.exp(2 * raw[:, 1]), atol=1e-12)
        self.assertGreater(uncertainty[1, 1], uncertainty[0, 1])
        prediction = model.predict(q)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())
        np.testing.assert_allclose(prediction, self.module.conditional_lognormal_mean(raw))
        np.testing.assert_allclose(model.predict(q.iloc[:1]), prediction[:1])
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(q), prediction)
        pd.testing.assert_frame_equal(X, original)
        self.assertEqual(model.audit_['fit_rows'], 360)
        self.assertFalse(model.audit_['base_used'])
        self.assertEqual(model.audit_['removed_rows'], 0)

    def test_target_and_schema_guards_precede_learner_fit(self):
        X = pd.DataFrame({'x': [1., 2., 3.]})
        for y in ([1., 2.], [1., -1., 3.], [1., np.nan, 3.], [1., np.inf, 3.]):
            with self.assertRaises(ValueError):
                self.module.ConditionalLognormalRegressor().fit(X, y)
        with self.assertRaises(ValueError):
            self.module.ConditionalLognormalRegressor(threads=5)
        with self.assertRaises(ValueError):
            self.module.ConditionalLognormalRegressor().predict(X)


if __name__ == '__main__':
    unittest.main()
