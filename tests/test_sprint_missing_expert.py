"""H90 observable gate and actual-subset fitting; synthetic inputs only."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd


class FixedHistoryBase:
    def fit(self, X, y):
        raise AssertionError('Supplied history base must never be refit')

    def predict(self, X):
        return np.full(len(X), 100.)


class MissingExpertTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_missing_expert'), 'H90 missing-rating expert absent')
        self.module = importlib.import_module('sprint_missing_expert')

    def test_insufficient_training_subset_keeps_base_bit_exact_without_fit(self):
        X = pd.DataFrame({'x': np.arange(20), 'energy_star_rating': [np.nan] * 19 + [0.]})
        model = self.module.MissingRatingExpert().fit(X, np.arange(20.), FixedHistoryBase())
        np.testing.assert_array_equal(model.predict(X), np.full(20, 100.))
        self.assertIsNone(model.model_)
        self.assertEqual(model.audit_['expert_fit_rows'], 19)
        self.assertEqual(model.audit_['expert_fit_count'], 0)

    def test_actual_fit_uses_only_missing_rows_and_query_gate_preserves_known_rating(self):
        X = pd.DataFrame({'x': np.linspace(-1, 1, 310), 'floor_area': np.linspace(200, 500, 310),
                          'energy_star_rating': [np.nan] * 300 + [50.] * 10,
                          'facility_type': ['Office'] * 310}, index=np.arange(1000, 1310))
        target = np.r_[600 + 80 * X.x.to_numpy()[:300], np.full(10, 1e9)]
        source = X.copy(deep=True)
        recorded = []
        def register(estimator, frame, y, configuration, **kwargs):
            recorded.append((frame.index.tolist(), np.asarray(y).copy(), configuration))
            return estimator.fit(frame, y, **kwargs)
        model = self.module.MissingRatingExpert(fit_hook=register).fit(X, target, FixedHistoryBase())
        self.assertEqual(recorded[0][0], list(range(1000, 1300)))
        np.testing.assert_array_equal(recorded[0][1], target[:300])
        self.assertEqual(model.feature_.audit_['fit_rows'], 300)
        self.assertEqual(model.audit_['expert_fit_count'], 1)
        np.testing.assert_array_equal(model.fit_positions_, np.arange(300))
        q = pd.DataFrame({'x': [0.] * 5, 'floor_area': [350.] * 5,
            'energy_star_rating': [50., 0., np.nan, np.inf, -np.inf],
            'facility_type': ['unseen', None, 'Office', 'unseen', None]}, index=[50, 51, 52, 53, 54])
        prediction = model.predict(q)
        np.testing.assert_array_equal(prediction[:2], [100., 100.])
        frame = model._prepare(model.feature_.transform(q.iloc[2:]))
        direct = np.maximum(model.model_.predict(frame), 0.)
        np.testing.assert_allclose(prediction[2:], .7 * direct + .3 * 100.)
        self.assertTrue((prediction[2:] > 300).all())
        np.testing.assert_allclose(model.predict(q.iloc[2:3]), prediction[2:3])
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(q), prediction)
        pd.testing.assert_frame_equal(X, source)

    def test_subset_rule_treats_zero_as_observed_and_unknown_text_as_missing(self):
        X = pd.DataFrame({'energy_star_rating': [0., 1., 'bad', None, np.inf, -np.inf, '50']})
        np.testing.assert_array_equal(self.module.missing_rating_mask(X), [False, False, True, True, True, True, False])
        with self.assertRaises(ValueError):
            self.module.missing_rating_mask(pd.DataFrame({'other': [1.]}))

    def test_baseline_required_and_invalid_targets_or_schema_rejected(self):
        X = pd.DataFrame({'energy_star_rating': [np.nan, 50.], 'x': [1., 2.]})
        with self.assertRaises(ValueError):
            self.module.MissingRatingExpert().fit(X, [1., 2.])
        for y in ([1.], [1., -1.], [1., np.nan], [1., np.inf]):
            with self.assertRaises(ValueError):
                self.module.MissingRatingExpert().fit(X, y, FixedHistoryBase())
        model = self.module.MissingRatingExpert().fit(X, [1., 2.], FixedHistoryBase())
        with self.assertRaises(ValueError):
            model.predict(X.rename(columns={'x': 'different'}))
        with self.assertRaises(ValueError):
            self.module.MissingRatingExpert(threads=5)


if __name__ == '__main__':
    unittest.main()
