import json
import pickle
import unittest

import numpy as np
import pandas as pd

from sprint_linkage import FuzzyTemporalOverlay


def frame(years, areas=None, built=None, rating=None):
    n = len(years)
    return pd.DataFrame({'Year_Factor': years, 'State_Factor': ['S'] * n,
        'building_class': ['Commercial'] * n, 'facility_type': ['Office'] * n,
        'floor_area': areas if areas is not None else [1000.] * n,
        'year_built': built if built is not None else [1950.] * n,
        'energy_star_rating': rating if rating is not None else [50.] * n})


def baseline(X):
    return np.full(len(X), 10.)


class FuzzyTemporalTests(unittest.TestCase):
    def test_exact_h70_unchanged_fuzzy_uses_latest_strictly_past(self):
        model = FuzzyTemporalOverlay().fit(frame([1, 3, 5]), [100., 300., 500.], baseline)
        q = frame([4, 1, 2, 4], areas=[1000., 1000.5, 1000.5, 1000.5])
        pred = model.predict(q)
        self.assertEqual(pred[0], 10.)
        self.assertEqual(pred[1], 10.)
        self.assertAlmostEqual(pred[2], 46.)  # .5*1/1.25*100 + .6*10
        self.assertAlmostEqual(pred[3], 10. + (.5*2/2.25)*(300.-10.))
        self.assertEqual(model.last_prediction_audit_['donor_years'], [None, None, 1., 3.])

    def test_tied_profiles_and_sameyear_collision_are_rejected(self):
        X = frame([1, 1, 1, 1], areas=[1000., 1001., 2000., 2000.])
        model = FuzzyTemporalOverlay().fit(X, [100., 900., 200., 800.], baseline)
        q = frame([2, 2], areas=[1000.5, 2000.5])
        np.testing.assert_array_equal(model.predict(q), [10., 10.])
        self.assertEqual(model.audit_['rejected_ambiguous_profiles'], 1)
        self.assertEqual(model.last_prediction_audit_['rejected_margin'], 1)
        protected = FuzzyTemporalOverlay().fit(frame([1, 1, 1], areas=[1000., 1000., 1001.]),
                                               [100., 999., 200.], baseline)
        np.testing.assert_array_equal(protected.predict(frame([2], areas=[1000.])), [10.])

    def test_missing_year_requires_exact_unique_area(self):
        model = FuzzyTemporalOverlay().fit(frame([1, 1], areas=[1000., 2000.]), [100., 200.], baseline)
        q = frame([2, 2, 2], areas=[1000., 1000.1, 2000.], built=[np.nan, np.nan, 0.])
        np.testing.assert_allclose(model.predict(q), [46., 10., 86.])
        repeated = FuzzyTemporalOverlay().fit(frame([1, 1], areas=[1000., 1000.], built=[1950., 1951.]),
                                               [100., 800.], baseline)
        np.testing.assert_array_equal(repeated.predict(q.iloc[[0]]), [10.])

    def test_fixed_transport_state_changes_prediction_without_estimation(self):
        X = frame([1], rating=[50.])
        state = {'rating_beta': .5, 'group_rating_betas': {'Office': .7}, 'log_change_cap': 1.5, 'beta_cap': 2.}
        model = FuzzyTemporalOverlay().fit(X, [100.], baseline, transport_state=state)
        pred = model.predict(frame([2, 2], areas=[1000.5, 1000.5], rating=[20., 80.]))
        self.assertGreater(pred[0], 46.)
        self.assertLess(pred[1], 46.)
        self.assertEqual(model.audit_['transport_state_fitted_here'], False)
        state['rating_beta'] = 999.
        self.assertEqual(model.transport_state_['rating_beta'], .5)

    def test_block_year_and_area_gates_and_pickle(self):
        model = FuzzyTemporalOverlay().fit(frame([1]), [100.], baseline)
        q = frame([2, 2, 2, 2], areas=[1002., 1000.5, 1000.5, 1000.5], built=[1950., 1952., 1950., 1950.])
        q.loc[2, 'State_Factor'] = 'OTHER'
        expected = [10., 10., 10., 46.]
        np.testing.assert_allclose(model.predict(q), expected)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict(q), expected)
        json.dumps(model.audit_); json.dumps(model.configuration())
        with self.assertRaises(ValueError):
            FuzzyTemporalOverlay().fit(frame([1]), [-1.], baseline)

    def test_prediction_queries_never_enter_donor_memory_and_input_order_is_irrelevant(self):
        X = frame([1, 3], areas=[1000., 1000.])
        a = FuzzyTemporalOverlay().fit(X, [100., 300.], baseline)
        b = FuzzyTemporalOverlay().fit(X.iloc[::-1], [300., 100.], baseline)
        q = frame([2, 3, 4], areas=[1000.5] * 3)
        prediction = a.predict(q)
        self.assertEqual(a.last_prediction_audit_['donor_years'], [1., 1., 3.])
        np.testing.assert_array_equal(b.predict(q), prediction)
        np.testing.assert_array_equal(a.predict(q.iloc[[2]]), prediction[[2]])

    def test_invalid_frozen_transport_and_baseline_fail_without_refitting(self):
        with self.assertRaises(ValueError):
            FuzzyTemporalOverlay().fit(frame([1]), [100.], baseline,
                                      {'rating_beta': np.nan})
        model = FuzzyTemporalOverlay().fit(frame([1]), [100.],
                                           lambda q: np.full(len(q), np.nan))
        with self.assertRaises(ValueError):
            model.predict(frame([2], areas=[1000.5]))


if __name__ == '__main__':
    unittest.main()
