import json
import pickle
import unittest

import numpy as np
import pandas as pd

from sprint_state_space import RatingStateSpaceHistory


def frame(years, areas=None, ratings=None):
    n = len(years)
    return pd.DataFrame({'Year_Factor': years, 'State_Factor': ['S'] * n,
        'building_class': ['Commercial'] * n, 'facility_type': ['Office'] * n,
        'floor_area': areas if areas is not None else [1000.] * n,
        'year_built': [1950.] * n,
        'energy_star_rating': ratings if ratings is not None else [50.] * n})


def baseline(X):
    return np.full(len(X), 10.)


class RatingStateSpaceTests(unittest.TestCase):
    def test_exact_kalman_recursion_and_lognormal_predictive_mean(self):
        # One-year gap: Ppred=.05, K=5/9, posteriorP=1/45.
        model = RatingStateSpaceHistory().fit(frame([1, 2]), [100., 400.], baseline)
        pred = model.predict(frame([3]))
        mu = (4/9)*np.log(100.) + (5/9)*np.log(400.)
        filtered = np.exp(mu + .5*(1/45+.01+.04))
        expected = (38/45)*filtered + (7/45)*10.
        self.assertAlmostEqual(pred[0], expected)
        self.assertEqual(model.last_prediction_audit_['filtered_rows'], 1)

    def test_strictpast_query_cannot_use_sameyear_or_future_posteriors(self):
        a = RatingStateSpaceHistory().fit(frame([1, 3, 5]), [100., 300., 500.], baseline)
        b = RatingStateSpaceHistory().fit(frame([1, 3, 5]), [100., 9999., 99999.], baseline)
        q = frame([1, 2, 3, 4])
        pred = a.predict(q)
        self.assertEqual(pred[0], 10.)
        self.assertAlmostEqual(pred[1], .76*100*np.exp(.045)+.24*10)
        self.assertAlmostEqual(pred[2], (.76*np.exp(-.25))*(100*np.exp(.05)-10)+10)
        np.testing.assert_array_equal(pred[:3], b.predict(q)[:3])
        self.assertEqual(a.last_prediction_audit_['posterior_years'], [None, 1., 1., 3.])

    def test_collision_missing_query_and_nonpositive_or_missing_donors_fallback(self):
        X = frame([1, 1, 1, 1, 1], areas=[1000., 1000., 2000., 3000., 4000.],
                  ratings=[50., 50., 50., np.nan, 50.])
        model = RatingStateSpaceHistory().fit(X, [100., 999., 0., 100., 100.], baseline)
        q = frame([2]*5, areas=[1000., 2000., 3000., 4000., 9999.],
                  ratings=[50., 50., 50., np.nan, 50.])
        np.testing.assert_array_equal(model.predict(q), [10.]*5)
        self.assertEqual(model.audit_['rejected_ambiguous_profiles'], 1)
        self.assertEqual(model.audit_['usable_filter_observations'], 1)
        self.assertEqual(model.last_prediction_audit_['missing_query_rating_rows'], 1)

    def test_frozen_rating_normalization_and_restore_uses_facility_beta(self):
        state = {'rating_beta': 0., 'group_rating_betas': {'Office': .5}}
        model = RatingStateSpaceHistory().fit(frame([1], ratings=[50.]), [100.], baseline,
                                              transport_state=state)
        prediction = model.predict(frame([2, 2], ratings=[20., 80.]))
        self.assertGreater(prediction[0], prediction[1])
        state['group_rating_betas']['Office'] = 999.
        np.testing.assert_array_equal(model.predict(frame([2, 2], ratings=[20., 80.])), prediction)
        self.assertFalse(model.audit_['transport_state_fitted_here'])

    def test_equal_normalized_states_restore_current_rating_exactly(self):
        # Independent normal quantiles ndtri(.505), ndtri(.805).
        q50, q20 = .012533469508069276, .8596173642419117
        current = 100*np.exp(.5*(q20-q50))
        state = {'rating_beta': .5}
        model = RatingStateSpaceHistory().fit(frame([1, 2], ratings=[50., 20.]),
                                              [100., current], baseline, state)
        prediction = model.predict(frame([3], ratings=[20.]))
        expected = (38/45)*current*np.exp(.5*(1/45+.01+.04))+(7/45)*10
        self.assertAlmostEqual(prediction[0], expected)

    def test_skipped_observations_use_gap_since_last_eligible_and_query_independence(self):
        model = RatingStateSpaceHistory().fit(frame([1, 2, 3], ratings=[50., np.nan, 50.]),
                                              [100., 9999., 400.], baseline)
        q = frame([4, 5])
        pred = model.predict(q)
        # Gap2: Ppred=.06, K=.6, posteriorP=.024; usable support2.
        filtered = np.exp(.4*np.log(100.)+.6*np.log(400.)+.5*(.024+.01+.04))
        self.assertAlmostEqual(pred[0], (38/45)*filtered+(7/45)*10)
        np.testing.assert_array_equal(model.predict(q.iloc[[1]]), pred[[1]])
        self.assertEqual(model.last_prediction_audit_['posterior_years'], [3.])

    def test_pickle_permutation_and_invalid_inputs(self):
        X = frame([1, 2, 3]); y = [100., 200., 150.]
        model = RatingStateSpaceHistory().fit(X, y, baseline)
        reverse = RatingStateSpaceHistory().fit(X.iloc[::-1], y[::-1], baseline)
        q = frame([4]); q['id'] = [123]
        np.testing.assert_array_equal(model.predict(q), reverse.predict(q))
        np.testing.assert_array_equal(model.predict(q), pickle.loads(pickle.dumps(model)).predict(q))
        json.dumps(model.audit_); json.dumps(model.configuration())
        with self.assertRaises(ValueError):
            RatingStateSpaceHistory().fit(frame([1]), [-1.])
        with self.assertRaises(ValueError):
            RatingStateSpaceHistory().fit(frame([1]), [100.], transport_state={'rating_beta': np.nan})
        bad = RatingStateSpaceHistory().fit(frame([1]), [100.], lambda q: np.full(len(q), np.nan))
        with self.assertRaises(ValueError):
            bad.predict(frame([2]))


if __name__ == '__main__':
    unittest.main()
