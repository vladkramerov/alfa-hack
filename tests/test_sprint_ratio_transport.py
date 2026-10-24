import json
import pickle
import unittest

import numpy as np
import pandas as pd

from sprint_ratio_transport import OriginalLossTransport


def frame(years, areas=None, ratings=None, facility=None):
    n = len(years)
    return pd.DataFrame({'Year_Factor': years, 'State_Factor': ['S'] * n,
        'building_class': ['Commercial'] * n,
        'facility_type': facility if facility is not None else ['Office'] * n,
        'floor_area': areas if areas is not None else [1000.] * n,
        'year_built': [1950.] * n,
        'energy_star_rating': ratings if ratings is not None else [50.] * n})


def baseline(X):
    return np.full(len(X), 10.)


class OriginalLossTransportTests(unittest.TestCase):
    def test_ratio_squared_weights_are_exact_original_unit_loss(self):
        donor = np.array([10., 100., 2.]); current = np.array([20., 50., 0.])
        ratio, weight = OriginalLossTransport.ratio_targets(donor, current)
        np.testing.assert_array_equal(ratio, [2., .5, 0.])
        np.testing.assert_array_equal(weight, [100., 10000., 4.])
        candidate = np.array([.5, 1.2, .3])
        np.testing.assert_allclose(weight * (ratio-candidate)**2, (current-donor*candidate)**2)

    def test_equal_score_weighted_optimum_is_not_unweighted_or_log_mean(self):
        X = frame([1, 2, 1, 2], areas=[1000., 1000., 2000., 2000.])
        model = OriginalLossTransport().fit(X, [10., 20., 100., 50.], baseline)
        prediction = model.predict(frame([3]))
        # Least squares ratio = (10*20+100*50)/(10²+100²) = 52/101.
        expected = (38/45)*(1040/101) + (7/45)*10
        self.assertAlmostEqual(prediction[0], expected)
        self.assertEqual(model.audit_['training_ratio_pairs'], 2)

    def test_monotone_ratio_bounds_group_shrink_and_missing_rating(self):
        ratings = [80., 20., 20., 80., 80., 20., 20., 80.]
        X = frame([1, 2]*4, areas=np.repeat([1000., 2000., 3000., 4000.], 2),
                  ratings=ratings, facility=['Office']*4+['School']*4)
        model = OriginalLossTransport(group_minimum_pairs=2).fit(X, [100., 400., 100., 25., 100., 200., 100., 50.], baseline)
        q = frame([3]*4, ratings=[0., 50., 90., np.nan])
        pred = model.predict(q)
        self.assertGreater(pred[0], pred[1]); self.assertGreater(pred[1], pred[2])
        # Global positive-delta ratio is 3, Office ratio4, count shrink2/202.
        self.assertAlmostEqual(pred[0], (38/45)*400*(304/101)+(7/45)*10)
        self.assertAlmostEqual(pred[3], (38/45)*400 + (7/45)*10)
        self.assertEqual(set(model.group_curves_), {'Office', 'School'})
        grid = np.linspace(-20, 20, 100)
        ratio = model.global_curve_.predict(grid)
        self.assertTrue((np.diff(ratio)>=0).all())
        self.assertGreaterEqual(ratio.min(), np.exp(-1.5))
        self.assertLessEqual(ratio.max(), np.exp(1.5))

    def test_observed_ratios_are_not_clipped_before_constrained_fitting(self):
        X = frame([1, 2, 1, 2], areas=[1000., 1000., 2000., 2000.])
        model = OriginalLossTransport().fit(X, [10., 1000., 10., 0.], baseline)
        pred = model.predict(frame([3]))
        # Equal weights at the same delta give true optimum50, then cap exp(1.5).
        self.assertAlmostEqual(pred[0], (38/45)*1000*np.exp(1.5)+(7/45)*10)

    def test_strictpast_collision_zero_donor_and_missing_pairs_fallback(self):
        X = frame([1, 3, 5, 1, 1, 1, 2], areas=[1000., 1000., 1000., 2000., 2000., 3000., 3000.],
                  ratings=[np.nan]*7)
        model = OriginalLossTransport().fit(X, [100., 300., 500., 200., 900., 0., 50.], baseline)
        q = frame([1, 2, 4, 5, 2, 2], areas=[1000., 1000., 1000., 1000., 2000., 3000.])
        pred = model.predict(q)
        np.testing.assert_allclose(pred[:3], [10., 78.4, (38/45)*300+(7/45)*10])
        self.assertAlmostEqual(pred[3], ((38/45)*300+(7/45)*10-10)*np.exp(-.25)+10)
        self.assertEqual(pred[4], 10.)
        self.assertEqual(model.last_prediction_audit_['donor_years'], [None, 1., 3., 3., None, 1.])
        self.assertEqual(model.audit_['rejected_ambiguous_profiles'], 1)
        self.assertEqual(model.audit_['training_ratio_pairs'], 0)
        self.assertIsNone(model.global_curve_)

    def test_no_current_target_in_pair_features_pickle_and_permutation(self):
        X = frame([1, 2, 3], ratings=[80., 50., 20.])
        a = OriginalLossTransport().fit(X, [100., 150., 200.], baseline)
        b = OriginalLossTransport().fit(X.iloc[::-1], [200., 150., 100.], baseline)
        self.assertEqual([(p['donor_year'],p['current_year']) for p in a.training_pair_audit_], [(1.,2.),(2.,3.)])
        q = frame([4, 5], ratings=[20., np.nan]); q['id'] = [999, 123]
        pred = a.predict(q)
        np.testing.assert_array_equal(pred, b.predict(q))
        np.testing.assert_array_equal(pred, pickle.loads(pickle.dumps(a)).predict(q))
        self.assertEqual(a.last_prediction_audit_['donor_years'], [3., 3.])
        json.dumps(a.audit_); json.dumps(a.configuration())

    def test_invalid_targets_and_baseline_rejected(self):
        with self.assertRaises(ValueError):
            OriginalLossTransport().fit(frame([1]), [-1.])
        with self.assertRaises(ValueError):
            OriginalLossTransport.ratio_targets([0.], [1.])
        model = OriginalLossTransport().fit(frame([1]), [100.], lambda q: np.full(len(q), np.nan))
        with self.assertRaises(ValueError):
            model.predict(frame([2]))


if __name__ == '__main__':
    unittest.main()
