import json
import pickle
import unittest

import numpy as np
import pandas as pd

from sprint_differences import PanelDifferenceRegressor


def frame(years, areas=None, ratings=None, facilities=None):
    n = len(years)
    return pd.DataFrame({'Year_Factor': years, 'State_Factor': ['S'] * n,
        'building_class': ['Commercial'] * n,
        'facility_type': facilities if facilities is not None else ['Office'] * n,
        'floor_area': areas if areas is not None else [1000.] * n,
        'year_built': [1950.] * n,
        'energy_star_rating': ratings if ratings is not None else [50.] * n,
        'january_avg_temp_c': np.asarray(years, float)*10})


def baseline(X):
    return np.full(len(X), 10.)


class PanelDifferenceTests(unittest.TestCase):
    def test_pair_features_are_changes_and_strictpast_target_only(self):
        X = frame([1, 2], ratings=[50., 20.]); X['id'] = [77, 99]
        model = PanelDifferenceRegressor().fit(X, [100., 150.], baseline)
        features = model._pair_features(X, [1], [0], [100.])
        self.assertEqual(features['donor_y'].iloc[0], 100.)
        self.assertEqual(features['age'].iloc[0], 1.)
        self.assertAlmostEqual(features['donor_y_delta_rating_score'].iloc[0],
                               100*(.8596173642419117-.012533469508069276))
        self.assertEqual(features['delta_january_avg_temp_c'].iloc[0], 10.)
        self.assertEqual(features['donor_rating_missing'].iloc[0], 0.)
        self.assertNotIn('id', features); self.assertNotIn('january_avg_temp_c', features)
        self.assertNotIn('energy_star_rating', features)

    def test_scaled_ridge_matches_independent_closed_form(self):
        X = frame([1, 2, 1, 2], areas=[1000., 1000., 2000., 2000.])
        model = PanelDifferenceRegressor(minimum_pairs=2).fit(X, [10., 20., 20., 40.], baseline)
        pred = model.predict(frame([3]))
        # Scaled donor_y=[-1,+1], centered delta_y=[-5,+5].
        # Ridge coef=10/(2+100)=5/51, intercept15; query donor20 => scaled+1.
        expected = (38/45)*(20+15+5/51)+(7/45)*10
        self.assertAlmostEqual(pred[0], expected, places=7)

    def test_missing_ratings_missing_weather_unseen_facility_and_serialization(self):
        X = frame([1, 2, 1, 2, 1], areas=[1000., 1000., 2000., 2000., 3000.],
                  ratings=[np.nan, 50., 50., np.nan, 50.],
                  facilities=['Office']*4+['School'])
        X['january_avg_temp_c'] = np.nan
        model = PanelDifferenceRegressor(minimum_pairs=2).fit(X, [100., 120., 200., 230., 150.], baseline)
        features = model._pair_features(X, [1, 3], [0, 2], [100., 200.])
        np.testing.assert_array_equal(features['donor_y_delta_rating_score'], [0., 0.])
        np.testing.assert_array_equal(features['donor_rating_missing'], [1., 0.])
        q = frame([3, 3], areas=[1000., 3000.], ratings=[np.nan, 50.], facilities=['Office', 'School'])
        pred = model.predict(q)
        self.assertTrue(np.isfinite(pred).all()); self.assertGreater(pred[1], 10.)
        np.testing.assert_array_equal(pred, pickle.loads(pickle.dumps(model)).predict(q))
        json.dumps(model.configuration()); json.dumps(model.audit_)

    def test_strictpast_collision_minimum_pairs_and_query_independence(self):
        X = frame([1, 3, 5, 1, 1], areas=[1000., 1000., 1000., 2000., 2000.])
        model = PanelDifferenceRegressor(minimum_pairs=2).fit(X, [100., 300., 500., 200., 999.], baseline)
        self.assertEqual([(p['donor_year'],p['current_year']) for p in model.training_pair_audit_], [(1.,3.),(3.,5.)])
        q = frame([1, 2, 3, 4, 2], areas=[1000., 1000., 1000., 1000., 2000.])
        pred = model.predict(q)
        self.assertEqual(pred[0], 10.); self.assertEqual(pred[4], 10.)
        self.assertEqual(model.last_prediction_audit_['donor_years'], [None, 1., 1., 3., None])
        np.testing.assert_array_equal(model.predict(q.iloc[[3]]), pred[[3]])
        small = PanelDifferenceRegressor().fit(X, [100., 300., 500., 200., 999.], baseline)
        np.testing.assert_array_equal(small.predict(q), [10.]*5)

    def test_zero_donor_caps_to_zero_and_reversed_rows_are_equivalent(self):
        X = frame([1, 2, 1, 2, 1], areas=[1000., 1000., 2000., 2000., 3000.])
        y = [100., 200., 200., 400., 0.]
        a = PanelDifferenceRegressor(minimum_pairs=2).fit(X, y, baseline)
        b = PanelDifferenceRegressor(minimum_pairs=2).fit(X.iloc[::-1], y[::-1], baseline)
        q = frame([3, 2], areas=[1000., 3000.])
        np.testing.assert_allclose(a.predict(q), b.predict(q))
        self.assertAlmostEqual(a.predict(q)[1], (.24)*10)
        with self.assertRaises(ValueError):
            PanelDifferenceRegressor().fit(frame([1]), [-1.])
        bad = PanelDifferenceRegressor().fit(frame([1]), [100.], lambda q: np.full(len(q), np.nan))
        with self.assertRaises(ValueError):
            bad.predict(frame([2]))

    def test_large_linear_changes_obey_fixed_multiplicative_bounds(self):
        X = frame([1, 2, 1, 2, 1], areas=[1000., 1000., 2000., 2000., 3000.])
        upper = PanelDifferenceRegressor(minimum_pairs=2).fit(X, [1., 1e6, 1., 1e6, 1.], baseline)
        lower = PanelDifferenceRegressor(minimum_pairs=2).fit(X, [1e6, 1., 1e6, 1., 1.], baseline)
        query = frame([2], areas=[3000.])
        self.assertAlmostEqual(upper.predict(query)[0], .76*np.exp(1.5)+.24*10)
        self.assertAlmostEqual(lower.predict(query)[0], .76*np.exp(-1.5)+.24*10)


if __name__ == '__main__':
    unittest.main()
