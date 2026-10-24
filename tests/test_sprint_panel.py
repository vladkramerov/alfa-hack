import json
import pickle
import unittest

import numpy as np
import pandas as pd

from sprint_panel import PanelTransitionRegressor


def frame(years, areas=None, ratings=None):
    n = len(years)
    return pd.DataFrame({
        'id': np.arange(n) + 100, 'Year_Factor': years, 'State_Factor': ['S'] * n,
        'building_class': ['Commercial'] * n, 'facility_type': ['Office'] * n,
        'floor_area': areas if areas is not None else [1000.] * n,
        'year_built': [1950.] * n,
        'energy_star_rating': ratings if ratings is not None else [50.] * n,
        'january_avg_temp': np.asarray(years, float) * 10,
        'heating_degree_days': np.asarray(years, float) * 100,
    })


def baseline(X):
    return np.full(len(X), 7.)


class SprintPanelTests(unittest.TestCase):
    def test_strict_past_pairs_and_query_donors_exclude_same_future(self):
        X = frame([5, 1, 3])
        model = PanelTransitionRegressor(minimum_pairs=20).fit(X, [500., 100., 300.], baseline)
        self.assertEqual(model.audit_['training_pairs'], 2)
        self.assertEqual([(p['donor_year'], p['current_year']) for p in model.training_pair_audit_],
                         [(1., 3.), (3., 5.)])
        result = model.predict(frame([1, 2, 4, 5, 6]))
        np.testing.assert_array_equal(result, [7., 7., 7., 7., 7.])
        self.assertEqual(model.last_prediction_audit_['donor_years'], [None, 1., 3., 3., 5.])

    def test_collision_rejects_whole_profile_and_missing_is_not_wildcard(self):
        model = PanelTransitionRegressor(minimum_pairs=20).fit(
            frame([1, 1, 2, 1, 2], areas=[1000, 1000, 1000, 2000, 2000]),
            [100., 999., 200., 10., 20.], baseline)
        self.assertEqual(model.audit_['rejected_ambiguous_profiles'], 1)
        self.assertEqual(model.audit_['training_pairs'], 1)
        q = frame([3, 3, 3], areas=[1000, 2000, 2000])
        q.loc[2, 'year_built'] = np.nan
        np.testing.assert_array_equal(model.predict(q), [7., 7., 7.])
        self.assertEqual(model.last_prediction_audit_['donor_years'], [None, 2., None])

    def test_features_use_past_target_weather_and_missing_rating_without_id(self):
        X = frame([1, 2, 3], ratings=[20., np.nan, 80.])
        model = PanelTransitionRegressor(minimum_pairs=20).fit(X, [100., 200., 300.], baseline)
        pairs = model._pair_frame(X, np.array([1, 2]), np.array([0, 1]), np.array([100., 200.]))
        np.testing.assert_array_equal(pairs['panel_donor_target'], [100., 200.])
        np.testing.assert_array_equal(pairs['panel_age'], [1., 1.])
        np.testing.assert_array_equal(pairs['panel_delta_january_avg_temp'], [10., 10.])
        self.assertTrue(np.isnan(pairs['panel_rating_delta']).all())
        self.assertEqual(pairs['panel_donor_rating_missing'].tolist(), [False, True])
        self.assertNotIn('id', pairs.columns)
        self.assertNotIn('site_eui', pairs.columns)

    def test_real_learner_missing_values_unseen_category_and_pickle(self):
        areas = np.repeat(np.arange(12) + 1000., 3)
        years = np.tile([1, 2, 3], 12)
        ratings = np.tile([20., np.nan, 80.], 12)
        X = frame(years, areas=areas, ratings=ratings)
        y = 100. + np.repeat(np.arange(12), 3) * 2 + years * 10
        model = PanelTransitionRegressor(parameters={'iterations': 12}, minimum_pairs=4,
                                         threads=1).fit(X, y, baseline)
        q = frame([4, 4], areas=[1000., 99999.], ratings=[np.nan, 80.])
        q.loc[1, 'facility_type'] = 'UNSEEN'
        prediction = model.predict(q)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreater(prediction[0], 7.)
        self.assertEqual(prediction[1], 7.)
        self.assertEqual(model.last_prediction_audit_['transition_rows'], 1)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(q), prediction)
        changed_ids = q.copy(); changed_ids['id'] += 10000
        np.testing.assert_array_equal(restored.predict(changed_ids), prediction)
        repeated = frame([4, 5], areas=[1000., 1000.], ratings=[np.nan, np.nan])
        batch = restored.predict(repeated)
        self.assertEqual(restored.last_prediction_audit_['donor_years'], [3., 3.])
        np.testing.assert_array_equal(restored.predict(repeated.iloc[[1]]), batch[[1]])
        json.dumps(model.configuration()); json.dumps(model.audit_)

    def test_row_order_does_not_change_supervised_transition_and_zero_alpha_is_baseline(self):
        X = frame(np.tile([1, 2, 3], 5), areas=np.repeat(np.arange(5) + 1000., 3))
        y = np.arange(15, dtype=float) + 20.
        a = PanelTransitionRegressor(parameters={'iterations': 8}, minimum_pairs=2, threads=1).fit(X, y, baseline)
        b = PanelTransitionRegressor(parameters={'iterations': 8}, minimum_pairs=2, threads=1).fit(
            X.iloc[::-1], y[::-1], baseline)
        q = frame([4, 4], areas=[1000., 1001.])
        np.testing.assert_array_equal(a.predict(q), b.predict(q))
        a.alpha = 0.
        np.testing.assert_array_equal(a.predict(q), [7., 7.])

    def test_invalid_target_and_baseline_are_rejected(self):
        with self.assertRaises(ValueError):
            PanelTransitionRegressor().fit(frame([1]), [-1.])
        model = PanelTransitionRegressor().fit(frame([1]), [10.],
                                              lambda q: np.full(len(q), np.nan))
        with self.assertRaises(ValueError):
            model.predict(frame([2]))


if __name__ == '__main__':
    unittest.main()
