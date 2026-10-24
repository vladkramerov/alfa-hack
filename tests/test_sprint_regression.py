import json
import unittest

import numpy as np
import pandas as pd

from sprint_regression import TemporalEntityLinker, RatingNormalizer, RatingNormalizedRegressor


def frame(years, ratings=None, areas=None, elevation=None):
    n = len(years)
    return pd.DataFrame({'Year_Factor': years, 'State_Factor': ['S'] * n,
        'building_class': ['Commercial'] * n, 'facility_type': ['Office'] * n,
        'floor_area': areas if areas is not None else [1000.] * n,
        'year_built': [1950.] * n, 'ELEVATION': elevation if elevation is not None else [10.] * n,
        'energy_star_rating': ratings if ratings is not None else [50.] * n})


class SprintRegressionTests(unittest.TestCase):
    def test_temporal_linker_excludes_same_and_future_and_links_changed_elevation(self):
        X = frame([1, 3, 5], elevation=[10, 20, 30])
        model = TemporalEntityLinker(alpha=.8).fit(X, [100., 300., 500.], base_predictor=lambda q: np.full(len(q), 20.))
        result = model.predict(frame([1, 2, 4], elevation=[999, 999, 999]))
        self.assertEqual(result[0], 20.)
        self.assertGreater(result[1], 20.)
        self.assertLess(result[1], 100.)
        self.assertGreater(result[2], 100.)
        self.assertLess(result[2], 300.)

    def test_ambiguous_profile_falls_back_and_missing_is_not_wildcard(self):
        X = frame([1, 1, 2], areas=[1000, 1000, 2000])
        model = TemporalEntityLinker().fit(X, [100., 900., 200.], base_predictor=lambda q: np.full(len(q), 7.))
        q = frame([3, 3], areas=[1000, 2000]); q.loc[1, 'year_built'] = np.nan
        np.testing.assert_array_equal(model.predict(q), [7., 7.])

    def test_rating_transfer_learns_direction_from_past_pairs_and_is_order_invariant(self):
        X = frame([1, 2, 1, 2], ratings=[20, 80, 20, 80], areas=[1000, 1000, 2000, 2000])
        y = np.array([200., 80., 300., 120.])
        model = TemporalEntityLinker().fit(X, y, base_predictor=lambda q: np.full(len(q), 100.))
        q = frame([3, 3], ratings=[20, 80])
        pred = model.predict(q)
        self.assertGreater(pred[0], pred[1])
        reverse = TemporalEntityLinker().fit(X.iloc[::-1], y[::-1], base_predictor=lambda q: np.full(len(q), 100.))
        np.testing.assert_allclose(pred, reverse.predict(q))
        json.dumps(model.configuration()); json.dumps(model.audit_)

    def test_normalizer_monotone_missing_and_unknown_groups(self):
        X = frame([1] * 100, ratings=np.tile([10., 90.], 50), areas=np.arange(100))
        y = np.tile([200., 50.], 50)
        normalizer = RatingNormalizer().fit(X, y)
        q = frame([2, 2, 2], ratings=[10, 90, np.nan]); q['facility_type'] = 'Unseen'
        scale = normalizer.transform(q)
        self.assertGreater(scale[0], scale[1]); self.assertEqual(scale[2], 1.)
        np.testing.assert_allclose(scale[:2], [2., .5], atol=1e-6)

    def test_weighted_normalized_error_is_raw_squared_error(self):
        y = np.array([10., 20., 300.]); g = np.array([.5, 2., 4.]); zhat = np.array([18., 12., 60.])
        z, weight = RatingNormalizedRegressor.normalized_targets(y, g)
        np.testing.assert_allclose(weight * (z - zhat) ** 2, (y - g * zhat) ** 2)

    def test_temporal_normalization_never_uses_own_year_targets(self):
        X = frame([1, 1, 2, 2, 3, 3], ratings=[10, 90] * 3)
        y = np.array([200., 50., 180., 60., 160., 70.])
        a = RatingNormalizedRegressor(parameters={'n_estimators': 3}, features=()).fit(X, y)
        other = y.copy(); other[4:] *= 10
        b = RatingNormalizedRegressor(parameters={'n_estimators': 3}, features=()).fit(X, other)
        np.testing.assert_array_equal(a.training_scales_, b.training_scales_)
        np.testing.assert_array_equal(a.training_scales_[:2], [1., 1.])
        q = frame([4, 4], ratings=[10, np.nan]); q['facility_type'] = 'new'
        self.assertTrue(np.isfinite(a.predict(q)).all())
        self.assertEqual(len(a.predict(q)), 2)


if __name__ == '__main__':
    unittest.main()
