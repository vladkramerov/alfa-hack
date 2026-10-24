"""Synthetic invariants for new unsupervised HYP-022..032 transformers."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from experiments.twenty_features import make_features
except ImportError:
    make_features = None


class TwentyFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(make_features, "new hypothesis feature transformers are missing")

    def test_winsorization_preserves_rows_and_uses_only_train_bounds(self):
        pre = make_features(22).fit(pd.DataFrame({"x": np.arange(201.), "cat": ["a"] * 201}))
        probe = pd.DataFrame({"x": [-999., 999., np.nan], "cat": ["z", "a", None]}, index=[3, 3, 1])
        out = pre.transform(probe)
        self.assertEqual(out.index.tolist(), [3, 3, 1])
        np.testing.assert_allclose(out.x[:2], [1, 199])
        self.assertTrue(np.isnan(out.x.iloc[2]))
        pd.testing.assert_series_equal(out.cat, probe.cat)

    def test_iterative_imputation_keeps_categories_and_empty_numeric_columns(self):
        train = pd.DataFrame({"x": [1., 2, 3, np.nan, 5], "z": [2., 4, 6, 8, 10],
                              "empty": [np.nan] * 5, "cat": ["a", None, "b", "a", "b"]})
        pre = make_features(23)
        out = pre.fit_transform(train, np.array([99, -7, 5, 8, 0]))
        self.assertEqual(out.columns.tolist(), train.columns.tolist())
        self.assertTrue(np.isfinite(out.select_dtypes("number")).all().all())
        self.assertAlmostEqual(out.x.iloc[3], 4., places=3)
        pd.testing.assert_series_equal(out.cat, train.cat)
        other = clone(pre).fit_transform(train, np.arange(5))
        pd.testing.assert_frame_equal(out, other)

    def test_group_median_hierarchy_counts_available_values(self):
        train = pd.DataFrame({"State_Factor": ["s1"] * 100 + ["s2"] * 100 + ["s3"] * 99,
                              "facility_type": ["office"] * 200 + ["shop"] * 99,
                              "x": [10.] * 100 + [30.] * 100 + [100.] * 99})
        pre = make_features(24).fit(train)
        probe = pd.DataFrame({"State_Factor": ["s1", "new", "s3", "new"],
                              "facility_type": ["office", "office", "shop", "unknown"],
                              "x": [np.nan] * 4}, index=[9, 9, 0, 8])
        out = pre.transform(probe)
        np.testing.assert_allclose(out.x, [10., 20., 30., 30.])
        np.testing.assert_array_equal(out.x__missing, [1, 1, 1, 1])
        self.assertEqual(out.index.tolist(), probe.index.tolist())

    def test_conditional_gaussian_has_protocol_scales_and_unknown_global_fallback(self):
        train = pd.DataFrame({"proto": ["tcp"] * 500 + ["udp"] * 500,
                              "x": np.r_[np.arange(500.), np.arange(1000., 1500.)],
                              "binary": [0, 1] * 500})
        pre = make_features(25).fit(train)
        probe = pd.DataFrame({"proto": ["tcp", "udp", "unknown", "unknown"],
                              "x": [250., 1250., -1e300, 1e300], "binary": [0, 1, 0, 1]})
        out = pre.transform(probe)
        self.assertLess(abs(out.x.iloc[0]), .02)
        self.assertLess(abs(out.x.iloc[1]), .02)
        self.assertTrue(np.isfinite(out.x).all())
        np.testing.assert_array_equal(out.binary, probe.binary)
        pd.testing.assert_frame_equal(out, pre.transform(probe))

    def test_zero_atom_keeps_zero_distinct_and_does_not_transform_negative_features(self):
        train = pd.DataFrame({"x": np.r_[np.zeros(50), np.arange(1., 51)],
                              "signed": np.arange(-50., 50), "allzero": np.zeros(100)})
        pre = make_features(26).fit(train)
        out = pre.transform(pd.DataFrame({"x": [0., 25., 1e300], "signed": [-1., 2, 3],
                                          "allzero": [0., 0, 0]}))
        self.assertEqual(out.x.iloc[0], 0)
        np.testing.assert_array_equal(out.x__is_zero, [1, 0, 0])
        self.assertNotIn("signed__is_zero", out)
        self.assertNotIn("allzero__is_zero", out)
        self.assertTrue(np.isfinite(out.x).all())

    def test_tcp_applicability_preserves_observed_tcp_zero(self):
        frame = pd.DataFrame({"proto": ["tcp", "udp", None], "swin": [0., 88., 0.], "dur": [0., 2, 3]}, index=[4, 4, 1])
        out = make_features(27).fit_transform(frame)
        self.assertEqual(out.swin.iloc[0], 0)
        self.assertTrue(out.swin.iloc[1:].isna().all())
        np.testing.assert_array_equal(out.swin__applicable, [1, 0, 0])
        pd.testing.assert_series_equal(out.dur, frame.dur)
        self.assertEqual(out.index.tolist(), frame.index.tolist())

    def test_wind_cycle_is_continuous_at_north_and_preserves_missing(self):
        frame = pd.DataFrame({"direction_max_wind_speed": [359., 1., np.nan], "speed": [1., 2., 3.]})
        out = make_features(29).fit_transform(frame)
        self.assertLess(abs(out.direction_max_wind_speed__sin.iloc[0] - out.direction_max_wind_speed__sin.iloc[1]), .04)
        self.assertAlmostEqual(out.direction_max_wind_speed__cos.iloc[0], out.direction_max_wind_speed__cos.iloc[1])
        self.assertTrue(np.isnan(out.direction_max_wind_speed__sin.iloc[2]))
        pd.testing.assert_frame_equal(out[frame.columns], frame)

    def test_dft_coefficients_respect_month_order_and_train_only_imputation(self):
        months = "january february march april may june july august september october november december".split()
        wave = 10 + 3 * np.cos(2 * np.pi * np.arange(12) / 12) + 2 * np.sin(4 * np.pi * np.arange(12) / 12)
        frame = pd.DataFrame({m + "_avg_temp_c": [v, v] for m, v in zip(months, wave)})
        pre = make_features(30).fit(frame)
        probe = frame.iloc[[0]].copy()
        probe.loc[:, "january_avg_temp_c"] = np.nan
        out = pre.transform(probe)
        self.assertAlmostEqual(out.monthly_temp__cos1.iloc[0], 3.)
        self.assertAlmostEqual(out.monthly_temp__sin2.iloc[0], 2.)
        self.assertAlmostEqual(out.monthly_temp__sin1.iloc[0], 0.)
        self.assertAlmostEqual(out.monthly_temp__cos2.iloc[0], 0.)
        self.assertTrue(np.isnan(out.january_avg_temp_c.iloc[0]))

    def test_group_relative_keeps_original_and_uses_training_iqr(self):
        frame = pd.DataFrame({"State_Factor": ["a"] * 100, "facility_type": ["office"] * 100,
                              "floor_area": np.tile([0., 10., 20., 30.], 25)})
        pre = make_features(31).fit(frame)
        probe = pd.DataFrame({"State_Factor": ["new"], "facility_type": ["unknown"], "floor_area": [30.]})
        out = pre.transform(probe)
        self.assertEqual(out.floor_area.iloc[0], 30.)
        self.assertAlmostEqual(out.floor_area__group_relative.iloc[0], 1.)

    def test_weather_area_interactions_learn_only_training_classes(self):
        frame = pd.DataFrame({"floor_area": [np.e - 1, np.e - 1], "heating_degree_days_c": [3., 5.],
                              "cooling_degree_days_c": [2., 7.], "building_class": ["Residential", "Commercial"]})
        pre = make_features(32).fit(frame)
        probe = frame.iloc[[0]].copy()
        probe["building_class"] = "new"
        out = pre.transform(probe)
        self.assertAlmostEqual(out.heating_degree_days_c__log_area.iloc[0], 3.)
        class_features = [c for c in out if "__class_" in c]
        self.assertEqual(len(class_features), 4)
        self.assertEqual(float(out[class_features].sum().sum()), 0.)

    def test_factory_unknown_is_explicit_error(self):
        with self.assertRaises(ValueError):
            make_features(999)

    def test_schema_mismatch_is_error_instead_of_refitting(self):
        pre = make_features(22).fit(pd.DataFrame({"x": [1., 2.]}))
        with self.assertRaises(ValueError):
            pre.transform(pd.DataFrame({"other": [1.]}))

    def test_all_transformers_are_unsupervised_cloneable_and_leave_input_untouched(self):
        frame = pd.DataFrame({"x": [1., np.nan, 3.], "cat": ["a", None, "b"]}, index=[7, 7, 2])
        original = frame.copy(deep=True)
        for number in (22, 23, 24, 25, 26, 27, 29, 30, 31, 32):
            with self.subTest(hypothesis=number):
                pre = make_features(number)
                out = pre.fit_transform(frame, [0., 100., -20.])
                other = clone(pre).fit_transform(frame, [7., 9., 111.])
                pd.testing.assert_frame_equal(out, other)
                pd.testing.assert_frame_equal(frame, original)
                self.assertEqual(out.index.tolist(), frame.index.tolist())
                self.assertEqual(out.columns.tolist(), pre.get_feature_names_out().tolist())
                pd.testing.assert_series_equal(out.cat, frame.cat)

    def test_gaussian_small_protocol_uses_same_global_mapping_as_unknown(self):
        frame = pd.DataFrame({"proto": ["tcp"] * 500 + ["rare"] * 30,
                              "x": np.arange(530.)})
        pre = make_features(25).fit(frame)
        probe = pd.DataFrame({"proto": ["rare", "unseen"], "x": [510., 510.]})
        out = pre.transform(probe)
        self.assertAlmostEqual(out.x.iloc[0], out.x.iloc[1])

    def test_generated_name_collision_fails_before_overwriting_input(self):
        frame = pd.DataFrame({"direction_max_wind_speed": [0., 90.],
                              "direction_max_wind_speed__sin": [123., 456.]})
        with self.assertRaises(ValueError):
            make_features(29).fit(frame)


if __name__ == "__main__":
    unittest.main()
