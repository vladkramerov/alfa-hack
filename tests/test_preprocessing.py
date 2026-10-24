"""Behavioral contracts for train-only, finite experiment preprocessing."""
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from experiments.preprocessing import DataPreprocessor
except ImportError:
    DataPreprocessor = None


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(DataPreprocessor, "DataPreprocessor is not implemented")

    def test_validation_cannot_change_training_median_scale_or_categories(self):
        train = pd.DataFrame({"x": [0.0, 2.0, np.nan], "kind": ["a", "b", None]})
        pre = DataPreprocessor("regression", scale=True).fit(train)
        audit = json.dumps(pre.audit_, sort_keys=True, allow_nan=False)
        out = pre.transform(pd.DataFrame({"x": [np.nan, 100.0], "kind": ["unseen", "a"]}))
        self.assertAlmostEqual(out[0, 0], 0.0, places=6)
        self.assertAlmostEqual(out[1, 0], 99 / np.sqrt(2 / 3), places=4)
        np.testing.assert_array_equal(out[0, 1:], 0)
        self.assertEqual(json.dumps(pre.audit_, sort_keys=True, allow_nan=False), audit)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape[1], len(pre.get_feature_names_out()))
        np.testing.assert_allclose(pre.transform(train), clone(pre).fit_transform(train))

    def test_constants_null_columns_infinite_inputs_and_unknown_categories_are_finite(self):
        train = pd.DataFrame({"empty": [np.nan] * 5, "constant": [7.0] * 5,
                              "x": [0, 1, np.inf, -np.inf, 2], "kind": [None] * 5})
        pre = DataPreprocessor("multiclass", gaussian=True, scale=True).fit(train)
        out = pre.transform(pd.DataFrame({"empty": [np.nan], "constant": [7],
                                          "x": [np.inf], "kind": ["new"]}))
        self.assertEqual(out.shape, (1, len(pre.get_feature_names_out())))
        self.assertTrue(np.isfinite(out).all())
        self.assertIn("empty", pre.audit_["numeric_columns"])
        json.dumps(pre.audit_, allow_nan=False)

    def test_greedy_pruning_uses_numeric_training_correlations_without_target(self):
        x = np.arange(1, 51, dtype=float)
        train = pd.DataFrame({"first": x, "negative": -2 * x,
                              "monotonic": x ** 5, "constant": np.ones(50)})
        pre = DataPreprocessor("regression", drop_correlated=True).fit(train, x)
        self.assertEqual(list(pre.get_feature_names_out()), ["first", "constant"])
        dropped = {entry["column"]: entry for entry in pre.audit_["dropped_columns"]}
        self.assertEqual(dropped["negative"]["correlated_with"], "first")
        self.assertEqual(dropped["monotonic"]["method"], "spearman")
        self.assertEqual(abs(dropped["monotonic"]["coefficient"]), 1.0)
        validation = train.iloc[:2].copy()
        validation["negative"] = [9000, 0]
        self.assertEqual(pre.transform(validation).shape[1], 2)

    def test_gaussian_transform_is_fitted_on_training_and_skips_discrete_columns(self):
        x = np.exp(np.linspace(0, 9, 200))
        train = pd.DataFrame({"skewed": x, "discrete": np.tile([0, 1], 100),
                              "constant": np.ones(200)})
        pre = DataPreprocessor("regression", gaussian=True).fit(train)
        transformed = pre.transform(train)
        self.assertLess(abs(pd.Series(transformed[:, 0]).skew()), 0.5)
        self.assertLess(abs(pre.audit_["skewness_after"]["skewed"]),
                        abs(pre.audit_["skewness_before"]["skewed"]))
        np.testing.assert_array_equal(transformed[:, 1], train["discrete"])
        frozen = json.dumps(pre.audit_, sort_keys=True)
        self.assertTrue(np.isfinite(pre.transform(train * 2)).all())
        self.assertEqual(json.dumps(pre.audit_, sort_keys=True), frozen)
        self.assertEqual(pre.audit_["gaussian_columns"], ["skewed"])

    def test_regression_engineering_handles_missing_year_and_building_aggregates(self):
        train = pd.DataFrame({"year_built": [0.0, 2000, 2010], "floor_area": [0.0, 10, 20],
                              "january_avg_temp": [0.0, 2, 4],
                              "february_avg_temp": [4.0, 6, 8],
                              "heating_degree_days": [10.0, 20, 30],
                              "cooling_degree_days": [1.0, 2, 3]})
        pre = DataPreprocessor("regression", engineer=True).fit(train)
        out = pd.DataFrame(pre.transform(train), columns=pre.get_feature_names_out())
        self.assertEqual(out.loc[0, "year_built"], 2005)
        self.assertEqual(out.loc[0, "year_built_missing"], 1)
        np.testing.assert_allclose(out["monthly_temp_mean"], [2, 4, 6])
        np.testing.assert_allclose(out["monthly_temp_spread"], [4, 4, 4])
        np.testing.assert_allclose(out["heating_cooling_total"], [11, 22, 33])
        np.testing.assert_allclose(out["floor_area_log1p"], np.log1p([0, 10, 20]))

    def test_multiclass_engineering_ratios_handle_zero_denominators(self):
        train = pd.DataFrame({"src_bytes": [10.0, 0, 30], "dst_bytes": [0.0, 10, 3],
                              "src_packets": [0.0, 2, 3], "dst_packets": [0.0, 1, 1]})
        pre = DataPreprocessor("multiclass", engineer=True).fit(train)
        out = pd.DataFrame(pre.transform(train), columns=pre.get_feature_names_out())
        self.assertTrue(np.isfinite(out.to_numpy()).all())
        self.assertEqual(out.loc[0, "src_bytes_per_packet"], 10)
        self.assertEqual(out.loc[2, "src_bytes_per_packet"], 10)
        np.testing.assert_allclose(out["src_bytes_log1p"], np.log1p([10, 0, 30]))
        self.assertNotIn("id", out.columns)
        self.assertNotIn("label", out.columns)

    def test_dataset_column_names_produce_domain_features(self):
        regression = pd.DataFrame({"january_avg_temp_c": [1, 3],
                                   "february_avg_temp_c": [5, 7],
                                   "heating_degree_days_c": [5, 6],
                                   "cooling_degree_days_c": [7, 8]})
        reg = DataPreprocessor("regression", engineer=True)
        reg_out = pd.DataFrame(reg.fit_transform(regression), columns=reg.get_feature_names_out())
        np.testing.assert_allclose(reg_out["monthly_temp_mean"], [3, 5])
        np.testing.assert_allclose(reg_out["heating_cooling_total"], [12, 14])
        network = pd.DataFrame({"sbytes": [10, 12], "dbytes": [5, 8],
                                "spkts": [0, 3], "dpkts": [1, 2]})
        pre = DataPreprocessor("multiclass", engineer=True)
        out = pd.DataFrame(pre.fit_transform(network), columns=pre.get_feature_names_out())
        np.testing.assert_allclose(out["src_bytes_per_packet"], [10, 4])
        np.testing.assert_allclose(out["dst_bytes_per_packet"], [5, 4])
        np.testing.assert_allclose(out["sbytes_log1p"], np.log1p([10, 12]))

    def test_missing_flags_are_selected_only_from_training_and_inputs_are_unchanged(self):
        train = pd.DataFrame({"complete": [1.0, 2, 3], "missing": [np.nan, 2, 3]})
        before = train.copy(deep=True)
        pre = DataPreprocessor("regression", engineer=True).fit(train)
        validation = pd.DataFrame({"complete": [np.nan], "missing": [np.nan]})
        out = pd.DataFrame(pre.transform(validation), columns=pre.get_feature_names_out())
        self.assertNotIn("complete_missing", out)
        self.assertEqual(out.loc[0, "missing_missing"], 1)
        self.assertEqual(out.loc[0, "complete"], 2)
        pd.testing.assert_frame_equal(train, before)

    def test_known_identifier_and_target_columns_are_rejected(self):
        for column in ("id", "site_eui", "attack_cat", "label", "target"):
            with self.subTest(column=column), self.assertRaises(ValueError):
                DataPreprocessor("regression").fit(pd.DataFrame({column: [0, 1]}))

    def test_residual_correlation_audit_describes_retained_training_predictors(self):
        train = pd.DataFrame({"x": [1, 2, 3, 4], "duplicate": [2, 4, 6, 8],
                              "z": [1, 2, 4, 3]})
        pre = DataPreprocessor("regression", drop_correlated=True).fit(train)
        self.assertAlmostEqual(pre.audit_["residual_max_abs_correlation"]["pearson"], 0.8)
        self.assertAlmostEqual(pre.audit_["residual_max_abs_correlation"]["spearman"], 0.8)

    def test_vif_removes_exact_joint_dependency_that_pairwise_filter_cannot_detect(self):
        x = np.tile([-1.0, -1, 1, 1], 30)
        z = np.tile([-1.0, 1, -1, 1], 30)
        train = pd.DataFrame({"x": x, "z": z, "sum": x + z})
        pairwise = DataPreprocessor("regression", drop_correlated=True).fit(train)
        self.assertEqual(len(pairwise.get_feature_names_out()), 3)
        pre = DataPreprocessor("regression", drop_correlated=True, drop_vif=True).fit(train)
        self.assertEqual(list(pre.get_feature_names_out()), ["x", "z"])
        self.assertEqual(pre.audit_["numeric_vif_before"]["sum"], "infinity")
        self.assertEqual(pre.audit_["vif_dropped_columns"][0]["column"], "sum")
        self.assertEqual(pre.audit_["vif_dropped_columns"][0]["reason"], "singular_dependency")
        self.assertAlmostEqual(pre.audit_["numeric_vif_after"]["x"], 1)
        self.assertAlmostEqual(pre.audit_["numeric_vif_after"]["z"], 1)
        json.dumps(pre.audit_, allow_nan=False)

    def test_vif_iteratively_removes_high_but_nonsingular_joint_collinearity(self):
        a = np.tile([-1.0, -1, -1, -1, 1, 1, 1, 1], 30)
        b = np.tile([-1.0, -1, 1, 1, -1, -1, 1, 1], 30)
        noise = np.tile([-1.0, 1, -1, 1, -1, 1, -1, 1], 30)
        train = pd.DataFrame({"a": a, "b": b, "almost_sum": a + b + 0.1 * noise,
                              "constant": np.ones(len(a)), "empty": np.nan})
        pre = DataPreprocessor("regression", drop_vif=True).fit(train)
        self.assertEqual(list(pre.get_feature_names_out()), ["a", "b"])
        self.assertAlmostEqual(pre.audit_["numeric_vif_before"]["almost_sum"], 201, places=7)
        self.assertTrue(all(value <= 10 for value in pre.audit_["numeric_vif_after"].values()))
        dropped = {entry["column"]: entry for entry in pre.audit_["vif_dropped_columns"]}
        self.assertEqual(dropped["constant"]["reason"], "constant")
        self.assertEqual(dropped["empty"]["reason"], "constant")
        self.assertEqual(dropped["almost_sum"]["reason"], "vif_above_threshold")

    def test_vif_selection_is_training_only_and_ties_remove_last_input_column(self):
        train = pd.DataFrame({"first": [-1., -1, 1, 1], "last": [-1., -1, 1, 1],
                              "independent": [-1., 1, -1, 1]})
        pre = DataPreprocessor("multiclass", drop_vif=True, vif_threshold=5).fit(train)
        self.assertEqual(list(pre.get_feature_names_out()), ["first", "independent"])
        state = json.dumps(pre.audit_, sort_keys=True, allow_nan=False)
        validation = pd.DataFrame({"first": [np.nan, 99], "last": [9., -9],
                                   "independent": [9., -9]})
        out = pre.transform(validation)
        np.testing.assert_allclose(out, [[0, 9], [99, -9]])
        self.assertEqual(json.dumps(pre.audit_, sort_keys=True, allow_nan=False), state)
        repeat = clone(pre).fit(train)
        self.assertEqual(repeat.audit_, pre.audit_)


if __name__ == "__main__":
    unittest.main()
