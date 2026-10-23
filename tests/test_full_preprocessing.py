"""Synthetic behavioral checks for mandatory train-only full-v1 processing."""
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from automl_preprocessing import FullPreprocessor, TrainingOutlierFilter
except ImportError:
    FullPreprocessor = TrainingOutlierFilter = None


class FullPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(FullPreprocessor, "FullPreprocessor is not implemented")

    def test_train_quantiles_and_all_numeric_columns_are_normalized(self):
        train = pd.DataFrame({"skewed": np.exp(np.linspace(0, 10, 300)),
                              "binary": np.tile([0., 1., 1.], 100),
                              "missing": np.r_[np.nan, np.arange(299, dtype=float)]})
        pre = FullPreprocessor(native=True)
        out = pre.fit_transform(train)
        self.assertIn("skewed", pre.continuous_columns_)
        self.assertNotIn("binary", pre.continuous_columns_)
        self.assertIn("missing__missing", out)
        np.testing.assert_allclose(out.mean().to_numpy(), 0, atol=1e-12)
        np.testing.assert_allclose(out.std(ddof=0).to_numpy(), 1, atol=1e-12)
        self.assertLess(abs(out["skewed"].skew()), .05)
        self.assertGreater(pre.audit_["skewness_before"]["skewed"], 2)
        self.assertTrue(pre.audit_["normalization_enabled"])
        json.dumps(pre.audit_, allow_nan=False)

    def test_validation_cannot_change_statistics_and_extreme_values_are_finite(self):
        train = pd.DataFrame({"x": np.r_[np.nan, np.exp(np.linspace(0, 4, 99))],
                              "discrete": np.tile([0, 1], 50), "cat": ["a", "b"] * 50})
        pre = FullPreprocessor().fit(train)
        state = json.dumps(pre.audit_, sort_keys=True, allow_nan=False)
        quantiles = pre.quantile_.quantiles_.copy()
        scaler_mean = pre.scaler_.mean_.copy()
        probe = pd.DataFrame({"x": [np.inf, -1e300, 1e300],
                              "discrete": [1e300, -1e300, np.nan], "cat": ["new", None, "a"]},
                             index=[8, 8, 2])
        out = pre.transform(probe)
        self.assertEqual(out.shape[0], 3)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out[:1], pre.transform(probe.iloc[:1]))
        self.assertEqual(json.dumps(pre.audit_, sort_keys=True, allow_nan=False), state)
        np.testing.assert_array_equal(pre.quantile_.quantiles_, quantiles)
        np.testing.assert_array_equal(pre.scaler_.mean_, scaler_mean)
        np.testing.assert_allclose(clone(pre).fit_transform(train), pre.transform(train))

    def test_drop_empty_constant_and_duplicate_numeric_but_keep_correlated_features(self):
        train = pd.DataFrame({"x": np.arange(40, dtype=float),
                              "copy": np.arange(40), "twice": np.arange(40) * 2,
                              "constant": [7] * 40, "empty": [np.nan] * 40})
        pre = FullPreprocessor(native=True).fit(train)
        self.assertEqual(pre.numeric_, ["x", "twice"])
        self.assertEqual(set(d["column"] for d in pre.audit_["dropped_columns"]),
                         {"copy", "constant", "empty"})
        self.assertEqual(pre.transform(train).shape, (40, 2))

    def test_missing_and_literal_sentinels_remain_distinct_in_native_categories(self):
        train = pd.DataFrame({"cat": [None, "__MISSING__", "a", "a"], "n": [1, 2, 3, 4]},
                             index=[3, 3, 8, 9])
        pre = FullPreprocessor(native=True).fit(train)
        out = pre.transform(train)
        self.assertTrue(out.index.equals(train.index))
        self.assertNotEqual(out.iloc[0]["cat"], out.iloc[1]["cat"])
        probe = pd.DataFrame({"cat": ["new", None], "n": [4, np.inf]}, index=[4, 2])
        transformed = pre.transform(probe)
        self.assertTrue(transformed.index.equals(probe.index))
        self.assertFalse(transformed["cat"].isna().any())
        self.assertTrue(np.isfinite(transformed.select_dtypes("number").to_numpy()).all())

    def test_category_only_all_null_and_high_cardinality_inputs_are_usable(self):
        for native in (False, True):
            train = pd.DataFrame({"empty": [np.nan] * 4})
            pre = FullPreprocessor(native=native).fit(train)
            values = pre.transform(train)
            self.assertEqual(values.shape, (4, 1))
            self.assertTrue(np.isfinite(np.asarray(values)).all())
            cats = pd.DataFrame({"category": ["a", "b", None, "__MISSING__"]})
            pre = FullPreprocessor(native=native).fit(cats)
            out = pre.transform(pd.DataFrame({"category": ["new", None]}))
            self.assertEqual(len(out), 2)
            json.dumps(pre.audit_, allow_nan=False)
        large = pd.DataFrame({"category": ["c" + str(i) for i in range(1000)]})
        pre = FullPreprocessor().fit(large)
        out = pre.transform(large)
        self.assertLessEqual(out.shape[1], 64)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out.shape[1], len(pre.get_feature_names_out()))
        probe = pre.transform(pd.DataFrame({"category": [None, "c0", "unseen"]}))
        self.assertEqual(probe[0].sum(), 1)
        self.assertEqual(probe[1].sum(), 1)
        self.assertFalse(np.array_equal(probe[0], probe[1]))
        self.assertEqual(probe[2].sum(), 0)

    def test_missing_flags_use_training_columns_and_discrete_ties_skip_gaussianization(self):
        train = pd.DataFrame({"mostly_zero": np.r_[np.zeros(170), np.arange(1, 31)],
                              "ordinary": np.arange(200, dtype=float)})
        pre = FullPreprocessor(native=True).fit(train)
        self.assertNotIn("mostly_zero", pre.continuous_columns_)
        probe = train.iloc[:2].copy()
        probe["ordinary"] = np.nan
        out = pre.transform(probe)
        self.assertNotIn("ordinary__missing", out)
        self.assertTrue(np.isfinite(out.to_numpy()).all())

    def test_categorical_cleanup_drops_null_constant_and_exact_tagged_duplicates(self):
        train = pd.DataFrame({"first": ["a", None, "__MISSING__", "b"],
                              "copy": ["a", pd.NA, "__MISSING__", "b"],
                              "constant": ["a"] * 4, "empty": [None] * 4,
                              "missing_informative": [None, "a", "a", "a"]})
        pre = FullPreprocessor(native=True).fit(train)
        self.assertEqual(pre.categorical_, ["first", "missing_informative"])
        drops = {d["column"]: d for d in pre.audit_["dropped_columns"]}
        self.assertEqual(drops["copy"]["reason"], "exact_duplicate")
        self.assertEqual(drops["copy"]["duplicate_of"], "first")
        self.assertEqual(drops["constant"]["reason"], "constant")
        self.assertEqual(drops["empty"]["reason"], "all_null")
        out = pre.transform(train)
        self.assertEqual(list(out), ["first", "missing_informative"])
        self.assertNotEqual(out.iloc[0]["first"], out.iloc[2]["first"])

    def test_all_constant_categories_fall_back_and_ood_cannot_restore_dropped_schema(self):
        train = pd.DataFrame({"constant": ["a"] * 4, "empty": [None] * 4}, index=[2] * 4)
        probe = pd.DataFrame({"constant": ["new", None], "empty": ["different", "values"]}, index=[8, 3])
        for native in (False, True):
            pre = FullPreprocessor(native=native).fit(train)
            state = json.dumps(pre.audit_, sort_keys=True, allow_nan=False)
            out = pre.transform(probe)
            self.assertEqual(out.shape, (2, 1))
            np.testing.assert_array_equal(np.asarray(out), 0)
            self.assertTrue(pre.audit_["fallback_bias"])
            self.assertFalse(pre.categorical_)
            self.assertEqual(json.dumps(pre.audit_, sort_keys=True, allow_nan=False), state)
            if native:
                self.assertTrue(out.index.equals(probe.index))


class TrainingOutlierTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TrainingOutlierFilter, "TrainingOutlierFilter is not implemented")
        rng = np.random.RandomState(7)
        self.X = pd.DataFrame({"x": np.r_[rng.normal(0, .1, 190), np.arange(10) * 20 + 100],
                               "z": rng.normal(size=200)}, index=[15] * 200)

    def test_regression_removes_worst_negative_scores_with_exact_global_cap_without_y(self):
        y = pd.Series(np.arange(200), index=self.X.index)
        detector = TrainingOutlierFilter()
        X_out, y_out, keep = detector.fit_filter(self.X, y)
        self.assertEqual(keep.dtype, bool)
        self.assertEqual(keep.shape, (200,))
        self.assertGreaterEqual((detector.scores_ < 0).sum(), 10)
        self.assertEqual((~keep).sum(), 10)
        np.testing.assert_array_equal(np.sort(detector.removed_positions_),
                                      np.sort(np.argsort(detector.scores_, kind="stable")[:10]))
        pd.testing.assert_frame_equal(X_out, self.X.iloc[np.flatnonzero(keep)])
        pd.testing.assert_series_equal(y_out, y.iloc[np.flatnonzero(keep)])
        other = TrainingOutlierFilter()
        other.fit_filter(self.X, np.arange(200) ** 3)
        np.testing.assert_array_equal(other.keep_mask_, keep)
        np.testing.assert_allclose(other.scores_, detector.scores_)
        self.assertLessEqual(detector.audit_["removed_fraction"], .05)
        json.dumps(detector.audit_, allow_nan=False)

    def test_classification_caps_each_class_and_protects_rare_class(self):
        y = np.array(["major"] * 150 + ["medium"] * 40 + ["rare"] * 10)
        detector = TrainingOutlierFilter()
        _, kept_y, keep = detector.fit_filter(self.X, y, classification=True)
        self.assertLessEqual((~keep).sum(), 10)
        self.assertLessEqual(((y == "major") & ~keep).sum(), 7)
        self.assertTrue(keep[y == "medium"].all())
        self.assertTrue(keep[y == "rare"].all())
        self.assertEqual((kept_y == "medium").sum(), 40)
        self.assertEqual((kept_y == "rare").sum(), 10)
        json.dumps(detector.audit_, allow_nan=False)

    def test_detector_uses_imputed_numeric_inputs_and_retains_fifty_small_class_rows(self):
        X = self.X.copy()
        X["empty"] = np.nan
        X["constant"] = 3.
        X["cat"] = ["a"] * len(X)
        y = np.array(["major"] * 149 + ["guarded"] * 51)
        detector = TrainingOutlierFilter()
        _, retained, keep = detector.fit_filter(X, y, classification=True)
        self.assertEqual(detector.detector_.n_features_in_, 4)
        self.assertEqual(detector.detector_imputer_.statistics_[2], 0)
        self.assertGreaterEqual((retained == "guarded").sum(), 50)
        self.assertLessEqual(((y == "guarded") & ~keep).sum(), 1)

    def test_zero_budget_and_no_varying_numeric_features_do_not_remove_rows(self):
        for X in (pd.DataFrame({"cat": ["a", "b"] * 10}),
                  pd.DataFrame({"x": [np.nan] * 20, "constant": [3] * 20})):
            detector = TrainingOutlierFilter()
            X_out, y_out, keep = detector.fit_filter(X, np.arange(20))
            self.assertTrue(keep.all())
            self.assertEqual(len(X_out), len(X))
            self.assertIn("no_varying_numeric", detector.audit_["reason"])
            self.assertFalse(detector.removed_positions_.size)
        detector = TrainingOutlierFilter(max_fraction=0)
        detector.fit_filter(self.X, np.arange(200))
        self.assertTrue(detector.keep_mask_.all())
        for fraction in (-.01, .051, float("nan")):
            with self.assertRaises(ValueError):
                TrainingOutlierFilter(max_fraction=fraction).fit_filter(self.X, np.arange(200))


if __name__ == "__main__":
    unittest.main()
