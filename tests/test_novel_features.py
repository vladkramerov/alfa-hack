"""Synthetic leakage, grouping and train-only state contracts for HYP015-017."""
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

try:
    from experiments.novel_features import (
        CrossFeatureTransformer, FrequencyFeatureTransformer, GroupTargetEncoder,
    )
except ImportError:
    CrossFeatureTransformer = FrequencyFeatureTransformer = GroupTargetEncoder = None


class TargetEncodingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(GroupTargetEncoder, "GroupTargetEncoder is not implemented")

    def test_unique_categories_use_fold_prior_and_exclude_their_own_targets(self):
        X = pd.DataFrame({"category": list("abcdefghij"), "number": range(10)},
                         index=[5] * 10)
        y = np.arange(10, dtype=float) ** 2
        enc = GroupTargetEncoder()
        oof = enc.fit_transform(X, y)
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        for train, valid in GroupKFold(5).split(X, y, groups):
            np.testing.assert_allclose(oof.iloc[valid]["category__target_mean"], y[train].mean())
            changed = y.copy()
            changed[valid] += 100000
            other = clone(enc).fit_transform(X, changed)
            np.testing.assert_allclose(other.iloc[valid]["category__target_mean"],
                                       oof.iloc[valid]["category__target_mean"])
        self.assertFalse(np.allclose(oof["category__target_mean"],
                                     enc.transform(X)["category__target_mean"]))
        pd.testing.assert_series_equal(oof["number"], X["number"])
        self.assertEqual(list(oof), ["number", "category__target_mean"])
        self.assertTrue(oof.index.equals(X.index))
        self.assertTrue(enc.audit_["no_self_target_leakage"])
        self.assertTrue(enc.audit_["no_group_overlap"])
        self.assertEqual(sum(f["valid_rows"] for f in enc.audit_["folds"]), len(X))
        json.dumps(enc.audit_, allow_nan=False)

    def test_exact_duplicates_share_fold_even_with_conflicting_targets(self):
        X = pd.DataFrame({"category": np.repeat(list("abcde"), 2),
                          "number": np.repeat(range(5), 2)})
        y = np.arange(10, dtype=float)
        enc = GroupTargetEncoder()
        out = enc.fit_transform(X, y)
        for i in range(0, 10, 2):
            self.assertEqual(out.iloc[i]["category__target_mean"],
                             out.iloc[i + 1]["category__target_mean"])
            self.assertEqual(len(np.unique(enc.fold_assignment_[i:i + 2])), 1)
        self.assertEqual(enc.audit_["n_groups"], 5)
        for f in enc.audit_["folds"]:
            self.assertEqual(f["group_overlap"], 0)
            self.assertEqual(f["train_rows"], 8)
            self.assertEqual(f["valid_rows"], 2)

    def test_persistable_fold_indices_cover_each_row_once_and_keep_groups_disjoint(self):
        X = pd.DataFrame({"category": np.repeat(list("abcde"), 2),
                          "number": np.repeat(range(5), 2)}, index=[17] * 10)
        enc = GroupTargetEncoder()
        enc.fit_transform(X, np.arange(10, dtype=float))
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        coverage = np.zeros(len(X), dtype=int)
        self.assertEqual(len(enc.fold_indices_), 5)
        for fold, (train_positions, valid_positions) in enumerate(enc.fold_indices_):
            self.assertIsInstance(train_positions, np.ndarray)
            self.assertIsInstance(valid_positions, np.ndarray)
            np.testing.assert_array_equal(np.sort(np.concatenate([train_positions, valid_positions])),
                                          np.arange(len(X)))
            self.assertEqual(np.intersect1d(groups[train_positions], groups[valid_positions]).size, 0)
            np.testing.assert_array_equal(enc.fold_assignment_[valid_positions], fold)
            coverage[valid_positions] += 1
        np.testing.assert_array_equal(coverage, 1)
        enc.fit(X, np.arange(10, dtype=float))
        self.assertFalse(hasattr(enc, "fold_indices_"))

    def test_full_fit_smoothing_missing_prior_unknown_and_batch_invariance(self):
        X = pd.DataFrame({"cat": ["a", "a", None, "__MISSING__", "z"], "n": range(5)})
        y = np.array([10., 20., 35., 40., 50.])
        enc = GroupTargetEncoder().fit(X, y)
        probe = pd.DataFrame({"cat": ["a", None, "unknown", "__MISSING__"], "n": [8, 9, 10, 11]})
        out = enc.transform(probe)
        np.testing.assert_allclose(out["cat__target_mean"], [650 / 22, 31, 31, 660 / 21])
        pd.testing.assert_frame_equal(out.iloc[:1], enc.transform(probe.iloc[:1]))
        self.assertEqual(list(enc.get_feature_names_out()), list(out))
        self.assertFalse(enc.audit_["no_self_target_leakage"])

    def test_oof_known_categories_use_only_fold_fit_sums_counts_and_prior(self):
        X = pd.DataFrame({"cat": list("ababababab"), "number": range(10)})
        y = np.arange(10, dtype=float) ** 2
        enc = GroupTargetEncoder()
        out = enc.fit_transform(X, y)
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        for train, valid in GroupKFold(5).split(X, y, groups):
            for row in valid:
                matching = train[X.iloc[train]["cat"].to_numpy() == X.iloc[row]["cat"]]
                expected = (y[matching].sum() + 20 * y[train].mean()) / (len(matching) + 20)
                self.assertAlmostEqual(out.iloc[row]["cat__target_mean"], expected)

    def test_rejects_insufficient_groups_and_invalid_regression_targets(self):
        X = pd.DataFrame({"cat": list("abcde")})
        for y in (["a"] * 5, [0, 1, np.nan, 3, 4], [0, 1, np.inf, 3, 4], [[1]] * 5):
            with self.assertRaises(ValueError):
                GroupTargetEncoder().fit_transform(X, y)
        with self.assertRaises(ValueError):
            GroupTargetEncoder().fit_transform(X.iloc[:4], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            GroupTargetEncoder(smoothing=-1).fit(X, range(5))


class NetworkFeatureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(CrossFeatureTransformer, "CrossFeatureTransformer is not implemented")

    def test_cross_keeps_columns_and_disambiguates_missing_and_separators(self):
        X = pd.DataFrame({"proto": [None, "__MISSING__", "a/b", "a", "a"],
                          "service": ["x", "x", "c", "b/c", "b/c"],
                          "state": ["s", "s", "d", "d", "d"]})
        out = CrossFeatureTransformer().fit_transform(X)
        pd.testing.assert_frame_equal(out[X.columns], X)
        self.assertEqual(out["proto_service_state"].nunique(), 4)
        self.assertTrue(all(isinstance(v, str) for v in out["proto_service_state"]))
        pd.testing.assert_frame_equal(out.iloc[:1],
                                      CrossFeatureTransformer().transform(X.iloc[:1]))

    def test_frequencies_use_only_fit_rows_and_unknowns_are_zero(self):
        X = pd.DataFrame({"proto": ["tcp", "tcp", "udp", None],
                          "service": ["http", "http", "http", "dns"],
                          "state": ["ok", "ok", "ok", "bad"]})
        enc = FrequencyFeatureTransformer().fit(X, y=object())
        probe = pd.DataFrame({"proto": ["tcp", None, "new", "tcp"],
                              "service": ["http", "dns", "http", "dns"],
                              "state": ["ok", "bad", "ok", "ok"]})
        out = enc.transform(probe)
        np.testing.assert_allclose(out["proto__frequency"], [.5, .25, 0, .5])
        np.testing.assert_allclose(out["service__frequency"], [.75, .25, .75, .25])
        np.testing.assert_allclose(out["state__frequency"], [.75, .25, .75, .75])
        np.testing.assert_allclose(out["proto_service_state__frequency"], [.5, .25, 0, 0])
        pd.testing.assert_frame_equal(out[probe.columns], probe)
        state = {k: dict(v) for k, v in enc.frequency_maps_.items()}
        pd.testing.assert_frame_equal(out.iloc[:1], enc.transform(probe.iloc[:1]))
        enc.transform(pd.concat([probe] * 30))
        self.assertEqual(enc.frequency_maps_, state)
        self.assertEqual(enc.n_fit_, 4)
        self.assertEqual(list(enc.get_feature_names_out()), list(out))
        pd.testing.assert_frame_equal(clone(enc).fit_transform(X), enc.transform(X))

    def test_missing_values_do_not_collide_with_literal_missing_tokens(self):
        X = pd.DataFrame({"proto": [None, None, "__MISSING__"],
                          "service": ["a"] * 3, "state": ["b"] * 3})
        enc = FrequencyFeatureTransformer().fit(X)
        out = enc.transform(X)
        np.testing.assert_allclose(out["proto__frequency"], [2 / 3, 2 / 3, 1 / 3])
        with self.assertRaises(ValueError):
            FrequencyFeatureTransformer().fit(X.iloc[:0])
        with self.assertRaises(ValueError):
            CrossFeatureTransformer().transform(X.drop(columns="state"))


if __name__ == "__main__":
    unittest.main()
