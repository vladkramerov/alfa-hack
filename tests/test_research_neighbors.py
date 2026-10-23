"""Exact-distance contracts for the independent mixed-neighbor learner."""
import json
import unittest

import numpy as np
import pandas as pd

try:
    from research_neighbors import MixedNeighborsClassifier
except ImportError:
    MixedNeighborsClassifier = None


def brute_probabilities(train, labels, query, k):
    """Small, scalar oracle independent of blocked implementation."""
    numeric = list(train.select_dtypes(include="number").columns)
    categorical = [c for c in train if c not in numeric]
    classes = list(dict.fromkeys(labels))
    medians, scales = {}, {}
    for col in numeric:
        finite = [float(x) for x in train[col] if pd.notna(x) and np.isfinite(x)]
        medians[col] = float(np.median(finite)) if finite else 0.0
        scales[col] = float(np.percentile(finite, 75) - np.percentile(finite, 25)) if finite else 1.0
        if not scales[col]:
            scales[col] = 1.0
    out = []
    for _, q in query.iterrows():
        distances = []
        for row, (_, t) in enumerate(train.iterrows()):
            parts = []
            if numeric:
                terms = []
                for col in numeric:
                    tm = pd.isna(t[col]) or not np.isfinite(t[col])
                    qm = pd.isna(q[col]) or not np.isfinite(q[col])
                    tv = medians[col] if tm else float(t[col])
                    qv = medians[col] if qm else float(q[col])
                    terms.extend([min(abs(qv - tv) / scales[col], 3.0), float(tm != qm)])
                parts.append(sum(terms) / len(terms))
            if categorical:
                mismatches = []
                for col in categorical:
                    tv = None if pd.isna(t[col]) else str(t[col])
                    qv = None if pd.isna(q[col]) else str(q[col])
                    mismatches.append(float(tv != qv))
                parts.append(sum(mismatches) / len(mismatches))
            distances.append((sum(parts) / len(parts), row))
        neighbors = sorted(distances)[:min(k, len(train))]
        out.append([sum(labels[i] == cls for _, i in neighbors) / len(neighbors) for cls in classes])
    return np.asarray(out)


class MixedNeighborsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(MixedNeighborsClassifier, "MixedNeighborsClassifier is not implemented")

    def test_blocked_distance_matches_scalar_brute_force(self):
        rng = np.random.default_rng(15)
        train = pd.DataFrame({"x": rng.normal(size=79), "z": rng.normal(size=79),
                              "kind": rng.choice(["a", "b", None], size=79)})
        train.loc[[0, 3, 15], "x"] = [np.nan, np.inf, -np.inf]
        query = pd.DataFrame({"x": [np.nan, -100, 100, .25], "z": [0, .1, 90, -2],
                              "kind": [None, "unknown", "a", "b"]})
        labels = np.array(["001", "NA", "2", "001"] * 20)[:len(train)]
        model = MixedNeighborsClassifier(n_neighbors=7, query_block_size=2, train_block_size=13).fit(train, labels)
        expected = brute_probabilities(train, labels, query, 7)
        np.testing.assert_allclose(model.predict_proba(query), expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(model.predict(query), model.classes_[expected.argmax(axis=1)])

    def test_train_class_order_row_ties_and_opaque_labels(self):
        train = pd.DataFrame({"x": [0.] * 30})
        labels = ["NA", "001"] + ["2"] * 28
        model = MixedNeighborsClassifier(n_neighbors=2, train_block_size=7).fit(train, labels)
        np.testing.assert_array_equal(model.classes_, ["NA", "001", "2"])
        np.testing.assert_array_equal(model.predict(train.iloc[:1]), ["NA"])
        np.testing.assert_array_equal(model.predict_proba(train.iloc[:1]), [[.5, .5, 0]])

    def test_train_only_statistics_and_query_batch_invariance(self):
        train = pd.DataFrame({"x": [0., 1., 2., 4., np.nan], "kind": ["a", "b", "a", None, "c"]})
        query = pd.DataFrame({"x": [1.2, np.nan, 1e200, -1e200], "kind": ["a", "new", None, "b"]})
        model = MixedNeighborsClassifier(n_neighbors=1, query_block_size=2).fit(train, ["a", "b", "c", "d", "e"])
        before = json.dumps(model.audit_, sort_keys=True, allow_nan=False)
        batched = model.predict_proba(query)
        individual = np.vstack([model.predict_proba(query.iloc[[i]]) for i in range(len(query))])
        np.testing.assert_array_equal(batched, individual)
        # Matching category 'a' makes x=2 closer than the numeric-only x=1.
        np.testing.assert_array_equal(batched[:1], [[0, 0, 1, 0, 0]])
        self.assertEqual(before, json.dumps(model.audit_, sort_keys=True, allow_nan=False))
        self.assertEqual(model.audit_["numeric_medians"], {"x": 1.5})
        self.assertEqual(model.audit_["numeric_scales"], {"x": 1.75})
        json.dumps(model.configuration(), allow_nan=False)

    def test_all_missing_constants_and_infinities_remain_finite(self):
        train = pd.DataFrame({"empty": [np.nan] * 4, "constant": [7.] * 4,
                              "kind": [None, "<MISSING>", "NA", "001"]})
        query = pd.DataFrame({"empty": [np.inf, 1.], "constant": [7., -np.inf],
                              "kind": [None, "new"]})
        labels = ["missing", "literal", "NA", "001"]
        model = MixedNeighborsClassifier(n_neighbors=1).fit(train, labels)
        np.testing.assert_array_equal(model.predict(query.iloc[:1]), ["missing"])
        self.assertTrue(np.isfinite(model.predict_proba(query)).all())
        self.assertEqual(model.audit_["numeric_scales"], {"empty": 1.0, "constant": 1.0})
        json.dumps(model.audit_, allow_nan=False)

    def test_numeric_missing_flag_distinguishes_imputed_equal_values(self):
        train = pd.DataFrame({"x": [2., np.nan]})
        model = MixedNeighborsClassifier(n_neighbors=1).fit(train, ["observed", "missing"])
        np.testing.assert_array_equal(model.predict(pd.DataFrame({"x": [np.nan, 2.]})), ["missing", "observed"])

    def test_categorical_only_preserves_missing_and_lexical_values(self):
        train = pd.DataFrame({"kind": ["001", "NA", None, "<MISSING>", "<UNKNOWN>"]})
        labels = ["a", "b", "c", "d", "e"]
        query = pd.DataFrame({"kind": [None, "NA", "001", "<UNKNOWN>", "unseen"]})
        model = MixedNeighborsClassifier(n_neighbors=1).fit(train, labels)
        np.testing.assert_array_equal(model.predict(query), ["c", "b", "a", "e", "a"])

    def test_default_25_neighbors_equal_votes_and_short_training(self):
        train = pd.DataFrame({"x": np.arange(30, dtype=float)})
        model = MixedNeighborsClassifier().fit(train, ["a"] * 13 + ["b"] * 17)
        np.testing.assert_array_equal(model.predict_proba(train.iloc[:1]), [[13 / 25, 12 / 25]])
        short = MixedNeighborsClassifier().fit(train.iloc[:2], ["a", "b"])
        np.testing.assert_array_equal(short.predict_proba(train.iloc[:1]), [[.5, .5]])

    def test_column_reordering_is_safe_and_invalid_shapes_are_rejected(self):
        train = pd.DataFrame({"x": [0., 1.], "kind": ["a", "b"]})
        model = MixedNeighborsClassifier(n_neighbors=1).fit(train, ["a", "b"])
        np.testing.assert_array_equal(model.predict(train[["kind", "x"]]), ["a", "b"])
        self.assertEqual(model.predict_proba(train.iloc[:0]).shape, (0, 2))
        with self.assertRaises(ValueError):
            model.predict(train[["x"]])
        with self.assertRaises(ValueError):
            MixedNeighborsClassifier().fit(train, ["a"])
        with self.assertRaises(ValueError):
            MixedNeighborsClassifier().fit(train.iloc[:0], [])
        with self.assertRaises(ValueError):
            MixedNeighborsClassifier(n_neighbors=0).fit(train, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
