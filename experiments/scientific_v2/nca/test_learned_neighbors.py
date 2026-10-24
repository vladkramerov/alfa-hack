from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
"""Meaningful synthetic behavioral checks; zero public-data fits."""
import io
import time
import unittest
import joblib
import numpy as np
import pandas as pd
import torch
from learned_neighbors import (LearnedNeighborClassifier, FitPreprocessor,
                               soft_neighbors, check_deadline)


class LearnedNeighborTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.X = pd.DataFrame({"a": np.r_[np.linspace(-2, -1, 24), np.linspace(1, 2, 24)],
                              "b": ["left"] * 24 + ["right"] * 24})
        cls.y = np.array(["low"] * 24 + ["high"] * 24)

    def test_same_raw_group_is_excluded_even_when_not_self_row(self):
        query = torch.tensor([[1., 0.]], requires_grad=True)
        neighbors = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
        targets = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        p, valid = soft_neighbors(query, neighbors, targets, .1,
            query_groups=torch.tensor([90]), candidate_groups=torch.tensor([90, 90, 91]))
        np.testing.assert_allclose(p.detach().numpy(), [[0., 1.]])
        self.assertTrue(bool(valid.all()))
        p.sum().backward()
        self.assertTrue(bool(torch.isfinite(neighbors.grad).all()))

    def test_all_masked_rows_are_explicit_and_finite(self):
        p, valid = soft_neighbors(torch.ones(1, 2), torch.ones(2, 2), torch.eye(2), .1,
            query_groups=torch.tensor([1]), candidate_groups=torch.tensor([1, 1]))
        self.assertFalse(bool(valid.any()))
        self.assertTrue(bool(torch.isfinite(p).all()))
        self.assertEqual(float(p.sum()), 0.)

    def test_inner_validation_targets_do_not_change_metric_training(self):
        validation = pd.DataFrame({"a": [-3., 3.], "b": ["left", "unseen"]})
        a = LearnedNeighborClassifier().fit(self.X, self.y, epochs=1,
            validation=(validation, np.array(["low", "high"])))
        b = LearnedNeighborClassifier().fit(self.X, self.y, epochs=1,
            validation=(validation, np.array(["high", "low"])))
        for name, value in a.model_.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), b.model_.state_dict()[name].numpy())
        np.testing.assert_allclose(a.predict_proba(validation), b.predict_proba(validation), atol=0)

    def test_serialization_probabilities_and_block_invariance(self):
        model = LearnedNeighborClassifier().fit(self.X, self.y, epochs=2)
        prediction = model.predict_proba(self.X, query_chunk=7, memory_chunk=9)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(prediction.sum(1), 1, atol=1e-7)
        self.assertGreater(np.mean(model.classes_[prediction.argmax(1)] == self.y), .95)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        np.testing.assert_allclose(prediction,
            restored.predict_proba(self.X, query_chunk=13, memory_chunk=17), atol=1e-6)

    def test_preprocessing_fit_only_and_unknown_category(self):
        X = pd.DataFrame({"n": [1., np.nan, 3.], "c": ["a", "a", "b"]})
        preparation = FitPreprocessor().fit(X)
        validation = pd.DataFrame({"n": [1e15, np.nan], "c": ["new", "b"]})
        result = preparation.transform(validation)
        self.assertEqual(preparation.medians_["n"], 2.)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(float(result[0, 1:].sum()), 0.)

    def test_expired_budget_stops_before_any_fit(self):
        with self.assertRaises(TimeoutError):
            LearnedNeighborClassifier().fit(self.X, self.y, deadline=time.monotonic() - 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
