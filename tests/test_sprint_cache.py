"""Cache boundary checks: corrupt positions must fail before teacher use."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.sprint_cache import _arrays, _validate_source, assemble_oof, validate_probabilities


class SprintCacheTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([40, 7, 91, 13])
        self.classes = np.array(['a', 'b'])
        self.expected = [(np.array([1, 3]), np.array([0, 2])),
                         (np.array([0, 2]), np.array([1, 3]))]
        self.folds = [dict(fit=f, valid=v, fit_positions=self.positions[f],
                           valid_positions=self.positions[v], classes=self.classes,
                           probabilities=p)
                      for (f, v), p in zip(self.expected,
                          [np.array([[.9, .1], [.8, .2]]),
                           np.array([[.3, .7], [.4, .6]])])]

    def test_places_local_oof_rows_in_original_training_order(self):
        result = assemble_oof(self.folds, self.positions, self.classes, self.expected)
        np.testing.assert_allclose(result['probabilities'],
                                   [[.9, .1], [.3, .7], [.8, .2], [.4, .6]])
        np.testing.assert_array_equal(result['fold_id'], [0, 1, 0, 1])

    def test_rejects_global_index_drift(self):
        self.folds[0]['valid_positions'] = np.array([40, 92])
        with self.assertRaises(ValueError):
            assemble_oof(self.folds, self.positions, self.classes, self.expected)

    def test_rejects_fold_different_from_expected_training_split(self):
        self.folds[0]['fit'] = np.array([0, 3])
        self.folds[0]['fit_positions'] = np.array([40, 13])
        with self.assertRaises(ValueError):
            assemble_oof(self.folds, self.positions, self.classes, self.expected)

    def test_rejects_class_order_drift(self):
        self.folds[0]['classes'] = np.array(['b', 'a'])
        with self.assertRaises(ValueError):
            assemble_oof(self.folds, self.positions, self.classes, self.expected)

    def test_rejects_missing_or_duplicate_oof_coverage(self):
        for folds in [self.folds[:1], self.folds + [copy.deepcopy(self.folds[0])]]:
            with self.subTest(count=len(folds)), self.assertRaises(ValueError):
                assemble_oof(folds, self.positions, self.classes)

    def test_rejects_repeated_original_positions(self):
        with self.assertRaises(ValueError):
            assemble_oof(self.folds, np.array([40, 7, 91, 40]), self.classes)

    def test_rejects_noninteger_local_indices(self):
        self.folds[0]['valid'] = np.array([0., 2.])
        with self.assertRaises(ValueError):
            assemble_oof(self.folds, self.positions, self.classes)

    def test_rejects_leaking_fit_valid_overlap(self):
        self.folds[0]['fit'] = np.array([0, 1, 3])
        self.folds[0]['fit_positions'] = self.positions[[0, 1, 3]]
        with self.assertRaises(ValueError):
            assemble_oof(self.folds, self.positions, self.classes)

    def test_rejects_out_of_range_valid_indices(self):
        self.folds[0]['valid'] = np.array([0, 4])
        with self.assertRaises(ValueError):
            assemble_oof(self.folds, self.positions, self.classes)

    def test_rejects_malformed_probability_arrays(self):
        for p in [np.array([[.4, .4]]), np.array([[np.nan, .1]]),
                  np.array([[1.1, -.1]]), np.ones((1, 3)), np.array([.5, .5])]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                validate_probabilities(p, self.classes, 1)

    def test_canonicalizes_float32_row_sum_roundoff_for_frozen_meta_state(self):
        p = validate_probabilities([[.4999999, .5]], self.classes, 1)
        self.assertAlmostEqual(float(p.sum()), 1., places=15)

    def test_reads_numeric_fields_without_unpickling_legacy_object_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            path = Path(directory) / 'saved.npz'
            np.savez(path, y=np.array(['a'], dtype=object), probabilities=[[.5, .5]])
            saved = _arrays(path, keys=['probabilities'])
            np.testing.assert_allclose(saved['probabilities'], [[.5, .5]])

    def test_rejects_dataset_hash_drift(self):
        from types import SimpleNamespace
        with self.assertRaises(ValueError):
            _validate_source(SimpleNamespace(hashes={'train.csv': 'changed'}),
                             {'hashes': {'train.csv': 'original'}})


if __name__ == '__main__':
    unittest.main()
