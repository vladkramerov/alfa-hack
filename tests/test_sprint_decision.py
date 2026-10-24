"""Synthetic checks only: affine decisions have no competition-data access."""
import importlib
import json
import pickle
import unittest

import numpy as np
import pandas as pd


class MacroDecisionTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('sprint_decision')
        self.X = pd.DataFrame({'unused': np.arange(10)})
        self.y = np.array(['a'] * 8 + ['b'] * 2)
        self.p = np.tile([.55, .45], (10, 1))

    def fit(self, p=None):
        return self.m.MacroF1Decision().fit(self.X, self.y, self.p if p is None else p)

    def test_training_path_improves_expected_objective_with_offsets(self):
        model = self.fit()
        self.assertGreater(model.audit_['selected_expected_macro_f1'],
                           model.audit_['initial_expected_macro_f1'])
        self.assertTrue(np.any(model.offset_ != 0))
        self.assertEqual(len(model.audit_['expected_objective_history']), 26)
        self.assertEqual(model.configuration()['iterations'], 25)
        self.assertEqual(model.configuration()['damping'], .5)

    def test_expected_objective_gradient_by_finite_difference(self):
        priors, q, t = np.array([.8, .2]), np.array([.7, .3]), np.array([.5, .1])
        value, dt, dq = self.m.expected_macro_f1_gradient(priors, q, t)
        self.assertTrue(np.isfinite(value))
        for vector, gradient in [(t, dt), (q, dq)]:
            for k in range(2):
                plus, minus = vector.copy(), vector.copy()
                plus[k] += 1e-6
                minus[k] -= 1e-6
                args_plus = (priors, q, plus) if vector is t else (priors, plus, t)
                args_minus = (priors, q, minus) if vector is t else (priors, minus, t)
                numerical = (self.m.expected_macro_f1_gradient(*args_plus)[0] -
                             self.m.expected_macro_f1_gradient(*args_minus)[0]) / 2e-6
                self.assertAlmostEqual(gradient[k], numerical, places=7)

    def test_predict_is_batch_invariant_and_train_only(self):
        model = self.fit()
        prior, gain, offset = model.class_prior_.copy(), model.gain_.copy(), model.offset_.copy()
        full = model.predict_proba(self.X, self.p)
        singles = np.vstack([model.predict_proba(self.X.iloc[[i]], self.p[[i]]) for i in range(10)])
        np.testing.assert_array_equal(full, singles)
        np.testing.assert_allclose(prior, [.8, .2])
        np.testing.assert_array_equal(model.class_prior_, prior)
        np.testing.assert_array_equal(model.gain_, gain)
        np.testing.assert_array_equal(model.offset_, offset)
        np.testing.assert_array_equal(model.predict(self.X, self.p), model.classes_[full.argmax(axis=1)])

    def test_probability_dataframe_reorders_by_explicit_class_names(self):
        model = self.fit(pd.DataFrame(self.p, columns=['a', 'b']))
        query = pd.DataFrame(self.p[:, ::-1], columns=['b', 'a'])
        np.testing.assert_allclose(model.predict_proba(self.X, query), model.predict_proba(self.X, self.p))
        for bad in [pd.DataFrame(self.p, columns=['a', 'z']),
                    pd.DataFrame(self.p, columns=['a', 'a'])]:
            with self.assertRaises(ValueError):
                model.predict_proba(self.X, bad)

    def test_invalid_teacher_is_rejected_in_fit_and_query(self):
        model = self.fit()
        for bad in [None, np.ones((10, 3)), np.zeros((10, 2)),
                    np.full((10, 2), np.nan), -np.ones((10, 2))]:
            with self.assertRaises(ValueError):
                self.m.MacroF1Decision().fit(self.X, self.y, bad)
            with self.assertRaises(ValueError):
                model.predict_proba(self.X, bad)

    def test_finite_normalized_serializable_and_unfitted(self):
        with self.assertRaises(ValueError):
            self.m.MacroF1Decision().predict_proba(self.X, self.p)
        model = self.fit()
        p = model.predict_proba(self.X, self.p)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict_proba(self.X, self.p), p)
        json.dumps(model.audit_)
        json.dumps(model.configuration())
        self.assertFalse(model.audit_['output_is_calibrated_posterior'])

    def test_only_label_counts_affect_fit_not_target_order_or_features(self):
        first = self.fit()
        second = self.m.MacroF1Decision().fit(self.X * 1000, self.y[::-1], self.p)
        np.testing.assert_array_equal(first.gain_, second.gain_)
        np.testing.assert_array_equal(first.offset_, second.offset_)

    def test_empty_query_and_one_class(self):
        model = self.fit()
        self.assertEqual(model.predict_proba(self.X.iloc[:0], self.p[:0]).shape, (0, 2))
        single = self.m.MacroF1Decision().fit(self.X, ['only'] * 10, np.ones((10, 1)))
        np.testing.assert_array_equal(single.predict(self.X, np.ones((10, 1))), ['only'] * 10)


if __name__ == '__main__':
    unittest.main()
