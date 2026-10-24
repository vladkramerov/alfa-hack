"""Meaningful synthetic checks; never reads competition data."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd


class SprintClassificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_classification'), 'new classifiers missing')
        self.m = importlib.import_module('sprint_classification')

    def test_soft_f1_gradient_central_difference_including_absent_class(self):
        z = np.array([[.2, -.3, .7], [.4, .1, -.2], [-.6, .8, .1]])
        y = np.array([0, 1, 0])
        loss, grad = self.m.soft_macro_f1_loss_and_gradient(z, y)
        self.assertTrue(np.isfinite(loss))
        numerical = np.zeros_like(z)
        for index in np.ndindex(z.shape):
            plus, minus = z.copy(), z.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numerical[index] = (self.m.soft_macro_f1_loss_and_gradient(plus, y)[0] -
                                self.m.soft_macro_f1_loss_and_gradient(minus, y)[0]) / 2e-5
        np.testing.assert_allclose(grad, numerical, atol=2e-8, rtol=2e-5)
        np.testing.assert_allclose(grad.sum(axis=1), 0, atol=1e-12)

    def test_zero_correction_preserves_teacher_and_training_changes_loss(self):
        X = pd.DataFrame({'number': np.r_[-np.ones(15), np.ones(15)], 'proto': ['a'] * 30})
        y = np.array(['left'] * 15 + ['right'] * 15)
        teacher = np.tile([.6, .4], (30, 1))
        initial = self.m.TeacherResidualF1Classifier(epochs=0, hidden_size=8).fit(X, y, teacher)
        np.testing.assert_allclose(initial.predict_proba(X, teacher), teacher, atol=1e-12)
        trained = self.m.TeacherResidualF1Classifier(epochs=30, hidden_size=8, learning_rate=.03).fit(X, y, teacher)
        self.assertLess(trained.audit_['loss_history'][-1], trained.audit_['loss_history'][0])
        self.assertGreater(np.mean(trained.predict(X, teacher) == y), .95)
        with self.assertRaises(ValueError):
            trained.predict_proba(X)

    def test_encoding_fits_only_training_and_unknown_categories_remain_finite(self):
        X = pd.DataFrame({'positive': [0., 1., np.nan, 3.], 'signed': [-1., 0., 1., 2.],
                          'proto': ['a', 'a', 'b', None]})
        model = self.m.TeacherResidualF1Classifier(epochs=0, hidden_size=4).fit(X, ['a', 'b', 'a', 'b'])
        before = model.encoder_.transform(X).copy()
        query = pd.DataFrame({'positive': [1e20], 'signed': [np.nan], 'proto': ['new']})
        self.assertTrue(np.isfinite(model.predict_proba(query)).all())
        np.testing.assert_equal(before, model.encoder_.transform(X))
        with self.assertRaises(ValueError):
            model.predict_proba(query, np.array([[0., 0.]]))

    def test_ecoc_code_separates_every_pair_and_decoder_recovers_all_labels(self):
        code = self.m.hadamard_code(10)
        self.assertEqual(code.shape, (10, 15))
        for i in range(10):
            for j in range(i):
                self.assertEqual(np.sum(code[i] != code[j]), 8)
        probability = np.where(code > 0, .99, .01)
        np.testing.assert_equal(self.m.decode_ecoc(probability, code).argmax(axis=1), np.arange(10))

    def test_ecoc_real_small_fit_and_each_head_hook(self):
        X = pd.DataFrame({'a': np.repeat([-2., 0., 2.], 12), 'cat': np.repeat(['a', 'b', 'c'], 12)})
        y = np.repeat(['x', 'y', 'z'], 12)
        visited = []
        def hook(model, values, target, config, **kwargs):
            visited.append(config['head'])
            self.assertAlmostEqual(float(np.mean(kwargs['sample_weight'])), 1.)
            return model.fit(values, target, **kwargs)
        model = self.m.ECOCClassifier(n_estimators=20, min_child_samples=2, fit_hook=hook).fit(X, y)
        self.assertEqual(visited, [0, 1, 2])
        self.assertGreater(np.mean(model.predict(X) == y), .95)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))


if __name__ == '__main__':
    unittest.main()
