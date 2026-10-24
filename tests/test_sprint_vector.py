"""H101: real installed vector-leaf training and teacher-margin contracts."""
import importlib
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
import xgboost as xgb

from sprint_classification import soft_macro_f1_loss_and_gradient


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_vector'), 'H101 vector learner absent')
        self.module = importlib.import_module('sprint_vector')

    def fixture(self):
        rng = np.random.default_rng(101)
        X = pd.DataFrame({'x': rng.normal(size=90), 'category': np.tile(['a', 'b', None], 30)})
        y = np.where(X.x < -.4, 'alpha', np.where(X.x > .4, 'zeta', 'middle'))
        p = np.tile([.2, .5, .3], (len(X), 1))
        return X, y, p

    def test_adapter_exact_fullbatch_gradient_finite_difference_and_positive_surrogate(self):
        model = self.module.VectorLeafTeacherBoost(iterations=0)
        z = np.array([[1.2, -.4, .1], [-.2, .9, .3], [.1, -.7, 1.4]])
        y = np.array([0, 2, 1])
        dm = xgb.DMatrix(np.ones((3, 1)), label=y, nthread=2)
        model.loss_history_ = []
        grad, hess = model._objective(z, dm)
        numerical = np.zeros_like(z)
        for i, j in np.ndindex(z.shape):
            a, b = z.copy(), z.copy()
            a[i, j] += 1e-6
            b[i, j] -= 1e-6
            numerical[i, j] = (soft_macro_f1_loss_and_gradient(a, y)[0]
                - soft_macro_f1_loss_and_gradient(b, y)[0]) / 2e-6
        np.testing.assert_allclose(grad / 3, numerical, atol=2e-9, rtol=1e-6)
        probability = softmax(z, axis=1)
        np.testing.assert_allclose(hess, np.maximum(.5 * probability * (1-probability), .001))
        np.testing.assert_allclose(grad.sum(axis=1), 0., atol=1e-15)

    def test_zero_rounds_exact_teacher_and_strict_sorted_class_alignment(self):
        X, y, p = self.fixture()
        before = X.copy(deep=True)
        model = self.module.VectorLeafTeacherBoost(iterations=0).fit(X, y, p)
        np.testing.assert_allclose(model.predict_proba(X, p), p, atol=0, rtol=0)
        self.assertIsNone(model.model_)
        self.assertEqual(model.audit_['learner_fits'], 0)
        pd.testing.assert_frame_equal(X, before)
        wrong = pd.DataFrame(p, columns=['zeta', 'middle', 'alpha'])
        with self.assertRaises(ValueError): model.predict_proba(X, wrong)
        with self.assertRaises(ValueError): model.predict_proba(X, None)
        with self.assertRaises(ValueError): model.predict_proba(X, np.full_like(p, np.inf))
        with self.assertRaises(ValueError): model.predict_proba(X[['category', 'x']], p)

    def test_two_rounds_are_two_shared_vector_trees_and_query_margin_is_added_once(self):
        X, y, p = self.fixture()
        model = self.module.VectorLeafTeacherBoost(iterations=2).fit(X, y, p)
        saved = json.loads(model.model_.save_raw(raw_format='json'))
        trees = saved['learner']['gradient_booster']['model']['trees']
        self.assertEqual(len(trees), 2)
        self.assertTrue(all(int(t['tree_param']['size_leaf_vector']) == 3 for t in trees))
        q = X.iloc[:6].copy()
        q.loc[q.index[0], 'category'] = 'never_seen'
        supplied = np.tile([.65, .1, .25], (len(q), 1))
        dm = xgb.DMatrix(model.encoder_.transform(q), base_margin=np.log(supplied), nthread=2)
        raw = model.model_.predict(dm, output_margin=True, strict_shape=True)
        actual = model.predict_proba(q, supplied)
        np.testing.assert_allclose(actual, softmax(raw.astype(np.float64), axis=1), atol=1e-12)
        self.assertGreater(np.max(np.abs(actual - softmax(raw + np.log(supplied), axis=1))), .05)
        np.testing.assert_allclose(actual.sum(axis=1), 1., atol=1e-15)
        self.assertEqual(model.audit_['rounds_completed'], 2)

    def test_pickle_json_roundtrip_batch_and_query_permutation(self):
        X, y, p = self.fixture()
        model = self.module.VectorLeafTeacherBoost(iterations=2).fit(X, y, p)
        expected = model.predict_proba(X, p)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict_proba(X, p), expected)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent) as tmp:
            path = Path(tmp) / 'vector.json'
            model.model_.save_model(path)
            restored.model_ = xgb.Booster(params={'nthread': 2})
            restored.model_.load_model(path)
            np.testing.assert_array_equal(restored.predict_proba(X, p), expected)
        order = np.random.default_rng(6).permutation(len(X))
        np.testing.assert_array_equal(model.predict_proba(X.iloc[order], p[order]), expected[order])
        np.testing.assert_array_equal(model.predict_proba(X.iloc[:1], p[:1]), expected[:1])
        np.testing.assert_array_equal(model.predict(X, p), model.classes_[expected.argmax(axis=1)])

    def test_configuration_and_invalid_fit_inputs_fail_closed(self):
        model = self.module.VectorLeafTeacherBoost()
        config = model.configuration()
        self.assertEqual(config['hypothesis'], 'HYP-101')
        self.assertEqual(config['iterations'], 300)
        for key, value in {'multi_strategy': 'multi_output_tree', 'tree_method': 'hist',
                           'max_depth': 6, 'eta': .05, 'lambda': 5., 'nthread': 2,
                           'subsample': 1., 'colsample_bytree': 1.}.items():
            self.assertEqual(config['parameters'][key], value)
        X, y, p = self.fixture()
        for teacher in (None, p[:, :2], np.zeros_like(p)):
            with self.assertRaises(ValueError): model.fit(X, y, teacher)
        with self.assertRaises(ValueError): model.fit(X, ['a'] * len(X), p[:, :1])
        for kwargs in ({'iterations': -1}, {'threads': 5}, {'max_seconds': 0}):
            with self.assertRaises(ValueError): self.module.VectorLeafTeacherBoost(**kwargs)


if __name__ == '__main__':
    unittest.main()
