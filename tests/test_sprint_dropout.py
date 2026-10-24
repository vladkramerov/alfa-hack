"""H104 DART dropout training with deterministic teacher-margin inference."""
import importlib
import importlib.util
import json
import pickle
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax
import xgboost as xgb

from sprint_classification import soft_macro_f1_loss_and_gradient


class DropoutTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_dropout'), 'H104 DART learner absent')
        self.module = importlib.import_module('sprint_dropout')

    def fixture(self):
        rng = np.random.default_rng(104)
        X = pd.DataFrame({'x': rng.normal(size=120), 'cat': np.tile(['tcp', 'udp', None], 40)})
        y = np.where(X.x < -.4, 'a', np.where(X.x > .4, 'z', 'middle'))
        return X, y, np.tile([.2, .5, .3], (120, 1))

    def test_frozen_scalar_dart_configuration(self):
        model = self.module.DropoutTeacherBoost()
        config = model.configuration()
        self.assertEqual(config['hypothesis'], 'HYP-104')
        self.assertEqual(config['iterations'], 300)
        for key, value in {'booster': 'dart', 'multi_strategy': 'one_output_per_tree',
                'rate_drop': .1, 'skip_drop': .5, 'sample_type': 'uniform',
                'normalize_type': 'tree', 'one_drop': 0, 'max_depth': 6,
                'eta': .05, 'lambda': 5., 'nthread': 2, 'subsample': 1.}.items():
            self.assertEqual(config['parameters'][key], value)

    def test_exact_custom_gradient_and_positive_surrogate(self):
        model = self.module.DropoutTeacherBoost(iterations=0)
        model.loss_history_ = []
        z = np.array([[1.1, -.2, .3], [-.7, 1.2, .1], [.2, -.8, 1.3]])
        labels = np.array([0, 2, 1])
        dm = xgb.DMatrix(np.ones((3, 1)), label=labels, nthread=2)
        gradient, hessian = model._objective(z, dm)
        numeric = np.zeros_like(z)
        for i, j in np.ndindex(z.shape):
            a, b = z.copy(), z.copy()
            a[i, j] += 1e-6
            b[i, j] -= 1e-6
            numeric[i, j] = (soft_macro_f1_loss_and_gradient(a, labels)[0]
                - soft_macro_f1_loss_and_gradient(b, labels)[0]) / 2e-6
        np.testing.assert_allclose(gradient/3, numeric, atol=2e-9, rtol=1e-6)
        p = softmax(z, axis=1)
        np.testing.assert_allclose(hessian, np.maximum(.5*p*(1-p), .001))

    def test_real_dart_scalar_trees_margin_once_and_inference_is_deterministic(self):
        X, y, teacher = self.fixture()
        before = X.copy(deep=True)
        model = self.module.DropoutTeacherBoost(iterations=12).fit(X, y, teacher)
        saved = json.loads(model.model_.save_raw(raw_format='json'))
        booster = saved['learner']['gradient_booster']
        self.assertEqual(booster['name'], 'dart')
        trees = booster['gbtree']['model']['trees']
        self.assertEqual(len(trees), 36)
        self.assertTrue(all(int(t['tree_param']['size_leaf_vector']) == 1 for t in trees))
        self.assertEqual(model.audit_['tree_leaf_vector_size'], 1)
        # At least one old tree was renormalized by actual training-time dropout.
        self.assertTrue(any(float(weight) != 1. for weight in booster['weight_drop']))
        q = X.iloc[:6].copy(); q.loc[q.index[0], 'cat'] = 'unknown'
        supplied = np.tile([.65, .1, .25], (len(q), 1))
        dm = xgb.DMatrix(model.encoder_.transform(q), base_margin=np.log(supplied), nthread=2)
        raw = model.model_.predict(dm, output_margin=True, strict_shape=True, training=False)
        actual = model.predict_proba(q, supplied)
        np.testing.assert_allclose(actual, softmax(raw.astype(float), axis=1), atol=1e-12)
        np.testing.assert_array_equal(model.predict_proba(q, supplied), actual)
        np.testing.assert_array_equal(model.predict_proba(q.iloc[:1], supplied[:1]), actual[:1])
        np.testing.assert_array_equal(model.predict_proba(q.iloc[::-1], supplied[::-1]), actual[::-1])
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict_proba(q, supplied), actual)
        loaded = xgb.Booster(params={'nthread': 2})
        loaded.load_model(model.model_.save_raw(raw_format='json'))
        model.model_ = loaded
        np.testing.assert_array_equal(model.predict_proba(q, supplied), actual)
        pd.testing.assert_frame_equal(X, before)

    def test_zero_round_teacher_only_and_class_schema_guards(self):
        X, y, teacher = self.fixture()
        model = self.module.DropoutTeacherBoost(iterations=0).fit(X, y, teacher)
        np.testing.assert_array_equal(model.predict_proba(X, teacher), teacher)
        self.assertEqual(model.audit_['learner_fits'], 0)
        self.assertEqual(model.audit_['configuration']['hypothesis'], 'HYP-104')
        with self.assertRaises(ValueError): model.predict_proba(X, None)
        with self.assertRaises(ValueError): model.predict_proba(X[['cat', 'x']], teacher)
        with self.assertRaises(ValueError): model.predict_proba(X, pd.DataFrame(teacher, columns=['z', 'middle', 'a']))


if __name__ == '__main__': unittest.main()
