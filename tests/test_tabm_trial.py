"""Independent checks for the experimental TabM-mini implementation."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TabMMiniTests(unittest.TestCase):
    def test_prediction_archive_is_readable_without_pickle(self):
        from experiments.tabm_trial import prediction_arrays
        arrays = prediction_arrays({'y': np.array(['a','long label'],dtype=object),
                                    'positions': np.array([2,9]), 'prediction': np.array([.2,.7])})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as directory:
            path = Path(directory)/'predictions.npz'
            np.savez_compressed(path,**arrays)
            with np.load(path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['y'], ['a','long label'])
                np.testing.assert_array_equal(saved['positions'], [2,9])
                np.testing.assert_array_equal(saved['prediction'], [.2,.7])

    def test_gradients_match_finite_differences_for_every_parameter(self):
        from experimental_tabm import MiniEnsemble
        rng = np.random.default_rng(12)
        X = rng.normal(size=(5, 3))
        for output in (1, 3):
            net = MiniEnsemble(3, output, members=2, width=4, seed=7, dtype=np.float64)
            y = rng.normal(size=5) if output == 1 else np.array([0, 1, 2, 1, 0])
            _, gradients = net.loss_grad(X, y)
            for name, parameter in net.params.items():
                for index in (0, parameter.size // 2, parameter.size - 1):
                    old = parameter.flat[index]
                    parameter.flat[index] = old + 1e-6
                    plus = net.loss_grad(X, y)[0]
                    parameter.flat[index] = old - 1e-6
                    minus = net.loss_grad(X, y)[0]
                    parameter.flat[index] = old
                    self.assertAlmostEqual(gradients[name].flat[index], (plus-minus)/2e-6,
                                           delta=2e-7, msg=f'{output}:{name}:{index}')

    def test_regression_learns_and_is_pickle_and_batch_invariant(self):
        from experimental_tabm import TabMMini
        x = np.linspace(-2, 2, 100)
        X = pd.DataFrame({'x': x, 'category': np.where(x > 0, 'right', 'left')})
        y = 12 + 3*x
        model = TabMMini(epochs=55, width=16, members=3, batch_size=100).fit(X, y)
        p = model.predict(X)
        self.assertLess(np.mean((p-y)**2), np.var(y)*.15)
        np.testing.assert_allclose(model.predict(X.iloc[[4, 2]]), p[[4, 2]], atol=2e-5)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), p)

    def test_classifier_preserves_labels_and_averages_probabilities(self):
        from experimental_tabm import TabMMini
        x = np.linspace(-2, 2, 100)
        X = pd.DataFrame({'x': x})
        y = np.where(x > 0, 'zebra', 'aardvark')
        model = TabMMini(kind='classification', epochs=40, width=16, members=3,
                        batch_size=100).fit(X, y)
        p = model.predict_proba(X)
        np.testing.assert_array_equal(model.classes_, ['aardvark', 'zebra'])
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        self.assertGreater(np.mean(model.predict(X) == y), .94)
        with self.assertRaises(ValueError): model.predict_proba(X.drop(columns='x'))

    def test_preprocessing_is_fit_only_unknown_missing_and_reordered_columns(self):
        from experimental_tabm import TabMMini
        X = pd.DataFrame({'value': [1., 2., np.nan, 4., 8., 16.],
                          'empty': [np.nan]*6, 'cat': ['a', 'b', None, 'a', 'b', 'a']})
        model = TabMMini(epochs=2, width=8, members=2).fit(X, np.arange(6.))
        saved = pickle.dumps(model.preprocessor_)
        q = pd.DataFrame({'value': [1e20, np.inf], 'empty': [np.nan, 7.], 'cat': ['unseen', None]})
        p = model.predict(q)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_array_equal(model.predict(q[['cat','empty','value']]), p)
        self.assertEqual(pickle.dumps(model.preprocessor_), saved)
        np.testing.assert_array_equal(X.value.isna(), [False,False,True,False,False,False])

    def test_invalid_targets_and_resource_settings_fail_before_fit(self):
        from experimental_tabm import TabMMini
        X = pd.DataFrame({'x': [0., 1., 2., 3.]})
        for params in ({'epochs': 0}, {'members': 0}, {'width': 10000}, {'seconds': 0}):
            with self.assertRaises(ValueError): TabMMini(**params).fit(X, [0., 1., 2., 3.])
        with self.assertRaises(ValueError): TabMMini().fit(X, [0., np.nan, 2., 3.])
        with self.assertRaises(ValueError): TabMMini(kind='classification').fit(X, ['same']*4)

    def test_independent_ensemble_heads_are_not_collapsed(self):
        from experimental_tabm import MiniEnsemble
        net = MiniEnsemble(3, 2, members=4, width=8, seed=42)
        p = net.forward(np.ones((2, 3), dtype=np.float32))[0]
        self.assertEqual(p.shape, (2, 4, 2))
        self.assertGreater(float(np.std(p[0], axis=0).sum()), .01)


if __name__ == '__main__': unittest.main()
