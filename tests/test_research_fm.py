"""Synthetic contract checks for SPEC010 FM; never load competition data."""
import importlib
import json
import unittest

import numpy as np
import pandas as pd
from scipy import sparse


class FactorizationRegressorTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module("research_fm")
        except ModuleNotFoundError as error:
            if error.name == "research_fm":
                self.fail("SPEC010 FM implementation is missing")
            raise

    def test_sparse_prediction_and_all_analytic_gradients_match_pairwise_oracle(self):
        # Wrong interaction self-terms, intercept regularization, or mean divisor fail.
        X = np.array([[1., 0., -2.], [.5, 2., 0.], [0., -1., 1.]])
        y = np.array([.2, -1., .7])
        w = np.array([.1, -.2, .3])
        factors = np.array([[.2, -.3], [.1, .4], [-.5, .2]])
        bias, l2 = .15, .07

        def oracle(b, linear, latent):
            prediction = []
            for row in X:
                interaction = sum(
                    row[i] * row[j] * float(latent[i] @ latent[j])
                    for i in range(3) for j in range(i + 1, 3)
                )
                prediction.append(b + float(row @ linear) + interaction)
            return (.5 * np.mean((np.array(prediction) - y) ** 2)
                    + .5 * l2 * (np.sum(linear ** 2) + np.sum(latent ** 2)))

        loss, db, dw, dv = self.api().fm_loss_gradient(
            sparse.csr_matrix(X), y, bias, w, factors, l2=l2
        )
        self.assertAlmostEqual(loss, oracle(bias, w, factors), places=13)
        epsilon = 1e-6
        self.assertAlmostEqual(
            db, (oracle(bias + epsilon, w, factors)
                 - oracle(bias - epsilon, w, factors)) / (2 * epsilon), places=8
        )
        for parameter, analytic in [(w, dw), (factors, dv)]:
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = oracle(bias, w, factors)
                parameter[index] = original - epsilon
                minus = oracle(bias, w, factors)
                parameter[index] = original
                numerical[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-9)

    def test_default_training_learns_additive_and_pair_signal_in_original_units(self):
        # A constant or linear-only learner cannot meet this held-out error bound.
        rng = np.random.default_rng(817)
        values = rng.uniform(-1., 1., size=(9216, 2))
        X = pd.DataFrame(values, columns=["first", "second"])
        y = 1000. + 17. * (.5 * values[:, 0] + .25 * values[:, 1]
                           + 1.5 * values[:, 0] * values[:, 1])
        model = self.api().FactorizationRegressor().fit(X.iloc[:8192], y[:8192])
        prediction = model.predict(X.iloc[8192:])
        baseline_mse = np.mean((y[8192:] - y[:8192].mean()) ** 2)
        self.assertLess(np.mean((prediction - y[8192:]) ** 2), .25 * baseline_mse)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(model.audit_["epochs_completed"], 20)

    def test_train_only_statistics_missing_flags_and_lexical_categories(self):
        X = pd.DataFrame({
            "number": [1., 3., np.nan, 5.],
            "all_missing": pd.Series([np.nan] * 4, dtype=float),
            "token": ["001", "1", None, "__MISSING__"],
        })
        model = self.api().FactorizationRegressor(epochs=2, batch_size=2).fit(
            X, np.array([100., 103., 106., 109.])
        )
        frozen = json.dumps(model.audit_, sort_keys=True)
        future = pd.DataFrame({
            "number": [np.nan, 3000., np.inf],
            "all_missing": [7., np.nan, -np.inf],
            "token": [None, "not-seen", "001"],
        })
        transformed = model._transform(future)
        self.assertTrue(sparse.isspmatrix_csr(transformed))
        # Numeric order: standardized values, then the original missing flags.
        np.testing.assert_allclose(transformed[:, :4].toarray(), [
            [0., 7., 1., 0.],
            [(3000. - 3.) / np.sqrt(2.), 0., 0., 1.],
            [0., 0., 1., 1.],
        ])
        encoded_train = model._transform(X)[:, 4:].toarray()
        self.assertEqual(encoded_train.shape[1], 4)
        np.testing.assert_array_equal(encoded_train.sum(axis=1), np.ones(4))
        self.assertEqual(len({tuple(row) for row in encoded_train}), 4)
        self.assertEqual(transformed[1, 4:].nnz, 0)
        self.assertTrue(np.isfinite(model.predict(future)).all())
        self.assertEqual(json.dumps(model.audit_, sort_keys=True), frozen)
        self.assertEqual(model.audit_["target_mean"], 104.5)
        self.assertAlmostEqual(model.audit_["target_scale"], np.sqrt(11.25))

    def test_predictions_do_not_depend_on_batch_companions_or_call_order(self):
        X = pd.DataFrame({"value": np.arange(40.) / 7., "category": ["a", "b"] * 20})
        model = self.api().FactorizationRegressor(batch_size=7, epochs=3).fit(
            X, 30. + np.arange(40.)
        )
        future = pd.DataFrame({"value": [3., np.nan, 900., -2.],
                               "category": ["b", "new", None, "a"]})
        together = model.predict(future)
        alone = np.array([model.predict(future.iloc[[i]])[0] for i in range(4)])
        np.testing.assert_allclose(together, alone, rtol=0., atol=1e-12)
        np.testing.assert_allclose(model.predict(future.iloc[::-1])[::-1], together,
                                   rtol=0., atol=1e-12)
        np.testing.assert_array_equal(model.predict(future), together)
        self.assertEqual(model.predict(future.iloc[:0]).shape, (0,))

    def test_constant_target_and_zero_feature_frame_return_constant(self):
        for X in [pd.DataFrame({"x": [1., np.nan, 3.], "c": ["a", "b", None]}),
                  pd.DataFrame(index=range(3))]:
            with self.subTest(columns=list(X.columns)):
                model = self.api().FactorizationRegressor().fit(X, [12345.] * 3)
                np.testing.assert_array_equal(model.predict(X), [12345.] * 3)
                self.assertEqual(model.audit_["target_scale"], 1.)

    def test_high_cardinality_data_remains_sparse_and_seed_is_deterministic(self):
        X = pd.DataFrame({"lexical": [f"id-{i:05}" for i in range(2000)]})
        target = np.sin(np.arange(2000.))
        first = self.api().FactorizationRegressor(epochs=1).fit(X, target)
        second = self.api().FactorizationRegressor(epochs=1).fit(X, target)
        encoded = first._transform(X)
        self.assertTrue(sparse.isspmatrix_csr(encoded))
        self.assertEqual(encoded.nnz, 2000)
        self.assertEqual(encoded.shape, (2000, 2000))
        np.testing.assert_array_equal(first.predict(X.iloc[:17]), second.predict(X.iloc[:17]))
        json.dumps(first.configuration(), allow_nan=False)
        json.dumps(first.audit_, allow_nan=False)

    def test_invalid_inputs_are_rejected_before_learning(self):
        api = self.api()
        for kwargs in [{"rank": 0}, {"epochs": -1}, {"batch_size": 0},
                       {"learning_rate": 0}, {"l2": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.FactorizationRegressor(**kwargs).fit(pd.DataFrame({"x": [1.]}), [1.])
        for target in [[1., np.nan], [1.], [[1.], [2.]]]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                api.FactorizationRegressor().fit(pd.DataFrame({"x": [1., 2.]}), target)
        with self.assertRaises(ValueError):
            api.FactorizationRegressor().fit(pd.DataFrame({"x": []}), [])
        with self.assertRaises(ValueError):
            api.FactorizationRegressor().predict(pd.DataFrame({"x": [1.]}))
        model = api.FactorizationRegressor(epochs=1).fit(pd.DataFrame({"x": [1., 2.]}), [1., 2.])
        with self.assertRaises(ValueError):
            model.predict(pd.DataFrame({"wrong": [1.]}))

    def test_failed_refit_invalidates_prior_parameters(self):
        # A failed schema-changing fit must not leave stale fitted state usable.
        model = self.api().FactorizationRegressor(epochs=1).fit(
            pd.DataFrame({"old": [1., 2.]}), [1., 2.]
        )
        with self.assertRaises(ValueError):
            model.fit(pd.DataFrame({"new": [3., 4.]}), [1., np.nan])
        with self.assertRaises(ValueError):
            model.predict(pd.DataFrame({"old": [1.]}))

    def test_nullable_and_all_missing_categories_keep_training_schema(self):
        # Nullable extension arrays and absent-at-fit levels must remain finite.
        X = pd.DataFrame({"numeric": pd.Series([1., pd.NA, 5.], dtype="Float64"),
                          "category": pd.Series([pd.NA] * 3, dtype="string")})
        model = self.api().FactorizationRegressor(epochs=1).fit(X, [1., 2., 4.])
        future = pd.DataFrame({"category": ["new", None], "numeric": [3., np.nan]})
        encoded = model._transform(future)
        self.assertEqual(encoded.shape, (2, 3))
        np.testing.assert_array_equal(encoded[:, -1].toarray().ravel(), [0., 1.])
        self.assertTrue(np.isfinite(model.predict(future)).all())


if __name__ == "__main__":
    unittest.main()
