"""SPEC-009 regression contracts, using synthetic observations only."""
import importlib
import unittest
import numpy as np
import pandas as pd


def data(rows=80):
    x = np.arange(rows)
    return pd.DataFrame({'Year_Factor': x % 4 + 1, 'number': x.astype(float),
                         'category': np.where(x % 2, 'A', 'B'), 'empty': np.nan}, index=x * 3 + 5)


class NextRegressionTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('automl_next_reg')

    def test_spline_keeps_rows_and_caps_training_categories(self):
        module = self.module()
        X = data(150)
        X['category'] = ['level' + str(i) for i in range(150)]
        model = module.build_next_reg({'strategy': 'spline_gam'}).fit(X, X['number'].to_numpy() * 2 + 3)
        valid = X.iloc[:3].copy()
        valid['category'] = 'unseen'
        values = model.transform(valid)
        self.assertEqual(values.shape[0], 3)
        self.assertTrue(np.isfinite(model.predict(valid)).all())
        self.assertLessEqual(len(model.ohe_.categories_[0]), 128)
        self.assertEqual(model.numeric_medians_['number'], 74.5)
        self.assertEqual(model.audit_['fit_retained_rows'], 150)
        self.assertEqual(model.removed_positions_.tolist(), [])

    def test_quantile_five_models_see_all_targets_and_mean_is_original_scale(self):
        module = self.module()
        X, y = data(), np.arange(80, dtype=float) * 3 + 10
        model = module.build_next_reg({'strategy': 'quantile_mean', 'parameters': {'n_estimators': 3}}, threads=1).fit(X, y)
        self.assertEqual(len(model.models_), 5)
        self.assertEqual(model.audit_['quantiles'], [.1, .3, .5, .7, .9])
        self.assertTrue(all(record['fit_rows'] == 80 for record in model.audit_['components']))
        values = model.encoder_.transform(X.iloc[:4])
        want = np.column_stack([branch.predict(values) for branch in model.models_]).mean(axis=1)
        np.testing.assert_allclose(model.predict(X.iloc[:4]), want)

    def test_masking_only_second_copy_preserves_protected_year_and_labels(self):
        module = self.module()
        X, y = data(200), np.arange(200, dtype=float)
        augmented, target, mask = module.mask_training_copy(X, y)
        pd.testing.assert_frame_equal(augmented.iloc[:200].reset_index(drop=True), X.reset_index(drop=True))
        self.assertEqual(augmented['Year_Factor'].tolist(), X['Year_Factor'].tolist() * 2)
        self.assertEqual(augmented['category'].tolist(), X['category'].tolist() * 2)
        np.testing.assert_array_equal(target, np.r_[y, y])
        self.assertTrue(mask.any())
        self.assertEqual(mask.shape, (200, 2))
        self.assertEqual(int(augmented.iloc[200:]['number'].isna().sum()), int(mask[:, 0].sum()))

    def test_domain_probabilities_are_oof_exclude_year_and_weights_normalized(self):
        module = self.module()
        X, y = data(80), np.arange(80, dtype=float) + 1
        model = module.build_next_reg({'strategy': 'covariate_shift', 'parameters': {'n_estimators': 3},
                                        'domain_parameters': {'n_estimators': 3}}, threads=1).fit(X, y)
        self.assertNotIn('Year_Factor', model.domain_columns_)
        self.assertTrue(np.isfinite(model.domain_probability_).all())
        self.assertAlmostEqual(model.sample_weight_.mean(), 1.)
        for event in model.audit_['domain_folds']:
            self.assertFalse(set(event['fit_positions']) & set(event['valid_positions']))
            self.assertTrue(event['year_excluded'])
        raw = np.clip(model.domain_probability_ / (1 - model.domain_probability_) * 3., .2, 5.)
        np.testing.assert_allclose(model.sample_weight_, raw / raw.mean())
        self.assertEqual(model.predict(X.iloc[:3]).shape, (3,))

    def test_residual_uses_temporal_oof_coverage_and_original_scale_targets(self):
        module = self.module()
        X = data(80)
        y = X['number'].to_numpy() + 1
        candidate = {'strategy': 'h43_residual', 'base_parameters': {'n_estimators': 3},
                     'parameters': {'n_estimators': 3}}
        model = module.build_next_reg(candidate, threads=1).fit(X, y)
        covered = X['Year_Factor'].isin([3, 4]).to_numpy()
        np.testing.assert_array_equal(model.residual_positions_, np.flatnonzero(covered))
        np.testing.assert_allclose(model.residual_target_, y[covered] - model.oof_prediction_[covered])
        for event in model.audit_['base_oof_folds']:
            self.assertLess(X.iloc[event['fit_positions']]['Year_Factor'].max(), X.iloc[event['valid_positions']]['Year_Factor'].min())
        want = model.base_.predict(X.iloc[:3]) + model.residual_.predict(X.iloc[:3])
        np.testing.assert_allclose(model.predict(X.iloc[:3]), want)

    def test_offline_residual_requires_saved_h43_before_any_new_fit(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        module = importlib.import_module('experiments.next_regression')
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as folder:
            context = {'output': folder, 'cache_dir': folder, 'h43_model_path': str(Path(folder) / 'absent.joblib'),
                       'baseline_prediction': np.zeros(2), 'fit_positions': np.arange(80), 'valid_positions': np.arange(2) + 80}
            with self.assertRaisesRegex(FileNotFoundError, 'H43'):
                module.evaluate_reg(56, data(80), np.arange(80), data(2), context)


if __name__ == '__main__':
    unittest.main()
