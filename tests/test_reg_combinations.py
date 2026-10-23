"""Behavioral contracts for SPEC-008 regression compositions (synthetic data only)."""
import importlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def buildings(rows=1000):
    x = np.arange(rows)
    frame = pd.DataFrame({'building_class': np.where(x < rows // 2, 'A', 'B'),
                          'Year_Factor': x % 4 + 1, 'State_Factor': 'S', 'facility_type': 'office',
                          'floor_area': x + 1., 'energy_star_rating': x % 100,
                          'direction_max_wind_speed': x % 360}, index=x * 2 + 7)
    for month in 'january february march april may june july august september october november december'.split():
        frame[month + '_avg_temp_c'] = x + 1.
    return frame


class SavedLegacyPredictor:
    def fit(self, X, y):
        raise AssertionError('An existing baseline must never be fitted')

    def predict(self, X):
        return X['floor_area'].to_numpy(dtype=float)


class RegressionCombinationsTest(unittest.TestCase):
    def module(self):
        return importlib.import_module('automl_reg_combo')

    def test_feature_composition_preserves_rows_and_training_state(self):
        module = self.module()
        X = buildings(20)
        X.loc[X.index[0], 'january_avg_temp_c'] = np.nan
        feature = module.FeatureComposition([29, 30, 31]).fit(X)
        valid = X.iloc[[0, 3]].copy()
        valid['floor_area'] = 100000
        result = feature.transform(valid)
        self.assertEqual(result.index.tolist(), [7, 13])
        self.assertIn('monthly_temp__cos2', result)
        self.assertIn('floor_area__group_relative', result)
        self.assertIn('direction_max_wind_speed__sin', result)
        self.assertEqual(feature.transforms_[1].medians_['january_avg_temp_c'], 11.)
        self.assertEqual(result['floor_area'].tolist(), [100000, 100000])

    def test_group_experts_fit_feature_state_on_their_own_rows(self):
        module = self.module()
        X = buildings()
        model = module.build_reg_combo({'strategy': 'reg_feature_experts', 'feature_numbers': [30],
                                        'parameters': {'n_estimators': 3}}, threads=1).fit(X, np.arange(1000) % 70 + 1)
        self.assertEqual(model.global_.feature_.transforms_[0].medians_['january_avg_temp_c'], 500.5)
        self.assertEqual(model.experts_['A'].feature_.transforms_[0].medians_['january_avg_temp_c'], 250.5)
        self.assertEqual(model.audit_['fit_retained_rows'], 1000)
        self.assertEqual(model.removed_positions_.tolist(), [])
        self.assertEqual(model.predict(X.iloc[[40, 900]]).shape, (2,))

    def test_convex_weights_use_equal_fold_weight_and_enforce_simplex(self):
        module = self.module()
        # Nine rows favor column0, one equally weighted fold favors column1;
        # equal-fold MSE has its unique optimum at [.5,.5].
        matrix = np.tile([0., 10., 10., 10.], (10, 1))
        target = np.r_[np.zeros(9), 10.]
        weights, audit = module.choose_convex_weights(target, matrix, np.r_[np.zeros(9), 1])
        self.assertAlmostEqual(weights[0], .5, places=5)
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(np.sum(weights)), 1.)
        self.assertEqual(audit['selection_source'], 'training_temporal_oof')

    def test_convex_optimum_is_invariant_to_target_units(self):
        module = self.module()
        matrix = np.tile([0., 1000000., 1000000., 1000000.], (10, 1))
        weights, _ = module.choose_convex_weights(np.r_[np.zeros(9), 1000000.], matrix, np.r_[np.zeros(9), 1])
        self.assertAlmostEqual(weights[0], .5, places=5)

    def test_tail_prediction_is_probability_weighted_original_scale(self):
        module = self.module()
        X = buildings(100)
        y = np.arange(100, dtype=float) + 1
        model = module.build_reg_combo({'strategy': 'target_tail_mixture', 'tail_quantile': .8,
                                        'parameters': {'n_estimators': 3},
                                        'gate_parameters': {'n_estimators': 3}}, threads=1).fit(X, y)
        self.assertAlmostEqual(model.threshold_, 80.2)
        valid = X.iloc[:5]
        gate = model.tail_probability(valid)
        expected = gate * model.high_.predict(valid) + (1 - gate) * model.low_.predict(valid)
        np.testing.assert_allclose(model.predict(valid), expected)
        self.assertEqual(model.audit_['low_rows'], 80)
        self.assertEqual(model.audit_['high_rows'], 20)
        self.assertEqual(model.audit_['fit_retained_rows'], 100)

    def test_frozen_convex_weights_are_validated_before_any_fit(self):
        with self.assertRaisesRegex(ValueError, 'weights'):
            self.module().build_reg_combo({'strategy': 'reg_convex_blend', 'blend_weights': [-1, 1, 1, 0]}).fit(buildings(8), np.arange(8))

    def test_factory_rejects_oversized_estimators_and_gpu_gate(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.build_reg_combo({'strategy': 'reg_feature_experts', 'parameters': {'n_estimators': 999999}})
        with self.assertRaises(ValueError):
            module.build_reg_combo({'strategy': 'target_tail_mixture', 'gate_parameters': {'device': 'cuda'}})

    def test_legacy_oof_is_read_only_and_missing_cache_never_fits(self):
        from tempfile import TemporaryDirectory
        module = importlib.import_module('experiments.reg_combinations')
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'legacy'):
                module.read_legacy_oof(buildings(12), np.arange(12), {'legacy_cache_dir': directory,
                                                                       'fit_positions': np.arange(12)})

    def test_convex_evaluation_reuses_saved_outer_and_never_fits_legacy(self):
        import joblib
        from tempfile import TemporaryDirectory
        from experiments.twenty_advanced import inner_fingerprint
        from experiments.reg_combinations import evaluate_reg
        module = self.module()
        X = buildings(20)
        y = X['floor_area'].to_numpy()
        positions = np.arange(20) + 100
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            directory = Path(directory)
            legacy = directory / 'legacy' / inner_fingerprint('regression', X, y, positions, {'policy': 'legacy'})
            legacy.mkdir(parents=True)
            for fold, (fit, valid) in enumerate(module.temporal_folds(X)):
                np.savez_compressed(legacy / f'fold-{fold}.npz', fit=fit, valid=valid,
                                    fit_positions=positions[fit], valid_positions=positions[valid], prediction=y[valid])
                joblib.dump(SavedLegacyPredictor(), legacy / f'fold-{fold}-model.joblib')
            context = {'legacy_cache_dir': directory / 'legacy', 'cache_dir': directory / 'new',
                       'fit_positions': positions, 'valid_positions': np.array([500, 501]),
                       'output': directory / 'output', 'outer_components': {'raw': [100., 100.],
                           'experts': [200., 200.], 'season': [300., 300.], 'relative': [400., 400.]}}
            result = evaluate_reg(44, X, y, X.iloc[:2], context)
            weights = np.asarray(result['learned_state']['blend_weights'])
            self.assertLess(float(weights[2:].sum()), 1e-5)
            np.testing.assert_allclose(result['prediction'], np.repeat(np.array([100., 200., 300., 400.]) @ weights, 2))
            self.assertEqual(result['audit']['outer_fit'], 'reused_four_saved_predictions')
            # A restart must consume all three learned component caches.
            again = evaluate_reg(44, X, y, X.iloc[:2], context)
            self.assertTrue(all(event['status'] == 'reused' for event in again['audit']['components']))
            np.testing.assert_allclose(again['prediction'], result['prediction'])


if __name__ == '__main__':
    unittest.main()
