"""Portable measured-winner behavior, using small synthetic fits only."""
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class AdvancedDeliveryTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('automl_advanced')

    def test_frozen_multiplier_alignment_preserves_probability_mass(self):
        api = self.api()
        result = api.calibrate_probabilities(np.array([[.4, .6], [.8, .2]]), ['a', 'z'], {'z': 2., 'a': 1.})
        np.testing.assert_allclose(result, [[.25, .75], [2/3, 1/3]])
        np.testing.assert_allclose(result.sum(axis=1), 1.)
        with self.assertRaises(ValueError):
            api.calibrate_probabilities(np.array([[.4, .6]]), ['a', 'z'], {'z': np.nan})

    def test_frozen_classifier_uses_fixed_coefficients_without_inner_fits(self):
        api = self.api()
        X = pd.DataFrame({'x': np.arange(18.), 'proto': ['tcp', 'udp', 'tcp'] * 6})
        y = np.array(['a', 'z', 'a'] * 6)
        candidate = {'strategy': 'macro_f1_calibrated', 'parameters': {'n_estimators': 3, 'max_depth': 2},
                     'class_multipliers': {'a': 1., 'z': 2.}}
        with patch.object(api, 'calibration_folds', side_effect=AssertionError('frozen policy requested OOF')):
            model = api.build_advanced('classification', candidate, threads=1).fit(X, y)
        base = model.base_.predict_proba(X)
        expected = base * [1., 2.]
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(model.predict_proba(X), expected)
        np.testing.assert_array_equal(model.predict(X), model.classes_[np.argmax(expected, axis=1)])
        assert model.audit_['calibration_multipliers'] == {'a': 1., 'z': 2.}
        assert model.keep_mask_.all() and not len(model.removed_positions_)

    def test_group_experts_shrink_and_unseen_groups_use_global(self):
        X = pd.DataFrame({'x': np.arange(1000.) % 17, 'building_class': ['a'] * 500 + ['b'] * 500})
        y = np.r_[np.repeat(2., 500), np.repeat(8., 500)]
        model = self.api().build_advanced('regression', {'strategy': 'group_experts', 'parameters': {'n_estimators': 3}}, threads=1).fit(X, y)
        probe = pd.DataFrame({'x': [4., 4., 4.], 'building_class': ['a', 'b', 'unseen']})
        global_prediction = model.global_.predict(probe)
        expected = global_prediction.copy()
        expected[0] = .5 * model.experts_['a'].predict(probe.iloc[:1])[0] + .5 * expected[0]
        expected[1] = .5 * model.experts_['b'].predict(probe.iloc[1:2])[0] + .5 * expected[1]
        np.testing.assert_allclose(model.predict(probe), expected)
        assert len(model.experts_) == 2
        assert model.audit_['fit_retained_rows'] == 1000

    def test_absent_group_column_fits_global_fallback(self):
        X = pd.DataFrame({'x': np.arange(12.)})
        model = self.api().build_advanced('regression', {'strategy': 'group_experts', 'parameters': {'n_estimators': 3}}, threads=1).fit(X, np.arange(12.))
        assert model.experts_ == {}
        np.testing.assert_allclose(model.predict(X), model.global_.predict(X))

    def test_unknown_dataset_inner_folds_keep_raw_duplicate_groups_together(self):
        X = pd.DataFrame({'x': np.repeat(np.arange(30), 2)})
        y = np.repeat(['a', 'b', 'c'] * 10, 2)
        folds, reason = self.api().calibration_folds(X, y)
        assert reason is None and len(folds) == 3
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        for fit, valid in folds:
            assert not set(groups[fit]) & set(groups[valid])

    def test_single_class_skips_calibration_and_produces_unit_probability(self):
        X = pd.DataFrame({'x': [1., 2., 3.]})
        model = self.api().build_advanced('classification', {'strategy': 'macro_f1_calibrated', 'parameters': {'n_estimators': 3}}, threads=1).fit(X, ['only'] * 3)
        np.testing.assert_allclose(model.predict_proba(X), np.ones((3, 1)))
        np.testing.assert_array_equal(model.predict(X), ['only'] * 3)
        assert model.audit_['calibration_multipliers'] == {'only': 1.}

    def test_multiplier_search_matches_the_measured_coordinate_rule(self):
        p = np.array([[.59, .41], [.56, .44], [.9, .1], [.2, .8]])
        mapping, score = self.api()._choose_multipliers(np.array(['b', 'b', 'a', 'b']), p, np.array(['a', 'b']))
        assert mapping == {'a': 1., 'b': 1.5}
        assert score == 1.

    def test_unknown_dataset_fits_grouped_oof_and_exports_frozen_mapping(self):
        X = pd.DataFrame({'x': np.repeat(np.arange(30.), 2), 'proto': ['tcp', 'udp'] * 30})
        y = np.repeat(['a', 'b', 'c'] * 10, 2)
        model = self.api().build_advanced('classification', {'strategy': 'macro_f1_calibrated',
                                                            'parameters': {'n_estimators': 3, 'max_depth': 2}}, threads=1).fit(X, y)
        assert model.audit_['calibration_source'] == 'training_grouped_oof'
        assert len(model.audit_['folds']) == 3
        assert model.calibration_multipliers_['a'] == 1.
        assert set(model.calibration_multipliers_.values()) <= {.75, 1., 1.25, 1.5, 2.}
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.)


if __name__ == '__main__':
    unittest.main()
