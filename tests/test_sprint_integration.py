"""Portable sprint wrappers with isolated, non-training base fixtures."""
import copy
import importlib
import importlib.util
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import softmax


class FixedRegressionBase:
    def __init__(self, candidate=None, threads=4):
        self.candidate, self.threads = candidate, threads

    def fit(self, X, y):
        self.audit_ = {'fit_original_rows': len(X), 'fit_retained_rows': len(X)}
        return self

    def predict(self, X):
        return np.full(len(X), 100.)


class FixedClassBase:
    def __init__(self, candidate, threads=4):
        self.candidate = candidate

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.audit_ = {'fit_retained_rows': len(X)}
        return self

    def predict_proba(self, X):
        return np.column_stack([X['p'].to_numpy(), 1 - X['p'].to_numpy()])


def energy_frame():
    return pd.DataFrame({'State_Factor': ['CA'] * 4, 'building_class': ['Commercial'] * 4,
        'facility_type': ['Office'] * 4, 'floor_area': [100.] * 4,
        'year_built': [1980.] * 4, 'ELEVATION': [3.] * 4,
        'energy_star_rating': [50.] * 4, 'Year_Factor': [1., 2., 3., 4.]})


def decision_candidate():
    return {'family': 'xgboost', 'parameters': {}, 'strategy': 'affine_macro_f1_stacking',
        'meta_state': {'classes': ['a', 'z'], 'component_order': ['raw', 'smote', 'frequency'],
                       'clip': 1e-6, 'coef': [[0.] * 6], 'intercept': [0.]},
        'decision_state': {'classes': ['a', 'z'], 'gain': [1., 2.], 'offset': [0., .2]}}


class SprintIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('automl_sprint'), 'portable sprint adapter missing')
        self.module = importlib.import_module('automl_sprint')

    def test_entity_wrapper_preserves_rows_and_actual_linking_frozen_config(self):
        X, y = energy_frame(), np.array([10., 20., 30., 40.])
        candidate = {'family': 'extra_trees', 'parameters': {}, 'strategy': 'entity_linked_history'}
        before = copy.deepcopy(candidate)
        with patch.object(self.module, 'TemporalProfileModel', FixedRegressionBase):
            model = self.module.SprintRegression(candidate, threads=2).fit(X.iloc[:3], y[:3])
        # Three donors: .95*3/3.25, newest donor30, fallback100.
        np.testing.assert_allclose(model.predict(X.iloc[3:]), [100 - .95 * 3 / 3.25 * 70])
        np.testing.assert_allclose(model.predict(X.iloc[:1]), [100.])
        self.assertTrue(model.keep_mask_.all())
        self.assertEqual(model.removed_positions_.tolist(), [])
        self.assertEqual(model.audit_['fit_retained_rows'], 3)
        self.assertEqual(model.audit_['correction']['alpha'], .95)
        self.assertEqual(candidate, before)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict(X.iloc[3:]), model.predict(X.iloc[3:]))

    def test_panel_wrapper_stages_insufficient_pairs_fallback_without_fit(self):
        X = energy_frame()
        with patch.object(self.module, 'TemporalProfileModel', FixedRegressionBase):
            model = self.module.SprintRegression({'strategy': 'panel_transition_history',
                'family': 'extra_trees', 'parameters': {}}, threads=1).fit(X.iloc[:3], [10., 20., 30.])
        np.testing.assert_allclose(model.predict(X.iloc[3:]), [100.])
        self.assertFalse(model.audit_['correction']['transition_fitted'])
        self.assertEqual(model.audit_['correction']['training_pairs'], 2)

    def test_offline_ledger_registers_base_and_correction_before_fit(self):
        X = energy_frame()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            root = Path(directory)
            (root / 'ledger.md').write_text('prior completed experiments', encoding='utf-8')
            env = {'COMBO_RUN_DIR': str(root / 'run'), 'COMBO_LEDGER_PATH': str(root / 'ledger.md')}
            with patch.dict(os.environ, env), patch.object(self.module, 'TemporalProfileModel', FixedRegressionBase):
                self.module.SprintRegression({'strategy': 'entity_linked_history'}).fit(X, [10., 20., 30., 40.])
            events = [json.loads(line) for line in (root / 'run/model-fit-attempts.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertEqual([e['status'] for e in events], ['started', 'started'])
            self.assertTrue(all(e['ledger_sha256'] for e in events))
            outcomes = [json.loads(line) for line in (root / 'run/model-fit-outcomes.jsonl').read_text().splitlines()]
            self.assertEqual([e['status'] for e in outcomes], ['completed', 'completed'])

    def test_rejects_unvalidated_regression_variants_and_invalid_target(self):
        for c in ({'strategy': 'wrong'}, {'strategy': 'entity_linked_history', 'parameters': {'n_estimators': 3}},
                  {'strategy': 'entity_linked_history', 'alpha': .5}):
            with self.assertRaises(ValueError):
                self.module.SprintRegression(c)
        with self.assertRaises(ValueError):
            self.module.SprintRegression({'strategy': 'entity_linked_history'}, threads=5)
        for y in ([1., 2.], [1., 2., np.nan, 4.], [1., -1., 3., 4.]):
            with self.assertRaises(ValueError):
                self.module.SprintRegression({'strategy': 'entity_linked_history'}).fit(energy_frame(), y)

    def test_frozen_affine_decision_reuses_state_without_corrector_fit(self):
        candidate = decision_candidate()
        original = copy.deepcopy(candidate)
        X = pd.DataFrame({'p': [.9, .4, .7, .2]})
        with patch.object(self.module, 'ClassCombination', FixedClassBase):
            model = self.module.SprintDecision(candidate, threads=1).fit(X, ['a', 'z', 'a', 'z'])
        expected = softmax([[.9, .4], [.4, 1.4], [.7, .8], [.2, 1.8]], axis=1)
        np.testing.assert_allclose(model.predict_proba(X), expected)
        self.assertEqual(model.predict(X).tolist(), ['a', 'z', 'z', 'z'])
        np.testing.assert_allclose(model.predict_proba(X.iloc[:1]), expected[:1])
        self.assertEqual(model.audit_['decision_fit_count'], 0)
        self.assertEqual(model.learned_state_['decision_state'], candidate['decision_state'])
        self.assertEqual(candidate, original)
        candidate['decision_state']['gain'][0] = 999.
        np.testing.assert_allclose(model.predict_proba(X), expected)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict_proba(X), expected)

    def test_refuses_absent_invalid_or_wrong_class_frozen_state(self):
        for field in ('decision_state', 'meta_state'):
            c = decision_candidate()
            del c[field]
            with self.assertRaises(ValueError):
                self.module.SprintDecision(c)
        for state in ({'classes': ['z', 'a'], 'gain': [1., 2.], 'offset': [0., .2]},
                      {'classes': ['a', 'z'], 'gain': [np.inf, 2.], 'offset': [0., .2]},
                      {'classes': ['a', 'z'], 'gain': [1.], 'offset': [0., .2]}):
            c = decision_candidate()
            c['decision_state'] = state
            with self.assertRaises(ValueError):
                self.module.SprintDecision(c)
        model = self.module.SprintDecision(decision_candidate())
        with self.assertRaises(ValueError):
            model.fit(pd.DataFrame({'p': [.4, .8]}), ['a', 'other'])
        with patch.object(self.module, 'ClassCombination', FixedClassBase):
            model.fit(pd.DataFrame({'p': [.4, .8]}), ['a', 'z'])
        for p in (np.nan, np.inf, -1.):
            with self.assertRaises(ValueError):
                model.predict_proba(pd.DataFrame({'p': [p]}))

    def test_proper_decision_applies_frozen_temperature_bias_before_affine(self):
        c = decision_candidate()
        c['strategy'] = 'proper_macro_f1_stacking'
        c['decision_state']['theta'] = [np.log(2), .2, -.1]
        X = pd.DataFrame({'p': [.9, .4]})
        with patch.object(self.module, 'ClassCombination', FixedClassBase):
            model = self.module.SprintDecision(c).fit(X, ['a', 'z'])
        # exp(-ln2)=.5 so calibrated odds are sqrt(p0/p1)*exp(.3).
        odds = np.sqrt(np.array([9., 2 / 3])) * np.exp(.3)
        calibrated_a = odds / (1 + odds)
        expected = softmax(np.column_stack([calibrated_a, 2 * (1 - calibrated_a) + .2]), axis=1)
        np.testing.assert_allclose(model.predict_proba(X), expected)
        np.testing.assert_allclose(model.predict_proba(X.iloc[:1]), expected[:1])
        self.assertEqual(model.audit_['decision_fit_count'], 0)
        for theta in (None, [0., 0.], [np.nan, 0., 0.], [3., 0., 0.]):
            bad = copy.deepcopy(c)
            bad['decision_state']['theta'] = theta
            with self.assertRaises(ValueError):
                self.module.SprintDecision(bad)


if __name__ == '__main__':
    unittest.main()
