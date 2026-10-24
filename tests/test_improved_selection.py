"""Selection contracts for measured portable SPEC-006 strategies; no real fits."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from automl_history import already_known, candidate_key, compatible_policy, known_task
from automl_known import KNOWN_TASKS as RUNTIME_KNOWN_TASKS
import json
from pathlib import Path
KNOWN_TASKS = json.loads((Path(__file__).resolve().parents[1]/'results/2026-10-24/sergey-improvement-v1/offline-historical-evidence.json').read_text(encoding='utf8'))
from automl_models import canonical_candidate


class ImprovedSelectionTests(unittest.TestCase):
    def test_runtime_contains_no_embedded_dataset_state(self):
        self.assertEqual(RUNTIME_KNOWN_TASKS, [])
        for prior in KNOWN_TASKS:
            self.assertIsNone(known_task(prior['hashes'],prior['split_key'],prior['metric']))

    def test_group_experts_keeps_exact_unscaled_policy(self):
        c = canonical_candidate({'family': 'extra_trees', 'strategy': 'group_experts'})
        self.assertEqual(c['preprocessing'], 'group-experts-v1')
        self.assertEqual(c['target_transform'], 'log1p')
        self.assertFalse(c['scale'])
        self.assertEqual(c['max_outlier_fraction'], 0)
        self.assertEqual(c['group_column'], 'building_class')
        self.assertEqual(c['minimum_expert_rows'], 500)
        self.assertEqual(c['shrinkage'], 500.)
        self.assertTrue(compatible_policy(c))

    def test_calibration_keeps_fixed_label_mapping(self):
        source = {'family': 'xgboost', 'strategy': 'macro_f1_calibrated',
                  'class_multipliers': {'rare': 2., 'common': .75}}
        c = canonical_candidate(source)
        self.assertEqual(c['class_multipliers'], source['class_multipliers'])
        self.assertEqual(c['calibration_folds'], 3)
        self.assertEqual(c['preprocessing'], 'macro-f1-v1')
        self.assertTrue(compatible_policy(c))
        c['class_multipliers']['rare'] = 1.
        self.assertEqual(source['class_multipliers']['rare'], 2.)

    def test_strategy_identity_covers_all_prediction_changing_fields(self):
        base = {'family': 'extra_trees', 'strategy': 'group_experts', 'group_column': 'building_class',
                'minimum_expert_rows': 500, 'shrinkage': 500.}
        for change in ({'strategy': 'ordinary'}, {'group_column': 'other'},
                       {'minimum_expert_rows': 501}, {'shrinkage': 400.}):
            with self.subTest(change=change):
                self.assertNotEqual(candidate_key(base), candidate_key({**base, **change}))
        macro = {'family': 'xgboost', 'strategy': 'macro_f1_calibrated',
                 'class_multipliers': {'a': 1., 'b': 2.}}
        self.assertNotEqual(candidate_key(macro), candidate_key({**macro, 'class_multipliers': {'a': 1., 'b': 1.}}))
        self.assertEqual(candidate_key(macro), candidate_key({**macro, 'class_multipliers': {'b': 2., 'a': 1.}}))

    def test_invalid_advanced_policies_are_rejected(self):
        for candidate in ({'family': 'lightgbm', 'strategy': 'group_experts'},
                          {'family': 'xgboost', 'strategy': 'macro_f1_calibrated', 'class_multipliers': {'a': np.nan}},
                          {'family': 'xgboost', 'strategy': 'macro_f1_calibrated', 'class_multipliers': {'a': 0.}}):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    canonical_candidate(candidate)
        self.assertFalse(compatible_policy({'family': 'extra_trees', 'preprocessing': 'group-experts-v1', 'scale': False}))

    def test_known_measured_winners_are_compatible_and_not_repeated(self):
        for metric, strategy, expected in (('rmse', 'group_experts', 41.480963027907194),
                                           ('macro_f1', 'macro_f1_calibrated', .6286047899141216)):
            prior = next(t for t in KNOWN_TASKS if t['metric'] == metric)
            measured = [r for r in prior['records'] if r['candidate'].get('strategy') == strategy]
            self.assertEqual(len(measured), 1)
            record = measured[0]
            self.assertEqual(record['score'], expected)
            self.assertTrue(compatible_policy(record['candidate']))
            self.assertTrue(already_known(canonical_candidate({'family': record['candidate']['family'], 'strategy': strategy}), [], prior))
            with patch('automl_history.KNOWN_TASKS', KNOWN_TASKS):
                self.assertIs(known_task(prior['hashes'], prior['split_key'], metric), prior)
            self.assertIsNone(known_task({**prior['hashes'], 'train.csv': 'unknown'}, prior['split_key'], metric))
            self.assertIsNone(known_task(prior['hashes'], 'different split', metric))
        macro = next(t for t in KNOWN_TASKS if t['metric'] == 'macro_f1')
        winner = next(r for r in macro['records'] if r['candidate'].get('strategy') == 'macro_f1_calibrated')
        self.assertEqual(winner['candidate']['class_multipliers']['Worms'], 2.)

    def test_old_spec006_directions_are_blocked_on_matching_task(self):
        prior = next(t for t in KNOWN_TASKS if t['metric'] == 'rmse')
        candidate = canonical_candidate({'family': 'extra_trees', 'hypothesis': 'HYP-030 renamed seasonal Fourier'})
        self.assertTrue(already_known(candidate, [], prior))
        self.assertFalse(already_known(candidate, [], None))
        self.assertGreaterEqual(len(prior.get('spec006_attempts', [])), 13)

    def test_generic_portfolio_includes_only_applicable_strategies(self):
        import solution
        X = pd.DataFrame({'building_class': ['a'] * 500 + ['b'] * 500, 'x': np.arange(1000.)})
        reg = solution.portfolio_candidates('regression', 'rmse', X, pd.Series(np.ones(1000)))
        self.assertEqual(reg[0]['strategy'], 'group_experts')
        single = solution.portfolio_candidates('regression', 'rmse', X.assign(building_class='a'), pd.Series(np.ones(1000)))
        self.assertFalse(any(c.get('strategy') == 'group_experts' for c in single))
        macro = solution.portfolio_candidates('classification', 'macro_f1', X, pd.Series(['x', 'y'] * 500))
        self.assertEqual(macro[0]['strategy'], 'macro_f1_calibrated')
        accuracy = solution.portfolio_candidates('classification', 'accuracy', X, pd.Series(['x', 'y'] * 500))
        self.assertFalse(any(c.get('strategy') == 'macro_f1_calibrated' for c in accuracy))

    def test_refit_candidate_freezes_calibration_without_changing_evaluation_identity(self):
        import solution
        candidate = canonical_candidate({'family': 'xgboost', 'strategy': 'macro_f1_calibrated'})
        original = copy.deepcopy(candidate)
        frozen = solution.refit_candidate(candidate, {'calibration_multipliers': {'a': 1., 'b': 2.}})
        self.assertEqual(candidate, original)
        self.assertEqual(frozen['class_multipliers'], {'a': 1., 'b': 2.})
        self.assertNotEqual(candidate_key(candidate), candidate_key(frozen))


if __name__ == '__main__':
    unittest.main()
