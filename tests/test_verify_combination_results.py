"""Independent verifier rejects corrupted ordering, labels, and promotion."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np


class CombinationVerificationTests(unittest.TestCase):
    def test_saved_regression_uses_original_truth_and_rejects_reordered_positions(self):
        from experiments.verify_combination_results import score_saved_predictions
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            path = Path(directory) / 'pred.npz'
            np.savez(path, y=[2., 4.], prediction=[3., 5.], positions=[8, 9])
            self.assertEqual(score_saved_predictions('regression', path, np.array([2., 4.]), np.array([8, 9])), 1.)
            with self.assertRaisesRegex(ValueError, 'positions'):
                score_saved_predictions('regression', path, np.array([2., 4.]), np.array([9, 8]))
            with self.assertRaisesRegex(ValueError, 'truth'):
                score_saved_predictions('regression', path, np.array([2., 5.]), np.array([8, 9]))

    def test_saved_class_prediction_must_equal_probability_argmax(self):
        from experiments.verify_combination_results import score_saved_predictions
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp') as directory:
            path = Path(directory) / 'pred.npz'
            np.savez(path, y=['a', 'b'], prediction=['a', 'b'], positions=[4, 5], classes=['a', 'b'], probabilities=[[.7,.3],[.8,.2]])
            with self.assertRaisesRegex(ValueError, 'argmax'):
                score_saved_predictions('multiclass', path, np.array(['a', 'b']), np.array([4, 5]), np.array(['a', 'b']))

    def test_promotion_requires_global_verification_and_strict_incumbent_improvement(self):
        from experiments.verify_combination_results import promotable_best
        table = [{'task': 'regression', 'status': 'completed', 'verified': True, 'score': 40.,
                  'hypothesis': 'HYP-044', 'improved': True, 'learned_state': {'blend_weights': [1,0,0,0]},
                  'candidate': {'strategy': 'reg_convex_blend'}, 'directory': 'r', 'metric': 'rmse'}]
        self.assertEqual(promotable_best(table, False), {})
        result = promotable_best(table, True)
        self.assertEqual(result['regression']['learned_state']['blend_weights'], [1,0,0,0])
        table[0]['improved'] = False
        self.assertEqual(promotable_best(table, True), {})


if __name__ == '__main__':
    unittest.main()
