"""Fit-free independent calculations for SPEC-009 verification."""
import unittest
import numpy as np
import pandas as pd


class AlternativeVerificationTests(unittest.TestCase):
    def test_memory_uses_only_strictly_past_targets_and_canonical_numeric_keys(self):
        from experiments.verify_alternative_results import past_profile_components
        X = pd.DataFrame({'Year_Factor': [1, 2, 5], 'State_Factor': ['s'] * 3,
                          'building_class': ['c'] * 3, 'facility_type': ['f'] * 3,
                          'floor_area': [100, 100, 100], 'year_built': [2000] * 3,
                          'ELEVATION': [np.nan] * 3})
        valid = pd.concat([X.iloc[[0]], X.iloc[[0]]], ignore_index=True)
        valid['Year_Factor'] = [3, 2]
        valid['floor_area'] = [100., 100.]
        means, alpha, _ = past_profile_components(X, np.array([10., 20., 99999.]), valid)
        np.testing.assert_allclose(means, [50 / 3, 10.])
        np.testing.assert_allclose(alpha, [2 / 3, .5])

    def test_density_weights_clip_before_normalization(self):
        from experiments.verify_alternative_results import density_ratio_weights
        weights = density_ratio_weights(np.array([.2, .5, .8]), np.array([0, 0, 0, 1]))
        np.testing.assert_allclose(weights, np.array([.75, 3., 5.]) / (8.75 / 3))

    def test_gamma_reconstruction_matches_manual_onehot_distance(self):
        from experiments.verify_alternative_results import independent_gamma
        # Standardized numeric values [-1,1], plus opposite one-hot categories:
        # squared distance is4+2=6, gamma=1/6.
        X = pd.DataFrame({'number': [0., 2.], 'category': ['a', 'b']})
        gamma, rows, metadata = independent_gamma(X)
        self.assertAlmostEqual(gamma, 1 / 6)
        np.testing.assert_array_equal(rows, [0, 1])
        self.assertEqual(metadata['category']['kept'], 2)


if __name__ == '__main__':
    unittest.main()
