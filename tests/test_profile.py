from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from automl_data import TaskData
from automl_engine import profile_data


class ProfileTests(unittest.TestCase):
    def test_drift_uses_test_features_but_profile_excludes_validation_targets(self):
        rng = np.random.default_rng(11)
        train = pd.DataFrame({'unstable': np.arange(100, dtype=float),
                              'stable': rng.normal(size=100),
                              'target': np.r_[np.arange(80, dtype=float), np.full(20, 1e9)]})
        test = pd.DataFrame({'unstable': np.arange(1000, 1020, dtype=float),
                             'stable': rng.normal(size=20)})
        bundle = TaskData(Path('/tmp/task.md'), '', {}, train, test,
                          pd.DataFrame({'target': np.zeros(20)}), pd.DataFrame(),
                          'target', None, 'regression', 'rmse')
        profile = profile_data(bundle, np.arange(80), 'temporal:test')
        self.assertEqual(profile['target_fit_summary']['max'], 79)
        self.assertIn('unstable', profile['drift_columns'])
        self.assertNotIn('stable', profile['drift_columns'])
        self.assertAlmostEqual(profile['distribution_shift']['unstable']['outside_fit_1_99'], 1.)


if __name__ == '__main__':
    unittest.main()
