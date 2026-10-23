from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from research_data import Task
from experiments.run_clean_rewrite import protocol


class ProtocolTests(unittest.TestCase):
    def test_temporal_confirmation_is_four_and_inner_is_strictly_past(self):
        X = pd.DataFrame({'Year_Factor': np.repeat([1, 2, 3, 4, 5], 20), 'x': np.arange(100)})
        task = Task(X, pd.Series(np.arange(100.)), pd.DataFrame({'Year_Factor': [6], 'x': [101]}),
                    pd.DataFrame({'id': ['001'], 'y': [0.]}), Path('task.md'), 'regression', 'rmse', 'id', 'y', {})
        result = protocol(task)
        self.assertEqual(set(X.iloc[result['confirmation_valid']].Year_Factor), {4})
        self.assertLess(X.iloc[result['inner_fit']].Year_Factor.max(), X.iloc[result['inner_valid']].Year_Factor.min())
        self.assertTrue(set(result['inner_fit']) <= set(result['confirmation_fit']))
        self.assertFalse(set(result['inner_valid']) & set(result['confirmation_valid']))

    def test_classification_exact_groups_never_cross_confirmation_or_inner(self):
        X = pd.DataFrame({'x': np.repeat(np.arange(100), 2), 'c': np.repeat(['a', 'b'], 100)})
        task = Task(X, pd.Series(np.repeat(np.where(np.arange(100) % 2, '001', 'NA'), 2)),
                    X.iloc[:4].copy(), pd.DataFrame({'id': ['1', '2', '3', '4'], 'y': ['NA'] * 4}),
                    Path('task.md'), 'classification', 'macro_f1', 'id', 'y', {}, raw_X=X.copy())
        result = protocol(task)
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        for stage in ['inner', 'confirmation']:
            self.assertFalse(set(groups[result[stage + '_fit']]) & set(groups[result[stage + '_valid']]))
        self.assertTrue(set(result['inner_fit']) <= set(result['confirmation_fit']))


if __name__ == '__main__':
    unittest.main()
