import importlib.util
from pathlib import Path
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split

spec = importlib.util.spec_from_file_location('tuning', Path(__file__).resolve().parents[1] / 'tune_models.py')
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)


class SplitIsolationTests(unittest.TestCase):
    def check_isolation(self, parts):
        fit, outer, inner_fit, inner_valid = map(set, parts[:4])
        self.assertFalse(fit & outer)
        self.assertFalse(inner_fit & inner_valid)
        self.assertEqual(inner_fit | inner_valid, fit)
        self.assertFalse((inner_fit | inner_valid) & outer)

    def test_regression_uses_only_past_years(self):
        X = pd.DataFrame({'Year_Factor': [1, 2, 3, 4, 5] * 10})
        parts = tuning.split_indices(X, pd.Series(range(50)), True)
        self.check_isolation(parts)
        self.assertEqual(set(X.loc[parts[1], 'Year_Factor']), {5})
        self.assertEqual(set(X.loc[parts[3], 'Year_Factor']), {4})
        self.assertLess(X.loc[parts[2], 'Year_Factor'].max(), 4)

    def test_classification_preserves_original_outer_split_and_classes(self):
        X = pd.DataFrame({'x': range(200)})
        y = pd.Series(['common'] * 160 + ['rare'] * 40)
        parts = tuning.split_indices(X, y, False)
        self.check_isolation(parts)
        old_fit, old_outer = train_test_split(X.index, test_size=0.2, random_state=42, stratify=y)
        self.assertEqual(list(parts[0]), list(old_fit))
        self.assertEqual(list(parts[1]), list(old_outer))
        for ids in parts[:4]: self.assertEqual(set(y.loc[ids]), set(y))


if __name__ == '__main__':
    unittest.main()
