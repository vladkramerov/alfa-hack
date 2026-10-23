import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research_memory import Memory, promotion, semantic_recipe


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path.cwd() / 'tmp').mkdir(exist_ok=True)

    def test_started_is_persistent_before_training_and_seed_does_not_allow_repeat(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as d:
            ledger = Path(d) / 'HYPOTHESES.md'
            ledger.write_text('HYP-052 planned', encoding='utf8')
            history = Memory(Path(d) / 'history.json', ledger)
            X, y = pd.DataFrame({'x': [1, 2, 3]}), np.array(['001', 'NA', '001'])
            fp = history.claim({'family': 'mixed_neighbors', 'seed': 1}, X, y)
            entry = json.loads((Path(d) / 'history.json').read_text())[0]
            self.assertEqual(entry['status'], 'started')
            self.assertTrue(entry['ledger_sha256'])
            history.finish(fp, {'status': 'failed', 'error': 'test failure'})
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                history.claim({'family': 'mixed_neighbors', 'seed': 2}, X, y)

    def test_actual_data_change_is_a_new_fit(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as d:
            history = Memory(Path(d) / 'history.json')
            X = pd.DataFrame({'x': [1, 2, 3]})
            a = history.claim({'family': 'factorization_machine'}, X, [1, 2, 3])
            b = history.claim({'family': 'factorization_machine'}, X, [1, 2, 4])
            self.assertNotEqual(a, b)

    def test_promotion_relative_tradeoff_and_strict_boundaries(self):
        self.assertTrue(promotion({'rmse': 100, 'macro_f1': .5}, {'rmse': 99, 'macro_f1': .5})['accepted'])
        self.assertFalse(promotion({'rmse': 100, 'macro_f1': .5}, {'rmse': 100, 'macro_f1': .5})['accepted'])
        self.assertTrue(promotion({'rmse': 100, 'macro_f1': .5}, {'rmse': 84, 'macro_f1': .455})['accepted'])
        self.assertFalse(promotion({'rmse': 100, 'macro_f1': .5}, {'rmse': 85, 'macro_f1': .455})['accepted'])
        self.assertFalse(promotion({'rmse': 100, 'macro_f1': .5}, {'rmse': 84, 'macro_f1': .45})['accepted'])
        self.assertTrue(promotion({'rmse': 100, 'macro_f1': .5}, {'rmse': 109, 'macro_f1': .58})['accepted'])
        self.assertFalse(promotion({'rmse': 100, 'macro_f1': .5}, {'rmse': 110, 'macro_f1': .58})['accepted'])

    def test_semantic_recipe_keeps_actual_parameters(self):
        a = semantic_recipe({'seed': 3, 'parameters': {'rank': 8, 'random_state': 1}})
        self.assertEqual(a, {'parameters': {'rank': 8}})


if __name__ == '__main__':
    unittest.main()
