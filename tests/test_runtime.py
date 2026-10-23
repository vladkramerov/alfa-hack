import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from automl_runtime import Budget, child_environment, run_child


class RuntimeTests(unittest.TestCase):
    def test_secret_free_worker_environment(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'secret', 'AWS_SECRET_ACCESS_KEY': 'also-secret'}, clear=False):
            env = child_environment(2)
            self.assertNotIn('OPENAI_API_KEY', env)
            self.assertNotIn('AWS_SECRET_ACCESS_KEY', env)
            self.assertEqual(env['OMP_NUM_THREADS'], '2')

    def test_kills_hanging_worker_before_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'hang.py'
            script.write_text('import time\ntime.sleep(60)\n')
            start = time.monotonic()
            result = run_child(script, [], Path(tmp) / 'run', .25)
            self.assertEqual(result['status'], 'timeout')
            self.assertLess(time.monotonic() - start, 3.)

    def test_budget_caps_platform_limit(self):
        budget = Budget(50000)
        self.assertEqual(budget.seconds, 10200)
        self.assertGreater(budget.reserve(), 0)
        self.assertLess(budget.reserve(), budget.remaining())


if __name__ == '__main__':
    unittest.main()
