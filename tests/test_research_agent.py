import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_agent import parse_action, validate_recipe, propose, execute_worker


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path.cwd() / 'tmp').mkdir(exist_ok=True)

    def test_empty_and_truncated_llm_responses_are_rejected(self):
        self.assertIsNone(parse_action({'choices': []}))
        self.assertIsNone(parse_action({'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}))
        self.assertIsNone(parse_action({'choices': [{'message': {'content': 'not json'}}]}))
        self.assertEqual(parse_action({'choices': [{'message': {'content': '{"stop":true}'}}]}), {'stop': True})

    def test_typed_recipe_rejects_wrong_kind_and_unbounded_parameters(self):
        with self.assertRaises(ValueError):
            validate_recipe({'family': 'mixed_neighbors'}, 'regression')
        with self.assertRaises(ValueError):
            validate_recipe({'family': 'factorization_machine', 'parameters': {'epochs': 100000}}, 'regression')
        with self.assertRaises(ValueError):
            validate_recipe({'family': 'factorization_machine', 'parameters': {'hidden_file': '/etc/passwd'}}, 'regression')
        self.assertEqual(validate_recipe({'family': 'mixed_neighbors'}, 'classification')['family'], 'mixed_neighbors')

    def test_unconfigured_api_preserves_local_execution(self):
        with patch.dict('os.environ', {}, clear=True):
            action, status = propose({'diagnostics': {'f1': .5}}, [])
        self.assertIsNone(action)
        self.assertEqual(status['status'], 'unconfigured')

    def test_worker_writes_failure_instead_of_silent_exception(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as d:
            request = {'task_path': str(Path(d) / 'missing.md'), 'output': d,
                       'recipe': {'family': 'mixed_neighbors'}, 'mode': 'evaluate'}
            self.assertNotEqual(execute_worker(request), 0)
            result = json.loads((Path(d) / 'metrics.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertTrue((Path(d) / 'traceback.txt').exists())


import numpy as np
import pandas as pd
from research_data import Task, save_submission
from research_memory import Memory, data_digest
from research_agent import main, launch_worker, prediction_for


class ControllerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path.cwd() / 'tmp').mkdir(exist_ok=True)

    def fixture(self, directory, metric='rmse', outputs=None):
        root = Path(directory)
        task_path = root / 'task.md'
        task_path.write_text('synthetic task', encoding='utf8')
        classification = metric in {'log_loss', 'roc_auc'}
        sample = pd.DataFrame({'id': ['001', 'NA']})
        for column in outputs or ['target']:
            sample[column] = 0.
        return Task(pd.DataFrame({'x': np.arange(12)}),
                    pd.Series(['b', 'a'] * 6 if classification else np.arange(12, dtype=float)),
                    pd.DataFrame({'x': [12, 13]}), sample, task_path,
                    'classification' if classification else 'regression', metric, 'id', 'target', {},
                    probability_columns=outputs or [], class_order=['a', 'b'] if classification else [])

    def run_controller(self, task, worker, proposal=None, extra=None):
        with patch('research_data.load_task', return_value=task), patch('research_data.split_data', return_value=(np.arange(8), np.arange(8, 12), {})), patch('research_agent.launch_worker', side_effect=worker), patch('research_agent.propose', side_effect=proposal or (lambda *a, **k: (None, {'status': 'unconfigured'}))):
            return main([str(task.task_path), '--time-budget', '100', '--max-experiments', '3', *(extra or [])])

    def test_unconfigured_api_searches_other_families_after_consulting_llm(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            task, events = self.fixture(directory), []
            def worker(request, timeout):
                events.append(request['recipe']['family'])
                return {'status': 'failed'}
            def proposal(*a, **k):
                events.append('api')
                return None, {'status': 'unconfigured'}
            self.run_controller(task, worker, proposal)
            self.assertEqual(events, ['factorization_machine', 'api', 'lightgbm', 'api', 'catboost'])

    def test_stop_action_prevents_local_search(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            task, families = self.fixture(directory), []
            self.run_controller(task, lambda r, t: families.append(r['recipe']['family']) or {'status': 'failed'}, lambda *a, **k: ({'stop': True}, {'status': 'ok'}))
            self.assertEqual(families, ['factorization_machine'])

    def test_api_retries_share_one_absolute_deadline(self):
        clock, timeouts = [0.], []
        def fail(request, timeout):
            timeouts.append(timeout)
            clock[0] += 4.
            raise TimeoutError()
        with patch.dict('os.environ', {'OPENAI_BASE_URL': 'https://unused.invalid', 'OPENAI_API_KEY': 'fake', 'OPENAI_MODEL': 'fake'}, clear=True), patch('research_agent.time.monotonic', side_effect=lambda: clock[0]), patch('research_agent.urllib.request.urlopen', side_effect=fail):
            propose({}, [], timeout=5)
        self.assertEqual(timeouts, [5., 1.])

    def test_expired_llm_budget_starts_no_worker_or_final_refit(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            task, clock, modes = self.fixture(directory), [0.], []
            def worker(request, timeout):
                modes.append(request['mode'])
                return {'status': 'completed', 'score': 1.}
            def proposal(*a, **k):
                clock[0] = 101.
                return {'family': 'lightgbm'}, {'status': 'ok'}
            with patch('research_agent.time.monotonic', side_effect=lambda: clock[0]):
                self.run_controller(task, worker, proposal)
            self.assertEqual(modes, ['evaluate'])

    def test_restart_keeps_valid_previous_submission(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            task = self.fixture(directory)
            save_submission(task, np.array([44., 55.]))
            self.run_controller(task, lambda *a: {'status': 'failed'}, extra=['--max-experiments', '0'])
            self.assertEqual(pd.read_csv(Path(directory) / 'submission.csv')['target'].tolist(), [44., 55.])

    def test_resume_uses_compatible_inner_and_full_refit_artifact_without_fit(self):
        from research_agent import validate_recipe
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            task = self.fixture(directory)
            root = Path(directory) / 'research_runs'
            root.mkdir()
            recipe = validate_recipe({'family': 'factorization_machine'}, 'regression')
            memory = Memory(root / 'hypotheses.json')
            fingerprint = memory.claim(recipe, task.X.iloc[:8], task.y.iloc[:8], 'evaluate')
            memory.finish(fingerprint, {'status': 'completed', 'mode': 'evaluate', 'metric': 'rmse', 'score': .01,
                          'validation_X_sha256': data_digest(task.X.iloc[8:]), 'validation_y_sha256': data_digest(task.y.iloc[8:].to_numpy().reshape(-1, 1))})
            artifact = root / 'old-final.csv'
            save_submission(task, np.array([77., 88.]), artifact)
            fingerprint = memory.claim(recipe, task.X, task.y, 'refit')
            memory.finish(fingerprint, {'status': 'completed', 'mode': 'refit', 'submission_path': str(artifact),
                          'test_X_sha256': data_digest(task.X_test), 'sample_sha256': data_digest(task.sample)})
            self.run_controller(task, lambda *a: self.fail('Compatible evidence must avoid model fitting'), extra=['--max-experiments', '1'])
            self.assertEqual(pd.read_csv(Path(directory) / 'submission.csv')['target'].tolist(), [77., 88.])

    def test_worker_frozen_source_override_is_executed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            root = Path(directory)
            frozen = root / 'frozen.py'
            frozen.write_text('import json,sys\nfrom pathlib import Path\nr=json.loads(Path(sys.argv[2]).read_text())\nPath(r["output"],"metrics.json").write_text(json.dumps({"status":"completed","frozen":sys.argv[1]}))\n', encoding='utf8')
            result = launch_worker({'output': str(root / 'run'), 'history': str(root / 'history.json'), 'worker_source': str(frozen.resolve())}, 10)
            self.assertEqual(result, {'status': 'completed', 'frozen': '--worker'})

    def test_refit_class_probability_columns_follow_labels_for_both_metrics(self):
        class Model:
            classes_ = np.array(['b', 'a'])
            audit_ = {}
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                return np.tile([.8, .2], (len(X), 1))
        for metric in ['log_loss', 'roc_auc']:
            for outputs in [['a', 'b'], ['b', 'a']]:
                with self.subTest(metric=metric, outputs=outputs), tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
                    task = self.fixture(directory, metric, outputs)
                    request = {'task_path': str(task.task_path), 'output': str(Path(directory) / 'refit'), 'history': str(Path(directory) / 'history.json'), 'recipe': {'family': 'mixed_neighbors'}, 'mode': 'refit'}
                    with patch('research_data.load_task', return_value=task), patch('research_agent.make_model', return_value=Model()):
                        self.assertEqual(execute_worker(request), 0)
                    frame = pd.read_csv(Path(request['output']) / 'candidate_submission.csv')
                    np.testing.assert_allclose(frame['a'], [.2, .2])
                    np.testing.assert_allclose(frame['b'], [.8, .8])

    def test_binary_log_loss_single_column_writes_positive_class_probability(self):
        class Model:
            classes_ = np.array(['b', 'a'])
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                return np.tile([.8, .2], (len(X), 1))
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as directory:
            task = self.fixture(directory, 'log_loss', ['b'])
            request = {'task_path': str(task.task_path), 'output': str(Path(directory) / 'refit'), 'history': str(Path(directory) / 'history.json'), 'recipe': {'family': 'mixed_neighbors'}, 'mode': 'refit'}
            with patch('research_data.load_task', return_value=task), patch('research_agent.make_model', return_value=Model()):
                self.assertEqual(execute_worker(request), 0)
            np.testing.assert_allclose(pd.read_csv(Path(request['output']) / 'candidate_submission.csv')['b'], [.8, .8])


if __name__ == '__main__':
    unittest.main()
