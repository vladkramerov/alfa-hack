"""Failure-injection checks for the controller, worker and optional LLM."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from automl_data import load_task
from automl_engine import main
from automl_llm import propose
from automl_runtime import run_child


def _response(action=None):
    arguments = json.dumps(action if action is not None else {'stop': True})
    return {'choices': [{'message': {'tool_calls': [{'function': {
        'name': 'run_experiment', 'arguments': arguments}}]}}]}


class MockAPI:
    def __init__(self, modes):
        self.modes = deque(modes)
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get('Content-Length', '0'))
                owner.requests.append(json.loads(self.rfile.read(length)))
                mode = owner.modes.popleft() if owner.modes else 'stop'
                if mode == 'delay':
                    time.sleep(.3)
                    mode = 'stop'
                if mode == 'http500':
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'failure')
                    return
                if mode == 'empty':
                    body = b''
                elif mode == 'invalid_json':
                    body = b'{not-json'
                elif mode == 'stop':
                    body = json.dumps(_response()).encode()
                elif isinstance(mode, dict):
                    body = json.dumps(_response(mode)).encode()
                else:
                    raise AssertionError(mode)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'


def _task(directory, metric='rmse', n=120):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(818)
    x = rng.normal(size=n + 24)
    z = rng.normal(size=n + 24)
    ids = [f'row-{i:04d}' for i in range(n + 24)]
    train = pd.DataFrame({'row_id': ids[:n], 'measurement': x[:n], 'other': z[:n],
                          'target_value': 2 + 3 * x[:n] - z[:n]})
    test = pd.DataFrame({'row_id': ids[n:], 'measurement': x[n:], 'other': z[n:]})
    sample = pd.DataFrame({'row_id': test.row_id.to_numpy()[::-1], 'target_value': 0.})
    train.to_csv(directory / 'observations.csv', index=False)
    test.to_csv(directory / 'queries.csv', index=False)
    sample.to_csv(directory / 'answer_template.csv', index=False)
    (directory / 'task.md').write_text(
        f'Primary metric: {metric}\nTarget: target_value\nIdentifier: row_id\n'
        'Training data: observations.csv\nTest data: queries.csv\n'
        'Sample submission: answer_template.csv\n', encoding='utf-8')
    return directory / 'task.md', sample


class EngineReliabilityTests(unittest.TestCase):
    def _api_env(self, server):
        return patch.dict(os.environ, {
            'OPENAI_BASE_URL': server.url,
            'OPENAI_API_KEY': 'test-only-key',
            'OPENAI_MODEL': 'mock-model',
        }, clear=False)

    def test_llm_http_failures_empty_invalid_repeated_and_timeout_are_single_attempts(self):
        with MockAPI(['empty', 'http500', 'invalid_json', 'stop', 'stop', 'delay']) as api:
            with self._api_env(api):
                results = [propose({'metric': 'rmse'}, [], timeout=.06) for _ in range(3)]
                repeated = [propose({'metric': 'rmse'}, [], timeout=.5) for _ in range(2)]
                delayed = propose({'metric': 'rmse'}, [], timeout=.06)
        self.assertEqual([item[1]['status'] for item in results], ['failed', 'failed', 'failed'])
        self.assertEqual([item[1]['attempts'] for item in results], [1, 1, 1])
        self.assertTrue(all(action is None for action, _ in results))
        self.assertEqual(repeated[0][0], {'stop': True})
        self.assertEqual(repeated[1][0], {'stop': True})
        self.assertEqual(delayed[1]['status'], 'failed')
        self.assertEqual(delayed[1]['error'], 'TimeoutError')
        self.assertEqual(len(api.requests), 6)

    def test_cli_evaluates_generated_feature_and_feeds_diagnostics_to_next_proposal(self):
        operation = {'family': 'linear', 'hypothesis': 'Use measured interaction',
                     'feature_operations': [{'op': 'mul', 'name': 'measured_product',
                                             'left': 'measurement', 'right': 'other'}]}
        with tempfile.TemporaryDirectory() as tmp, MockAPI([operation, 'stop']) as api:
            task, _ = _task(Path(tmp) / 'fixture')
            env = os.environ.copy()
            env.update(OPENAI_BASE_URL=api.url, OPENAI_API_KEY='test-only-key',
                       OPENAI_MODEL='mock-model', AUTOML_TIME_LIMIT='45',
                       AUTOML_MAX_EXPERIMENTS='3', AUTOML_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(Path(tmp) / 'mpl'))
            started = time.monotonic()
            process = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'solution.py'), str(task)],
                cwd=task.parent, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=45)
            self.assertEqual(process.returncode, 0, process.stdout[-4000:])
            self.assertLess(time.monotonic() - started, 45)
            self.assertEqual(len(api.requests), 2)
            second_prompt = api.requests[1]['messages'][1]['content']
            prompt_data = json.loads(second_prompt)
            self.assertTrue(prompt_data['observations'][0]['diagnostics'])
            runs = sorted((task.parent / 'agent_runs').glob('*/experiment-*/generated_features.py'))
            self.assertTrue(runs, 'generated feature was not exported by a completed worker')
            source = runs[0].read_text(encoding='utf-8')
            self.assertIn("X['measured_product']", source)
            summaries = list((task.parent / 'agent_runs').glob('*/summary.json'))
            summary = json.loads(summaries[0].read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'completed')
            self.assertGreaterEqual(summary['experiments'], 2)

    def test_unknown_feature_fails_worker_and_keeps_valid_fallback_submission(self):
        operation = {'family': 'linear', 'hypothesis': 'Bad column probe',
                     'feature_operations': [{'op': 'mul', 'name': 'broken_product',
                                             'left': 'missing_feature', 'right': 'other'}]}
        with tempfile.TemporaryDirectory() as tmp, MockAPI([operation, 'stop']) as api:
            task, sample = _task(Path(tmp) / 'fixture')
            env = os.environ.copy()
            env.update(OPENAI_BASE_URL=api.url, OPENAI_API_KEY='test-only-key',
                       OPENAI_MODEL='mock-model', AUTOML_TIME_LIMIT='45',
                       AUTOML_MAX_EXPERIMENTS='2', AUTOML_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(Path(tmp) / 'mpl'))
            process = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'solution.py'), str(task)],
                cwd=task.parent, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=45)
            self.assertEqual(process.returncode, 0, process.stdout[-4000:])
            output = pd.read_csv(task.parent / 'submission.csv')
            self.assertEqual(output['row_id'].tolist(), sample['row_id'].tolist())
            record_files = list((task.parent / 'agent_runs').glob('*/experiments.json'))
            records = json.loads(record_files[0].read_text(encoding='utf-8'))
            failed = next(record for record in records if record['candidate'].get('feature_operations'))
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('unknown column', failed['error'].lower())

    def test_engine_deduplicates_identical_candidate_despite_new_hypothesis(self):
        first = {'family': 'linear', 'parameters': {'alpha': 2.0},
                 'hypothesis': 'Try a regularized linear fit'}
        repeated = {'family': 'linear', 'parameters': {'alpha': 2.0},
                    'hypothesis': 'A different explanation for the same fit'}
        with tempfile.TemporaryDirectory() as tmp, MockAPI([first, repeated, 'stop']) as api:
            task, _ = _task(Path(tmp) / 'fixture')
            env = os.environ.copy()
            env.update(OPENAI_BASE_URL=api.url, OPENAI_API_KEY='test-only-key',
                       OPENAI_MODEL='mock-model', AUTOML_TIME_LIMIT='45',
                       AUTOML_MAX_EXPERIMENTS='4', AUTOML_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(Path(tmp) / 'mpl'))
            process = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'solution.py'), str(task)],
                cwd=task.parent, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=45)
            self.assertEqual(process.returncode, 0, process.stdout[-4000:])
            self.assertEqual(len(api.requests), 3)
            summaries = list((task.parent / 'agent_runs').glob('*/summary.json'))
            summary = json.loads(summaries[0].read_text(encoding='utf-8'))
            self.assertEqual(summary['experiments'], 2)  # baseline and one distinct LLM candidate
            self.assertEqual(summary['duplicate_proposals'], 1)
            self.assertEqual(summary['llm_calls'], 3)

    def test_two_second_budget_child_timeout_preserves_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, sample = _task(Path(tmp) / 'fixture')
            with patch.dict(os.environ, {'AUTOML_TIME_LIMIT': '2', 'AUTOML_MAX_EXPERIMENTS': '1',
                                         'AUTOML_THREADS': '2'}, clear=False):
                with patch('automl_engine.run_child', return_value={'status': 'timeout', 'seconds': .2}):
                    summary = main(str(task), threads=2)
            self.assertEqual(summary['status'], 'fallback')
            self.assertEqual(pd.read_csv(task.parent / 'submission.csv')['row_id'].tolist(),
                             sample['row_id'].tolist())
            record_files = list((task.parent / 'agent_runs').glob('*/experiments.json'))
            records = json.loads(record_files[0].read_text(encoding='utf-8'))
            self.assertEqual(records[0]['status'], 'timeout')
            self.assertLessEqual(summary['seconds'], 2.5)

    def test_unknown_explicit_metric_fails_before_engine_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, _ = _task(Path(tmp) / 'fixture', metric='mystery_score')
            with self.assertRaisesRegex(ValueError, 'Unsupported explicit metric'):
                load_task(task)
            self.assertFalse((task.parent / 'submission.csv').exists())

    def test_run_child_kills_worker_process_tree_on_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'hang.py'
            script.write_text('import time\ntime.sleep(60)\n', encoding='utf-8')
            result = run_child(script, [], Path(tmp) / 'run', .2, threads=2)
            self.assertEqual(result['status'], 'timeout')
            self.assertLess(result['seconds'], 3.)


if __name__ == '__main__':
    unittest.main()
