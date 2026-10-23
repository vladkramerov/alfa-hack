import json
import os
import unittest
from unittest.mock import patch

from automl_llm import parse_action, propose, research_context


class LLMTests(unittest.TestCase):
    def test_empty_and_malformed_responses(self):
        for value in (None, {}, {'choices': []}, {'choices': [None]},
                      {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]},
                      {'choices': [{'message': {'content': '{bad'}}]},
                      {'choices': [{'message': {'tool_calls': [{'function': {'name': 'open', 'arguments': '{}'}}]}}]}):
            self.assertIsNone(parse_action(value))

    def test_valid_tool_action(self):
        action = {'family': 'linear', 'hypothesis': 'Measured residual relation'}
        response = {'choices': [{'message': {'tool_calls': [{'function': {
            'name': 'run_experiment', 'arguments': json.dumps(action)}}]}}]}
        self.assertEqual(parse_action(response), action)
        self.assertEqual(parse_action({'choices': [{'message': {'content': '{"stop":true}'}}]}), {'stop': True})

    def test_missing_credentials_never_calls_network(self):
        with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen') as call:
            action, status = propose({}, [])
        self.assertIsNone(action)
        self.assertEqual(status['status'], 'unconfigured')
        call.assert_not_called()

    def test_http_error_is_redacted_and_one_attempt(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'secret-never-log', 'OPENAI_BASE_URL': 'http://localhost/v1', 'OPENAI_MODEL': 'test'}, clear=True):
            with patch('urllib.request.urlopen', side_effect=TimeoutError('secret-never-log')) as call:
                action, status = propose({}, [], timeout=.1)
        self.assertIsNone(action)
        self.assertEqual(status['attempts'], 1)
        self.assertNotIn('secret-never-log', json.dumps(status))
        self.assertEqual(call.call_count, 1)

    def test_context_retains_diagnostics_as_valid_json(self):
        payload = json.loads(research_context({'task': 'z' * 50000, 'numeric_summary': {str(i): [1] * 500 for i in range(100)}},
            [{'candidate': {'family': 'linear'}, 'diagnostics': {'per_class': {'rare': {'f1-score': .1}}}}]))
        self.assertIn('rare', payload['observations'][0]['diagnostics']['per_class'])
        self.assertLess(len(json.dumps(payload)), 12000)


if __name__ == '__main__':
    unittest.main()
