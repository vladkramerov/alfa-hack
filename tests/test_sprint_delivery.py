"""Delivery guard tests; no production archive or public learner fit."""
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


class SprintDeliveryTests(unittest.TestCase):
    def api(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), 'Sprint delivery script absent')
        return importlib.import_module(name)

    def test_clean_environment_removes_research_and_network_overrides(self):
        api = self.api('experiments.verify_sprint_delivery')
        env = api.clean_environment({'OPENAI_API_KEY': 'secret', 'OPENAI_BASE_URL': 'x',
            'SPEC006_X': '1', 'COMBO_RUN_DIR': 'x', 'PYTHONPATH': 'checkout',
            'AUTOML_MAX_EXPERIMENTS': '8', 'AUTOML_PORTFOLIO': '1', 'PATH': 'keep'})
        self.assertEqual(env['PATH'], 'keep')
        self.assertEqual(env['AUTOML_MAX_EXPERIMENTS'], '0')
        self.assertEqual(env['AUTOML_PORTFOLIO'], '0')
        self.assertEqual(env['AUTOML_TIME_LIMIT'], '1200')
        self.assertFalse(any(k.startswith(('OPENAI', 'SPEC006', 'COMBO')) for k in env))
        self.assertNotIn('PYTHONPATH', env)

    def test_transitive_local_imports_are_flat_and_missing_runtime_module_fails(self):
        api = self.api('experiments.package_sprint_delivery')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            (root/'solution.py').write_text('import automl_one\n', encoding='utf8')
            (root/'automl_one.py').write_text('def f():\n from sprint_two import answer\n', encoding='utf8')
            (root/'sprint_two.py').write_text('answer=42\n', encoding='utf8')
            self.assertEqual(api.runtime_files(root, ['solution.py']),
                             ['solution.py', 'automl_one.py', 'sprint_two.py'])
            (root/'sprint_two.py').unlink()
            with self.assertRaises(ValueError): api.runtime_files(root, ['solution.py'])

    def test_unimported_monotone_files_are_not_runtime_seeds(self):
        api = self.api('experiments.package_sprint_delivery')
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            (root/'solution.py').write_text('pass\n', encoding='utf8')
            (root/'sprint_monotone.py').write_text('pass\n', encoding='utf8')
            (root/'automl_sprint_monotone.py').write_text('import sprint_monotone\n', encoding='utf8')
            with patch.object(api, 'BASE_FILES', ['solution.py']):
                self.assertEqual(api.runtime_files(root), ['solution.py'])
                (root/'solution.py').write_text('import automl_sprint_monotone\n', encoding='utf8')
                self.assertEqual(api.runtime_files(root),
                    ['solution.py', 'automl_sprint_monotone.py', 'sprint_monotone.py'])

    def test_verified_pointer_checks_payload_hash_status_and_current_adapter_source(self):
        api = self.api('experiments.package_sprint_delivery')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            source = root/'automl_sprint.py'; source.write_text('x=1\n', encoding='utf8')
            record = root/'verified.json'
            record.write_text(json.dumps({'status': 'passed', 'source_sha256': {
                'automl_sprint.py': api.sha256(source)}}), encoding='utf8')
            pointer = {'path': 'verified.json', 'sha256': api.sha256(record)}
            evidence = api.read_verified(root, pointer, 'adapter')
            api.verify_declared_sources(root, evidence['record'])
            source.write_text('x=2\n', encoding='utf8')
            with self.assertRaises(ValueError): api.verify_declared_sources(root, evidence['record'])
            record.write_text('{"status":"failed"}', encoding='utf8')
            with self.assertRaises(ValueError): api.read_verified(root, pointer, 'adapter')
            pointer['sha256'] = api.sha256(record)
            with self.assertRaises(ValueError): api.read_verified(root, pointer, 'adapter')
            with self.assertRaises(ValueError): api.read_verified(root, {'path': '../escape.json', 'sha256': 'x'}, 'adapter')

    def test_latest_verification_pointer_is_dereferenced_without_weakening_hash_checks(self):
        api = self.api('experiments.package_sprint_delivery')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            result = root/'final.json'; result.write_text('{"passed":true}', encoding='utf8')
            pointer = root/'pointer.json'
            pointer.write_text(json.dumps({'passed': True, 'latest_verification': 'final.json',
                'sha256': api.sha256(result)}), encoding='utf8')
            evidence = api.read_verified(root, {'path': 'pointer.json', 'sha256': api.sha256(pointer)}, 'final')
            self.assertTrue(evidence['record']['passed'])
            self.assertEqual(evidence['resolved_path'], 'final.json')

    def test_flat_archive_rejects_extra_file_changed_source_and_traversal(self):
        api = self.api('experiments.verify_sprint_delivery')
        import hashlib
        payload = b'pass\n'; digest = hashlib.sha256(payload).hexdigest()
        for names, expected in [({'solution.py': payload}, True),
                                ({'solution.py': b'changed'}, False),
                                ({'solution.py': payload, 'data.csv': b'private'}, False),
                                ({'../solution.py': payload}, False)]:
            memory = io.BytesIO()
            with zipfile.ZipFile(memory, 'w') as archive:
                for name, value in names.items(): archive.writestr(name, value)
            memory.seek(0)
            with zipfile.ZipFile(memory) as archive:
                if expected:
                    api.verify_archive_members(archive, {'solution.py': digest})
                else:
                    with self.assertRaises(ValueError): api.verify_archive_members(archive, {'solution.py': digest})

    def test_summary_rejects_candidate_state_or_score_change_and_additional_refits(self):
        api = self.api('experiments.verify_sprint_delivery')
        expected = {'metric': 'rmse', 'score': 34., 'candidate': {'strategy': 'temporal_transport_pool',
            'pool_weights': [.4, .5, .1]}, 'source': 'results/run/report.md'}
        summary = {'status': 'completed', 'metric': 'rmse', 'selected_score': 34.,
            'selected': expected['candidate'], 'cached_history': True, 'submission_rows': 3, 'llm_calls': 0}
        refits = [{'candidate': expected['candidate'], 'result': {'status': 'completed',
            'mode': 'refit', 'train_rows': 8, 'test_rows': 3}}]
        api.verify_summary(summary, expected, refits, train_rows=8, test_rows=3)
        for changed in ({**summary, 'selected_score': 33.},
                        {**summary, 'selected': {'strategy': 'temporal_transport_pool', 'pool_weights': [1., 0., 0.]}},
                        {**summary, 'cached_history': False}):
            with self.assertRaises(ValueError): api.verify_summary(changed, expected, refits, train_rows=8, test_rows=3)
        with self.assertRaises(ValueError): api.verify_summary(summary, expected, refits*2, train_rows=8, test_rows=3)

    def test_package_refuses_overwrite_before_any_archive_mutation(self):
        api = self.api('experiments.package_sprint_delivery')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp); destination = root/'existing.zip'; destination.write_bytes(b'keep')
            with self.assertRaises(FileExistsError): api.package(root=root, destination=destination)
            self.assertEqual(destination.read_bytes(), b'keep')

    def test_embedded_winner_exact_state_source_outer_verification_and_adapter_coverage(self):
        api = self.api('experiments.package_sprint_delivery')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            baseline, known, runs = {}, [], []
            for kind, metric, number in [('regression', 'rmse', 91), ('multiclass', 'macro_f1', 78)]:
                folder = root/kind; folder.mkdir(); (folder/'report.md').write_text('verified', encoding='utf8')
                candidate = {'family': 'extra_trees', 'strategy': kind, 'frozen': [1., 2.]}
                baseline[kind] = {'metric': metric, 'score': .7, 'candidate': candidate,
                    'source': kind+'/report.md', 'hypothesis': 'HYP-'+str(number).zfill(3)}
                known.append({'metric': metric, 'hashes': {'train.csv': 'abc'}, 'split_key': 'split',
                    'records': [{'candidate': candidate, 'score': .7, 'source': kind+'/report.md'}]})
                runs.append({'directory': kind, 'stage': 'outer', 'status': 'completed',
                    'verified': True, 'metric': metric, 'score': .7})
            (root/'automl_known.py').write_text('KNOWN_TASKS = '+repr(known), encoding='utf8')
            identities = api.verify_winners(root, baseline, {'runs': runs}, {'checks': {'H91': {}, 'H78': {}}})
            self.assertEqual(identities['regression']['source'], 'regression/report.md')
            with self.assertRaises(ValueError): api.verify_winners(root, baseline, {'runs': runs}, {'checks': {'H78': {}}})
            runs[0]['verified'] = False
            with self.assertRaises(ValueError): api.verify_winners(root, baseline, {'runs': runs}, {'checks': {'H91': {}, 'H78': {}}})
            runs[0]['verified'] = True
            baseline['regression']['candidate']['frozen'][0] = 9.
            with self.assertRaises(ValueError): api.verify_winners(root, baseline, {'runs': runs}, {'checks': {'H91': {}, 'H78': {}}})


if __name__ == '__main__': unittest.main()
