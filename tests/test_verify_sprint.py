"""Synthetic corruption tests for the SPEC011 read-only verifier (zero fits)."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import verify_sprint as v


class MetricTests(unittest.TestCase):
    def test_rmse_casts_before_integer_subtraction_and_square(self):
        self.assertEqual(v.metric(np.array([255], np.uint8), np.array([0], np.uint8), 'rmse'), 255.)

    def test_rmse_known_arithmetic(self):
        self.assertEqual(v.metric(np.array([1., 2.]), np.array([4., 6.]), 'rmse'), np.sqrt(12.5))

    def test_macro_f1_union_includes_predicted_only_label(self):
        self.assertAlmostEqual(v.metric(np.array(['a', 'a']), np.array(['a', 'b']), 'macro_f1'), 1 / 3)

    def test_lexical_numeric_class_names_remain_distinct(self):
        self.assertEqual(v.metric(np.array(['01', '1']), np.array(['01', '1']), 'macro_f1'), 1.)

    def test_invalid_metric_inputs_rejected(self):
        invalid = [([], [], 'rmse'), ([1.], [[1.]], 'rmse'), (['1'], ['1'], 'rmse'),
                   ([np.nan], [0], 'rmse'), ([1], [np.inf], 'rmse'),
                   (np.array(['a'], object), np.array(['a']), 'macro_f1'), ([1], [1], 'macro_f1'),
                   (['a'], ['a'], 'accuracy')]
        for y, p, name in invalid:
            with self.subTest(y=y, metric=name), self.assertRaises(ValueError):
                v.metric(np.asarray(y), np.asarray(p), name)

    def test_probability_roundoff_normalized(self):
        result = v.probabilities(np.array([[.3, .70000001]], np.float32), np.array(['a', 'b']), 1)
        self.assertAlmostEqual(result.sum(), 1.)

    def test_invalid_probabilities_rejected(self):
        for p in [[[.2, .2]], [[-.1, 1.1]], [[np.nan, .5]], [[1, 0], [0, 1]], [['0', '1']]]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                v.probabilities(np.asarray(p), np.array(['a', 'b']), 1)
        with self.assertRaises(ValueError):
            v.probabilities(np.array([[.5, .5]]), np.array(['a', 'a']), 1)
        with self.assertRaises(ValueError):
            v.probabilities(np.array([[.5, .5]]), np.array(['a', 'b'], object), 1)

    def test_close_rejects_nonfinite_and_mismatch(self):
        for a, b in [(1., 1.001), (np.nan, np.nan), (np.inf, np.inf)]:
            with self.assertRaises(ValueError):
                v.close(a, b, 'synthetic')


class PositionTests(unittest.TestCase):
    def test_disjoint_complete_partition(self):
        v.partition(np.array([2, 0]), np.array([1, 3]), 4, full=True)

    def test_corrupt_positions_rejected(self):
        for values in [[1., 2.], [-1, 2], [1, 1], [0, 4], [], [[1]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                v.indices(np.asarray(values), 'synthetic', 4)

    def test_overlap_and_incomplete_partition_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            v.partition(np.array([0, 1]), np.array([1, 2]), 3)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            v.partition(np.array([0]), np.array([2]), 3, full=True)


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / v.SUITE
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, prefix='verification-test-')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding='utf8')
        return path


class FileEvidenceTests(TemporaryFiles):
    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            v.safe_path(self.root, '../outside')

    def test_npz_rejects_object_without_pickle(self):
        path = self.root / 'unsafe.npz'
        np.savez(path, y=np.array(['a'], object), prediction=np.array(['a']))
        with self.assertRaises(ValueError):
            v.arrays(path)
        # Historical unnecessary object fields need not be deserialized.
        self.assertEqual(v.arrays(path, ['prediction'])['prediction'].tolist(), ['a'])

    def test_input_hashes_actual_bytes_and_aliases(self):
        aliases = {'train': 'train.csv', 'test': 'test.csv', 'sample': 'sample_submission.csv',
                   'dictionary': 'data_dictionary.csv'}
        hashes = {key: v.sha256(self.write(name, key)) for key, name in aliases.items()}
        self.assertEqual(v.verify_hashes(self.root, hashes), hashes)
        self.write('train.csv', 'mutated')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            v.verify_hashes(self.root, hashes)

    def test_missing_input_evidence_rejected(self):
        with self.assertRaises(ValueError):
            v.verify_hashes(self.root, {})

    def source_fixture(self):
        source = self.write('source/sprint_toy.py', 'answer = 1\n')
        hashes = {'sprint_toy.py': v.sha256(source)}
        self.write('frozen.json', json.dumps({'spec': 'SPEC011', 'sources': hashes,
                    'created': '2026-10-24T01:00:00+03:00'}))
        return {'sources': hashes, 'started': '2026-10-24T01:01:00+03:00'}

    def test_source_manifest_exact(self):
        self.assertEqual(v.verify_sources(self.root, self.source_fixture())['files'], 1)

    def test_source_byte_mutation_rejected(self):
        event = self.source_fixture()
        self.write('source/sprint_toy.py', 'answer = 2\n')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            v.verify_sources(self.root, event)

    def test_unmanifested_source_rejected(self):
        event = self.source_fixture()
        self.write('source/extra.py', 'pass\n')
        with self.assertRaisesRegex(ValueError, 'file set'):
            v.verify_sources(self.root, event)

    def test_manifest_evidence_disagreement_rejected(self):
        event = self.source_fixture()
        event['sources'] = {'sprint_toy.py': 'fake'}
        with self.assertRaisesRegex(ValueError, 'manifests differ'):
            v.verify_sources(self.root, event)

    def test_late_source_freeze_rejected(self):
        event = self.source_fixture()
        event['started'] = '2026-10-24T00:59:00+03:00'
        with self.assertRaisesRegex(ValueError, 'before attempt'):
            v.verify_sources(self.root, event)

    def configuration_fixture(self):
        self.write('source/experiments/sprint_worker.py', 'from sprint_toy import Toy\n')
        self.write('source/sprint_toy.py', '''class Toy:
    def __init__(self, alpha=.8): self.alpha = alpha
    def configuration(self): return {'family': 'toy', 'alpha': self.alpha}
    def fit(self, *args): raise AssertionError('Verifier must never fit')
''')
        return {'number': 999, 'parameters': {'alpha': .95}}

    def test_new_hypothesis_frozen_constructor_no_fit(self):
        request = self.configuration_fixture()
        result = v.expected_configuration(self.root, request, {'family': 'toy', 'alpha': .95,
            'teacher_classes': ['a'], 'teacher_train_sha256': 'synthetic'})
        self.assertEqual(result['constructor'], 'Toy')
        self.assertEqual(result['fits'], 0)
        self.assertFalse(list((self.root / 'source').rglob('__pycache__')))

    def test_parameter_tampering_rejected(self):
        request = self.configuration_fixture()
        with self.assertRaisesRegex(ValueError, 'frozen constructor'):
            v.expected_configuration(self.root, request, {'family': 'toy', 'alpha': .5})

    def test_same_constructor_imported_by_two_dispatch_branches_is_not_ambiguous(self):
        request = self.configuration_fixture()
        self.write('source/experiments/sprint_worker.py', 'from sprint_toy import Toy\nfrom sprint_toy import Toy\n')
        result = v.expected_configuration(self.root, request, {'family': 'toy', 'alpha': .95})
        self.assertEqual(result['constructor'], 'Toy')

    def test_runtime_num_class_derivation_is_traced_through_frozen_parent_class(self):
        self.write('source/experiments/sprint_worker.py', 'from sprint_toy import Toy\n')
        self.write('source/sprint_toy.py', '''class Base:
    def fit(self, X, y):
        self.parameters['num_class'] = len(self.classes_)
        raise AssertionError('No fits in verifier')
class Toy(Base):
    def __init__(self): self.parameters = {'iterations': 3}
    def configuration(self): return {'parameters': self.parameters}
''')
        result = v.expected_configuration(self.root, {'number': 999, 'parameters': {}},
                                           {'parameters': {'iterations': 3}})
        self.assertTrue(result['derived_num_class_equals_fitted_classes'])

    def test_one_bad_check_does_not_mask_subsequent_checks(self):
        checks = v.Checks()
        checks.run('bad', lambda: v.require(False, 'bad record'))
        checks.run('good', lambda: {'count': 1})
        self.assertEqual([r['passed'] for r in checks.rows], [False, True])

    def test_missing_run_artifacts_are_per_run_errors(self):
        report = v.verify_run(self.root, self.root / 'HYP-999',
            {'directory': 'missing', 'hypothesis': 'HYP-999', 'status': 'completed', 'stage': 'outer'}, {}, {})
        self.assertFalse(report['verified'])
        self.assertEqual(report['checks'][-1]['name'], 'required_artifact_or_structure')

    def test_frozen_time_orders_low_number_consumer_after_its_source(self):
        self.write('HYP-057/frozen.json', json.dumps({'created': '2026-10-24T02:58:00+03:00'}))
        self.write('HYP-070/frozen.json', json.dumps({'created': '2026-10-24T01:00:00+03:00'}))
        self.assertEqual([p.name for p in v.ordered_hypothesis_directories(self.root)], ['HYP-070', 'HYP-057'])

    def test_bad_manifest_does_not_prevent_other_runs_from_being_ordered(self):
        self.write('HYP-057/frozen.json', 'invalid')
        self.write('HYP-070/frozen.json', json.dumps({'created': '2026-10-24T01:00:00+03:00'}))
        self.assertEqual([p.name for p in v.ordered_hypothesis_directories(self.root)], ['HYP-070', 'HYP-057'])


class FingerprintTests(unittest.TestCase):
    def test_flat_audit_counter_can_override_protocol_description(self):
        constructor = {'audit_spreads_configuration': True, 'audit_explicit_fields': ['training_pairs']}
        config = {'alpha': .9, 'training_pairs': 'strictly_past'}
        audit = {'alpha': .9, 'training_pairs': 8544}
        v.audit_configuration(config, audit, constructor)
        audit['alpha'] = .8
        with self.assertRaisesRegex(ValueError, 'Inherited audit/configuration'):
            v.audit_configuration(config, audit, constructor)

    def test_nested_diagnostic_audit_not_misread_as_configuration(self):
        constructor = {'audit_spreads_configuration': False, 'audit_explicit_fields': ['decision']}
        config = {'decision': {'iterations': 25}}
        v.audit_configuration(config, {'decision': {'selected_iteration': 7}}, constructor)
        with self.assertRaisesRegex(ValueError, 'Audit configuration differs'):
            v.audit_configuration(config, {'configuration': {'decision': {'iterations': 26}}}, constructor)

    def test_semantic_recipe_excludes_seed_but_keeps_scientific_parameters(self):
        self.assertEqual(v.semantic({'random_state': 42, 'parameters': {'n_jobs': 4, 'depth': 5}}),
                         {'parameters': {'depth': 5}})

    def test_digest_sensitive_to_dtype_and_order(self):
        a = pd.DataFrame({'x': [1, 2]})
        self.assertNotEqual(v.data_digest(a), v.data_digest(a.astype(float)))
        self.assertNotEqual(v.data_digest(a), v.data_digest(a.iloc[::-1]))

    def test_cache_key_sensitive_to_global_positions_and_labels(self):
        X, y = pd.DataFrame({'x': [1, 2]}), np.array(['01', '1'])
        base = v.cache_key(X, y, np.array([4, 6]), {'policy': 'legacy'})
        self.assertNotEqual(base, v.cache_key(X, y[::-1], np.array([4, 6]), {'policy': 'legacy'}))
        self.assertNotEqual(base, v.cache_key(X, y, np.array([6, 4]), {'policy': 'legacy'}))

    def test_frozen_meta_stack_formula_and_class_order(self):
        classes = np.array(['a', 'b', 'c'])
        parts = [np.array([[.2, .3, .5]])] * 3
        state = {'classes': classes.tolist(), 'component_order': ['raw', 'smote', 'frequency'], 'clip': 1e-6,
                 'coef': np.zeros((3, 9)).tolist(), 'intercept': np.log([1., 2., 3.]).tolist()}
        np.testing.assert_allclose(v.stack_probability(parts, state, classes), [[1 / 6, 2 / 6, 3 / 6]])
        with self.assertRaisesRegex(ValueError, 'protocol'):
            v.stack_probability(parts, state, classes[::-1])


class TeacherCacheTests(TemporaryFiles):
    def cache_fixture(self):
        from sklearn.model_selection import StratifiedGroupKFold
        bundle = SimpleNamespace(X=pd.DataFrame({'x': np.arange(18)}),
                                 y=pd.Series(['a', 'b', 'c'] * 6))
        positions = np.arange(18)
        config = {'policy': 'legacy'}
        key = v.cache_key(bundle.X, bundle.y, positions, config)
        directory = self.root / 'results/2026-10-23/series-twenty-hypotheses-v1/cache' / key
        directory.mkdir(parents=True)
        (directory / 'metadata.json').write_text(json.dumps({'key': key, 'config': config}), encoding='utf8')
        groups = pd.util.hash_pandas_object(bundle.X, index=False).to_numpy()
        paths = []
        for i, (fit, valid) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(bundle.X, bundle.y, groups)):
            path = directory / f'fold-{i}.npz'
            np.savez(path, fit=fit, valid=valid, fit_positions=positions[fit], valid_positions=positions[valid],
                     classes=np.array(['a', 'b', 'c']), probabilities=np.full((len(valid), 3), 1 / 3))
            paths.append(str(path))
        return bundle, positions, {'key': key, 'config': config, 'model_fits': 0, 'artifacts': paths}

    def test_complete_teacher_oof_reconstructed(self):
        bundle, positions, audit = self.cache_fixture()
        _, classes, oof, folds = v.read_oof(self.root, bundle, positions, 'raw', audit)
        self.assertEqual(classes.tolist(), ['a', 'b', 'c'])
        self.assertEqual(len(folds), 3)
        np.testing.assert_allclose(oof, 1 / 3)

    def test_teacher_global_alignment_corruption_rejected(self):
        bundle, positions, audit = self.cache_fixture()
        path = Path(audit['artifacts'][0])
        saved = v.arrays(path)
        saved['valid_positions'] = saved['valid_positions'][::-1]
        np.savez(path, **saved)
        with self.assertRaisesRegex(ValueError, 'global position alignment'):
            v.read_oof(self.root, bundle, positions, 'raw', audit)

    def test_teacher_class_order_corruption_rejected(self):
        bundle, positions, audit = self.cache_fixture()
        path = Path(audit['artifacts'][0])
        saved = v.arrays(path)
        saved['classes'] = saved['classes'][::-1]
        np.savez(path, **saved)
        with self.assertRaisesRegex(ValueError, 'class order'):
            v.read_oof(self.root, bundle, positions, 'raw', audit)

    def test_teacher_audit_cannot_claim_fit(self):
        bundle, positions, audit = self.cache_fixture()
        audit['model_fits'] = 1
        with self.assertRaisesRegex(ValueError, 'audit mismatch'):
            v.read_oof(self.root, bundle, positions, 'raw', audit)


class CompleteRunTests(TemporaryFiles):
    def run_fixture(self):
        bundle = SimpleNamespace(X=pd.DataFrame({'Year_Factor': [1, 2, 3, 4]}),
                                 y=pd.Series([10., 20., 30., 40.]), id_column='ID', target_column='target')
        hypothesis = self.root / 'HYP-070'
        output = self.root / 'run'
        output.mkdir()
        worker = self.write('HYP-070/source/experiments/sprint_worker.py', 'from sprint_toy import TemporalEntityLinker\n')
        model = self.write('HYP-070/source/sprint_toy.py', '''class TemporalEntityLinker:
    def __init__(self, alpha=.8): self.alpha = alpha
    def configuration(self): return {'strategy': 'synthetic', 'alpha': self.alpha}
    def fit(self, *args): raise AssertionError('No fit allowed')
''')
        sources = {'experiments/sprint_worker.py': v.sha256(worker), 'sprint_toy.py': v.sha256(model)}
        parameters = {'alpha': .95}
        config = {'strategy': 'synthetic', **parameters}
        fit, valid = np.array([0, 1, 2]), np.array([3])
        identity = {'recipe': v.semantic(config), 'X_sha256': v.data_digest(bundle.X.iloc[fit]),
                    'y_sha256': v.data_digest(bundle.y.to_numpy()[fit].reshape(-1, 1))}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        result = {'status': 'completed', 'metric': 'rmse', 'score': 1., 'reference_score': 2., 'relative_gain': .5,
                  'fingerprint': fingerprint, 'fit_rows': 3, 'valid_rows': 1, 'old_baseline_refits': 0}
        event = {**result, 'hypothesis': 'HYP-070', 'stage': 'inner', 'variant': 'default',
                 'parameters': parameters, 'sources': sources, 'directory': 'run',
                 'started': '2026-10-24T01:01:00+03:00', 'finished': '2026-10-24T01:02:00+03:00',
                 'ledger_sha256': 'recorded-ledger-read'}
        request = {'number': 70, 'stage': 'inner', 'parameters': parameters, 'repo_root': str(self.root),
                   'output': str(output), 'history': str(hypothesis / 'model-fit-history.json')}
        claim = {**identity, **result, 'hypothesis': 'HYP-070', 'stage': 'inner',
                 'started': '2026-10-24T01:01:01+03:00', 'finished': '2026-10-24T01:01:59+03:00',
                 'ledger_sha256': 'recorded-ledger-read'}
        self.write('HYP-070/frozen.json', json.dumps({'created': '2026-10-24T01:00:00+03:00', 'spec': 'SPEC011',
                    'sources': sources, 'variants': {'default': parameters}}))
        self.write('HYP-070/model-fit-history.json', json.dumps([claim]))
        hashes = {}
        for name in ['train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv']:
            hashes[name] = v.sha256(self.write('data/regression/' + name, name))
        metadata = {'evidence.json': event, 'request.json': request, 'metrics.json': result,
                    'configuration.json': config, 'data.json': {'hashes': hashes, 'id_excluded_from_X': True,
                        'target_excluded_from_X': True}, 'audit.json': {'strategy': 'synthetic', 'fit_rows': 3},
                    'teacher-audit.json': {'baseline_source': str(self.root / v.REG_INNER), 'old_model_refits': 0}}
        for name, value in metadata.items():
            self.write('run/' + name, json.dumps(value))
        np.savez(output / 'splits.npz', fit=fit, valid=valid)
        np.savez(output / 'validation_predictions.npz', positions=valid, y=np.array([40.]), prediction=np.array([39.]), reference=np.array([38.]))
        baseline = self.root / v.REG_INNER
        baseline.mkdir(parents=True)
        (baseline / 'evidence.json').write_text(json.dumps({'data': {'hashes': hashes}}), encoding='utf8')
        np.savez(baseline / 'splits.npz', fit=fit, valid=valid)
        np.savez(baseline / 'validation_predictions.npz', positions=valid, prediction=np.array([38.]))
        return bundle, hypothesis, output, event

    def test_complete_synthetic_run_verified_without_fit(self):
        bundle, hypothesis, _, event = self.run_fixture()
        with patch('research_data.load_task', return_value=bundle):
            report = v.verify_run(self.root, hypothesis, event, {}, {})
        self.assertTrue(report['verified'], report['checks'])

    def test_changed_validation_target_does_not_pass(self):
        bundle, hypothesis, output, event = self.run_fixture()
        path = output / 'validation_predictions.npz'
        saved = v.arrays(path)
        saved['y'] = np.array([41.])
        np.savez(path, **saved)
        with patch('research_data.load_task', return_value=bundle):
            report = v.verify_run(self.root, hypothesis, event, {}, {})
        self.assertFalse(report['verified'])
        self.assertTrue(any(c['name'] == 'position_and_target_alignment' and not c['passed'] for c in report['checks']))

    def test_changed_saved_reference_detected_even_if_score_self_consistent(self):
        bundle, hypothesis, output, event = self.run_fixture()
        path = output / 'validation_predictions.npz'
        saved = v.arrays(path)
        saved['reference'] = np.array([42.])  # Same RMSE=2, different historical prediction.
        np.savez(path, **saved)
        with patch('research_data.load_task', return_value=bundle):
            report = v.verify_run(self.root, hypothesis, event, {}, {})
        self.assertFalse(report['verified'])
        self.assertTrue(any(c['name'] == 'compatible_saved_regression_reference' and not c['passed'] for c in report['checks']))


class OverlayProvenanceTests(TemporaryFiles):
    def overlay_fixture(self):
        source = self.root / 'results/2026-10-24/HYP-070-03-inner-high-sprint'
        source.mkdir(parents=True)
        fit, valid = np.array([0, 1]), np.array([2, 3])
        inherited = {'positions': valid, 'y': np.array([10., 20.]), 'prediction': np.array([9., 19.])}
        state = {'rating_beta': .25, 'group_rating_betas': {'office': .3}}
        (source / 'audit.json').write_text(json.dumps(state), encoding='utf8')
        np.savez(source / 'validation_predictions.npz', **inherited)
        np.savez(source / 'splits.npz', fit=fit, valid=valid)
        extra = {'transport_state_sha256': v.sha256(source / 'audit.json'),
                 'inherited_prediction_sha256': hashlib.sha256(inherited['prediction'].tobytes()).hexdigest()}
        audit = {**state, 'transport_state_fitted_here': False, 'transport_state_supplied': True, 'base_refit': False}
        teacher = {**extra, 'inherited_baseline_source': str(source)}
        saved = {'y': inherited['y'], 'prediction': np.array([8., 18.])}
        prior = [{'directory': str(source.relative_to(self.root)), 'verified': True}]
        return fit, valid, saved, extra, audit, teacher, prior

    def test_inherited_hashes_checked_and_negative_increment_reported(self):
        result = v.inherited_overlay(self.root, 'inner', *self.overlay_fixture())
        self.assertFalse(result['incremental_improvement_over_H70'])
        self.assertEqual(result['incremental_H70_control_score'], 1.)
        self.assertEqual(result['overlay_score'], 2.)
        self.assertEqual(set(result['verified_extra_fields']), {'transport_state_sha256', 'inherited_prediction_sha256'})

    def test_inherited_prediction_sha_tampering_rejected(self):
        fit, valid, saved, config, audit, teacher, prior = self.overlay_fixture()
        config['inherited_prediction_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'source SHA mismatch'):
            v.inherited_overlay(self.root, 'inner', fit, valid, saved, config, audit, teacher, prior)

    def test_refitted_transport_state_rejected(self):
        fit, valid, saved, config, audit, teacher, prior = self.overlay_fixture()
        audit['transport_state_fitted_here'] = True
        with self.assertRaisesRegex(ValueError, 'transport state'):
            v.inherited_overlay(self.root, 'inner', fit, valid, saved, config, audit, teacher, prior)

    def test_source_must_itself_be_verified(self):
        fit, valid, saved, config, audit, teacher, prior = self.overlay_fixture()
        prior[0]['verified'] = False
        with self.assertRaisesRegex(ValueError, 'not independently verified'):
            v.inherited_overlay(self.root, 'inner', fit, valid, saved, config, audit, teacher, prior)

    def test_prediction_only_expert_hashes_h70_without_claiming_beta_use(self):
        fit, valid, saved, config, audit, teacher, prior = self.overlay_fixture()
        del audit['rating_beta'], audit['group_rating_betas']
        result = v.inherited_overlay(self.root, 'inner', fit, valid, saved, config, audit, teacher, prior, use_transport=False)
        self.assertFalse(result['transport_state_used'])


class CalibrationStateTests(TemporaryFiles):
    def fixture(self, query_labels=False):
        source = self.write('results/2026-10-24/HYP-078-01-inner-default-sprint/audit.json',
                            json.dumps({'theta': [-.1, .2, .3]}))
        self.write('snapshot/source/sprint_cohort.py', '''class CohortMacroDecision:
    def predict(self, X, %s): return X
    def predict_proba(self, X, teacher_probabilities): return teacher_probabilities
''' % ('y' if query_labels else 'teacher_probabilities'))
        config = {'calibration_state': {'theta': [-.1, .2, .3]}, 'query_labels_used': False}
        audit = {'learner_fits': 0, 'calibration_refitted': False, 'query_target_used': False}
        teacher = {'calibration_state_source': str(source), 'calibration_state_sha256': v.sha256(source)}
        prior = [{'directory': str(source.parent.relative_to(self.root)), 'verified': True}]
        return config, audit, teacher, prior

    def test_actual_theta_hash_and_label_free_query_validated(self):
        config, audit, teacher, prior = self.fixture()
        result = v.calibration_provenance(self.root, 'inner', config, audit, teacher, np.array(['a', 'b']), self.root / 'snapshot', prior)
        self.assertEqual(result['constructor_parameters'], {'calibration_state': {'theta': [-.1, .2, .3]}})
        self.assertFalse(result['batch_invariant'])

    def test_changed_theta_cannot_be_injected_as_trusted_constructor_state(self):
        config, audit, teacher, prior = self.fixture()
        config['calibration_state']['theta'][0] = -.2
        with self.assertRaisesRegex(ValueError, 'theta differs'):
            v.calibration_provenance(self.root, 'inner', config, audit, teacher, np.array(['a', 'b']), self.root / 'snapshot', prior)

    def test_composed_h78_state_requires_exact_gain_offset_and_theta(self):
        source = self.write('results/2026-10-24/HYP-078-01-inner-default-sprint/audit.json',
                            json.dumps({'theta': [0., 0., 0.], 'decision': {'gain': [2., 3.], 'offset': [-.1, -.2]}}))
        state = {'classes': ['a', 'b'], 'theta': [0., 0., 0.], 'gain': [2., 3.], 'offset': [-.1, -.2]}
        config = {'input_state': state, 'input_state_refitted': False, 'training_meta_is_independent_OOF': False}
        teacher = {'input_state_source': str(source), 'input_state_sha256': v.sha256(source)}
        prior = [{'directory': str(source.parent.relative_to(self.root)), 'verified': True}]
        result = v.composed_state_provenance(self.root, 'inner', config, teacher, np.array(['a', 'b']), prior)
        self.assertEqual(result['constructor_parameters']['state'], state)
        config['input_state']['gain'][1] = 4.
        with self.assertRaisesRegex(ValueError, 'input state differs'):
            v.composed_state_provenance(self.root, 'inner', config, teacher, np.array(['a', 'b']), prior)

    def test_label_accepting_query_rejected_despite_false_audit_flag(self):
        config, audit, teacher, prior = self.fixture(query_labels=True)
        with self.assertRaisesRegex(ValueError, 'exposes labels'):
            v.calibration_provenance(self.root, 'inner', config, audit, teacher, np.array(['a', 'b']), self.root / 'snapshot', prior)


class ActualFitAuditTests(unittest.TestCase):
    def test_runtime_class_count_accepted_only_with_frozen_derivation_and_correct_count(self):
        constructor = {'audit_explicit_fields': [], 'audit_spreads_configuration': False,
                       'derived_num_class_equals_fitted_classes': True}
        config = {'parameters': {'iterations': 300}, 'teacher_classes': ['a', 'b']}
        audit = {'configuration': {'parameters': {'iterations': 300, 'num_class': 2}}}
        result = v.audit_configuration(config, audit, constructor)
        self.assertEqual(result['runtime_num_class_independently_derived'], 2)
        audit['configuration']['parameters']['num_class'] = 3
        with self.assertRaisesRegex(ValueError, 'Runtime num_class differs'):
            v.audit_configuration(config, audit, constructor)

    def test_native_feature_preparation_imputes_train_medians_and_keeps_categories(self):
        X = pd.DataFrame({'facility_type': ['b', None, 'a'], 'energy_star_rating': [0., 2., 4.],
                          'floor_area': [1., np.nan, 5.]})
        native, categories = v.expert_frame(X, relative=False, native=True)
        self.assertEqual(list(native.columns), ['energy_star_rating', 'floor_area', 'facility_type'])
        self.assertEqual(native['floor_area'].tolist(), [1., 3., 5.])
        self.assertEqual(list(native['facility_type'].cat.categories), ['a', 'b'])
        self.assertTrue(pd.isna(native['facility_type'].iloc[1]))
        self.assertEqual(categories, ['facility_type'])
    def test_actual_claim_hash_depends_on_raw_X_y_and_configuration(self):
        X, y, config = pd.DataFrame({'x': [1., 2.]}), np.array([3., 4.]), {'depth': 6}
        identity = {'recipe': config, 'X_sha256': v.data_digest(X), 'y_sha256': v.data_digest(y.reshape(-1, 1))}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        claim = {**identity, 'fingerprint': fingerprint, 'stage': 'inner', 'status': 'completed', 'ledger_sha256': 'recorded'}
        self.assertEqual(v.verify_claim(claim, config, X, y, 'inner'), fingerprint)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            v.verify_claim(claim, config, X, y[::-1], 'inner')

    def test_expert_preparation_manual_wind_relative_and_tagged_categories(self):
        X = pd.DataFrame({'direction_max_wind_speed': [0., 90., 180., 270.],
                          'floor_area': [1., 3., 5., 7.], 'energy_star_rating': [np.nan] * 4,
                          'State_Factor': ['s'] * 4, 'facility_type': ['office'] * 4})
        frame, categories = v.expert_frame(X)
        np.testing.assert_allclose(frame['direction_max_wind_speed__sin'], [0., 1., 0., -1.], atol=1e-15)
        np.testing.assert_allclose(frame['floor_area__group_relative'], [-1., -1 / 3, 1 / 3, 1.])
        self.assertEqual(frame['facility_type'].tolist(), ['V:office'] * 4)
        self.assertEqual(categories, ['State_Factor', 'facility_type'])
        self.assertTrue(frame['energy_star_rating__group_relative'].isna().all())


class MissingExpertAuditTests(TemporaryFiles):
    def fixture(self):
        from threadpoolctl import threadpool_limits
        X = pd.DataFrame({'energy_star_rating': [np.nan, 0.] * 300 + [0., np.nan, np.inf, 10.],
                          'floor_area': np.arange(604, dtype=float), 'State_Factor': ['s'] * 604,
                          'facility_type': ['office'] * 604})
        y = np.arange(1, 605, dtype=float)
        bundle, fit, valid = SimpleNamespace(X=X, y=pd.Series(y)), np.arange(600), np.arange(600, 604)
        local = np.arange(0, 600, 2)
        with threadpool_limits(limits=4):
            frame, categories = v.expert_frame(X.iloc[local])
        configuration = {'strategy': 'synthetic_missing_expert'}
        claim_config = {**configuration, 'stage': 'missing_rating_actual_learner',
                        'training_subset_rows': 300, 'feature_columns': list(frame.columns)}
        identity = {'recipe': v.semantic(claim_config), 'X_sha256': v.data_digest(frame),
                    'y_sha256': v.data_digest(y[local].reshape(-1, 1))}
        record = {**identity, 'fingerprint': hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest(),
                  'stage': 'inner', 'status': 'completed', 'ledger_sha256': 'recorded'}
        self.write('run/expert-fits.json', json.dumps([record]))
        self.write('run/prediction-audit.json', json.dumps({'query_missing_rows': 2, 'expert_applied_rows': 2, 'known_rating_unchanged': True}))
        self.write('snapshot/source/experiments/sprint_run.py', "env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')\n")
        mask = np.arange(600) % 2 == 0
        audit = {'expert_fit_mask': mask.tolist(), 'expert_fit_positions': local.tolist(),
                 'expert_fit_original_indices': local.tolist(), 'expert_fit_rows': 300, 'removed_rows': 0,
                 'expert_fit_count': 1, 'base_refit': False, 'categorical_columns': categories,
                 'feature_audit': {'numbers': [29, 30, 31], 'fit_rows': 300, 'target_used': False}}
        return bundle, fit, valid, {'prediction': np.array([1., 9., 9., 4.])}, configuration, audit, self.root / 'run', 'inner', np.array([1., 2., 3., 4.]), self.root / 'snapshot'

    def test_actual_missing_subset_learner_claim_and_observed_zero_gate(self):
        result = v.missing_expert_provenance(*self.fixture())
        self.assertEqual(result['actual_subset_rows'], 300)
        self.assertFalse(result['H70_transport_beta_used'])
        self.assertEqual(result['feature_reconstruction_BLAS_threads'], 4)

    def test_mask_tampering_rejected_even_with_valid_learner_claim(self):
        arguments = list(self.fixture())
        arguments[5]['expert_fit_mask'][0] = False
        with self.assertRaisesRegex(ValueError, 'subset/mask/indices'):
            v.missing_expert_provenance(*arguments)

    def test_known_rating_prediction_cannot_change(self):
        arguments = list(self.fixture())
        arguments[3]['prediction'][0] = 7.
        with self.assertRaisesRegex(ValueError, 'Known-rating rows changed'):
            v.missing_expert_provenance(*arguments)


class MonotoneExpertAuditTests(TemporaryFiles):
    def fixture(self):
        X = pd.DataFrame({'energy_star_rating': [0.] * 300 + [np.nan] * 300 + [0., 20., np.nan, 20.],
                          'floor_area': list(np.arange(1., 601.)) + [1., 10000., 10001., 2.],
                          'Year_Factor': [1.] * 600 + [2., 2., 2., 0.],
                          'State_Factor': ['s'] * 604, 'building_class': ['b'] * 604,
                          'facility_type': ['office'] * 604, 'year_built': [2000.] * 604})
        y, local = np.arange(604, dtype=float), np.arange(300)
        native, categorical = v.expert_frame(X.iloc[local], relative=False, native=True)
        config = {'strategy': 'synthetic_monotone'}
        constraints = [-1 if c == 'energy_star_rating' else 0 for c in native]
        actual = {**config, 'stage': 'known_rating_actual_monotone_learner', 'training_subset_rows': 300,
                  'feature_columns': list(native), 'monotone_constraints': constraints}
        identity = {'recipe': v.semantic(actual), 'X_sha256': v.data_digest(native),
                    'y_sha256': v.data_digest(y[local].reshape(-1, 1))}
        claim = {**identity, 'fingerprint': hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest(),
                 'stage': 'inner', 'status': 'completed', 'ledger_sha256': 'recorded'}
        self.write('run/expert-fits.json', json.dumps([claim]))
        self.write('run/prediction-audit.json', json.dumps({'query_known_rating_rows': 3, 'query_matched_past_rows': 1,
                   'eligible_unmatched_known_rows': 2, 'expert_applied_rows': 2,
                   'matched_or_missing_unchanged': True, 'query_statistics': False}))
        self.write('snapshot/source/experiments/sprint_run.py', "env.update(OPENBLAS_NUM_THREADS='4')\n")
        audit = {'expert_fit_mask': (np.arange(600) < 300).tolist(), 'expert_fit_positions': local.tolist(),
                 'expert_fit_original_indices': local.tolist(), 'expert_fit_rows': 300, 'removed_rows': 0,
                 'expert_fit_count': 1, 'native_feature_columns': list(native), 'categorical_columns': categorical,
                 'monotone_constraints': constraints, 'feature_audit': {'numbers': [29, 30], 'fit_rows': 300, 'target_used': False}}
        return SimpleNamespace(X=X, y=pd.Series(y)), np.arange(600), np.arange(600, 604), {'prediction': np.array([1., 9., 3., 10.])}, config, audit, self.root / 'run', 'inner', np.array([1., 2., 3., 4.]), self.root / 'snapshot'

    def test_zero_rating_is_known_and_only_strictly_past_profile_is_matched(self):
        result = v.monotone_expert_provenance(*self.fixture())
        self.assertEqual(result['actual_known_subset_rows'], 300)
        self.assertEqual(result['preserved_H70_rows'], 2)
        self.assertFalse(result['H70_transport_beta_used'])

    def test_claimed_wrong_monotonicity_is_rejected(self):
        args = list(self.fixture())
        args[5]['monotone_constraints'] = [0] * len(args[5]['monotone_constraints'])
        with self.assertRaisesRegex(ValueError, 'constraint audit'):
            v.monotone_expert_provenance(*args)

    def test_matched_profile_cannot_be_overwritten_by_expert(self):
        args = list(self.fixture())
        args[3]['prediction'][0] = 9.
        with self.assertRaisesRegex(ValueError, 'outside fixed expert gate'):
            v.monotone_expert_provenance(*args)

    def test_no_expert_fit_requires_all_predictions_to_remain_base(self):
        args = list(self.fixture())
        args[0].X.loc[:299, 'energy_star_rating'] = np.nan
        args[5].update(expert_fit_mask=[False] * 600, expert_fit_positions=[], expert_fit_original_indices=[],
                       expert_fit_rows=0, expert_fit_count=0)
        state = v.read_json(self.root / 'run/prediction-audit.json')
        state['expert_applied_rows'] = 0
        self.write('run/prediction-audit.json', json.dumps(state))
        self.write('run/expert-fits.json', '[]')
        with self.assertRaisesRegex(ValueError, 'without an actual expert fit'):
            v.monotone_expert_provenance(*args)


class AssignmentAuditTests(TemporaryFiles):
    def fixture(self):
        X = pd.DataFrame({'Year_Factor': [1., 2., 1., 3., 3.], 'floor_area': [1., 2., 3., 4., 5.],
                          'State_Factor': ['s'] * 5, 'building_class': ['b'] * 5,
                          'facility_type': ['office'] * 5, 'year_built': [2000.] * 5,
                          'energy_star_rating': [np.nan] * 5})
        self.write('run/prediction-audit.json', json.dumps({'matching_target_used': False, 'query_rows_added_to_memory': False,
                   'donor_positions': [1, None], 'donor_years': [2., None], 'assigned_rows': 1}))
        return (SimpleNamespace(X=X, y=pd.Series([100., 200., 300., 0., 0.])), np.arange(3), np.arange(3, 5),
                {'prediction': np.array([125., 60.])}, {'alpha': .5, 'age_decay': .25},
                {'matching_target_used': False, 'validation_target_used': False, 'rating_beta': 1., 'group_rating_betas': {}},
                self.root / 'run', np.array([50., 60.]))

    def test_training_donor_blend_manual_arithmetic_and_unassigned_preservation(self):
        result = v.assignment_provenance(*self.fixture())
        self.assertEqual(result['strictly_past_training_donors'], 1)
        self.assertFalse(result['physical_identity_proven'])

    def test_same_donor_profile_cannot_be_reused_within_query_year(self):
        args = list(self.fixture())
        args[3]['prediction'][1] = 130.
        state = v.read_json(self.root / 'run/prediction-audit.json')
        state.update(donor_positions=[1, 1], donor_years=[2., 2.], assigned_rows=2)
        self.write('run/prediction-audit.json', json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'profile reused'):
            v.assignment_provenance(*args)

    def test_nonpast_donor_rejected(self):
        args = list(self.fixture())
        args[0].X.loc[3, 'Year_Factor'] = 2.
        with self.assertRaisesRegex(ValueError, 'non-past donor'):
            v.assignment_provenance(*args)

    def test_zero_assignment_is_valid_and_preserves_all_base_predictions(self):
        args = list(self.fixture())
        args[3]['prediction'] = np.array([50., 60.])
        self.write('run/prediction-audit.json', json.dumps({'matching_target_used': False, 'query_rows_added_to_memory': False,
                   'donor_positions': [None, None], 'donor_years': [None, None], 'assigned_rows': 0}))
        self.assertEqual(v.assignment_provenance(*args)['strictly_past_training_donors'], 0)


class EmpiricalDecisionAuditTests(unittest.TestCase):
    def fixture(self):
        return [np.array([[.8, .2], [.4, .6]]), np.array([[.5, .5]]), np.array(['a', 'b']), np.array(['a', 'b']),
                {'offset': [0., 0.], 'classes': ['a', 'b'], 'initial_training_macro_f1': 1., 'selected_training_macro_f1': 1.,
                 'training_macro_f1_history': [1.], 'training_macro_f1_is_independent_evaluation': False, 'query_statistics': False},
                {'probabilities': np.array([[.5, .5]])}, {}]

    def test_manual_equal_query_and_training_objective_scope(self):
        result = v.empirical_decision_arithmetic(*self.fixture())
        self.assertTrue(result['training_objective_checked_only_not_independent_evaluation'])
        self.assertEqual(result['parameter_refits_performed_by_verifier'], 0)

    def test_saved_probability_corruption_is_rejected(self):
        args = self.fixture()
        args[5]['probabilities'][0] = [.6, .4]
        with self.assertRaisesRegex(ValueError, 'query probability arithmetic'):
            v.empirical_decision_arithmetic(*args)

    def test_training_metric_cannot_be_labeled_independent(self):
        args = self.fixture()
        args[4]['training_macro_f1_is_independent_evaluation'] = True
        with self.assertRaisesRegex(ValueError, 'history/scope'):
            v.empirical_decision_arithmetic(*args)


class AmplitudeTests(unittest.TestCase):
    def test_known_interior_least_squares_parameter(self):
        self.assertEqual(v.amplitude_parameter([12., 24.], [10., 20.], [13., 26.]), 1.5)

    def test_bounds_and_zero_delta_are_fixed(self):
        self.assertEqual(v.amplitude_parameter([12., 24.], [10., 20.], [20., 40.]), 2.)
        self.assertEqual(v.amplitude_parameter([12., 24.], [10., 20.], [9., 18.]), 0.)
        self.assertEqual(v.amplitude_parameter([10., 20.], [10., 20.], [9., 18.]), 1.)

    def test_nonfinite_or_unaligned_vectors_rejected(self):
        with self.assertRaises(ValueError):
            v.amplitude_parameter([np.nan], [1.], [2.])
        with self.assertRaises(ValueError):
            v.amplitude_parameter([1., 2.], [1.], [2.])


class HonestTeacherAuditTests(TemporaryFiles):
    def fixture(self):
        from sklearn.model_selection import StratifiedGroupKFold
        X, y = pd.DataFrame({'x': np.arange(24)}), pd.Series(['a', 'b', 'c'] * 8)
        bundle = SimpleNamespace(X=X, y=y, hashes={'train.csv': 'synthetic-input-sha'})
        outer_fit, outer_valid = np.arange(18), np.arange(18, 24)
        context = self.root / v.CLASS_OUTER
        context.mkdir(parents=True)
        np.savez(context / 'splits.npz', fit=outer_fit, valid=outer_valid)
        classes = np.array(['a', 'b', 'c'])
        raw_groups = pd.util.hash_pandas_object(X.iloc[outer_fit], index=False).to_numpy()
        original_folds = []
        raw_folder = self.root / 'raw'
        raw_folder.mkdir()
        for number, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(X.iloc[outer_fit], y.iloc[outer_fit], raw_groups)):
            saved = {'fit_positions': outer_fit[a], 'valid_positions': outer_fit[b],
                     'probabilities': np.tile([.2, .3, .5], (len(b), 1))}
            original_folds.append(saved)
            np.savez(raw_folder / f'fold-{number}.npz', **saved)
        fit, valid = original_folds[0]['fit_positions'], original_folds[0]['valid_positions']
        out = self.root / 'run'
        folder = out / 'honest-raw-oof'
        folder.mkdir(parents=True)
        names = ['automl_class_combo.py', 'automl_models.py', 'automl_combo_memory.py',
                 'sprint_contextual.py', 'sprint_honest.py', 'sprint_crossfit_cache.py']
        hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
        snapshot = self.root / 'snapshot'
        self.write('snapshot/frozen.json', json.dumps({'sources': hashes}))
        groups = pd.util.hash_pandas_object(X.iloc[fit], index=False).to_numpy()
        events, records = [], []
        phash = lambda values: hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()
        for number, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(X.iloc[fit], y.iloc[fit], groups)):
            config = {'parameters': {'n_estimators': 600, 'max_depth': 6, 'learning_rate': .05, 'reg_lambda': 1,
                                     'tree_method': 'hist', 'device': 'cpu', 'objective': 'multi:softprob'},
                      'component': 'raw', 'encoder': {'native': False, 'scale': False}, 'parameters_override': {},
                      'threads': 2, 'nested_fold': number, 'stage': 'inner', 'fit_positions': fit[a].tolist(),
                      'valid_positions': fit[b].tolist(), 'fit_positions_sha256': phash(fit[a]), 'valid_positions_sha256': phash(fit[b]),
                      'data_hashes': bundle.hashes, 'source_hashes': hashes, 'X_sha256': v.data_digest(X.iloc[fit[a]]),
                      'y_sha256': v.data_digest(y.to_numpy()[fit[a]].reshape(-1, 1))}
            saved = {'fit': a, 'valid': b, 'fit_positions': fit[a], 'valid_positions': fit[b],
                     'classes': classes, 'probabilities': np.tile([.1, .2, .7], (len(b), 1))}
            artifact, start, outcome = [folder / f'fold-{number}{suffix}' for suffix in ['.npz', '.started.json', '.outcome.json']]
            np.savez(artifact, **saved)
            np.savez(folder / f'fold-{number}.splits.npz', **{k: saved[k] for k in ['fit', 'valid', 'fit_positions', 'valid_positions']})
            start.write_text(json.dumps({'status': 'started', 'classes': classes.tolist(), 'configuration': config,
                                        'started': '2026-10-24T01:00:00+03:00'}), encoding='utf8')
            outcome.write_text(json.dumps({'status': 'completed', 'prediction_sha256': v.sha256(artifact), 'fit_rows': len(a),
                                          'valid_rows': len(b), 'finished': '2026-10-24T01:03:00+03:00'}), encoding='utf8')
            events.append({'fold': number, 'started_evidence': str(start), 'outcome': str(outcome), 'artifact': str(artifact),
                           'prediction_sha256': v.sha256(artifact), 'fit_rows': len(a), 'valid_rows': len(b), 'reused': False})
            identity = {'recipe': v.semantic(config), 'X_sha256': config['X_sha256'], 'y_sha256': config['y_sha256']}
            records.append({**identity, 'fingerprint': hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest(),
                            'stage': 'inner', 'status': 'completed', 'ledger_sha256': 'recorded',
                            'started': '2026-10-24T01:01:00+03:00', 'finished': '2026-10-24T01:02:00+03:00'})
        self.write('run/teacher-fits.json', json.dumps(records))
        train = np.tile([.1, .2, .7], (len(fit), 1))
        # Match the public provider's final normalization boundary.
        train = train / train.sum(axis=1, keepdims=True)
        audit = {'teacher': 'raw_xgboost_grouped_oof', 'protocol': 'H88_honest_raw_context', 'independent_training_oof': True,
                 'H48_meta_state_used': False, 'old_model_refits': 0, 'raw_component': {}, 'fit_positions_sha256': phash(fit),
                 'valid_positions_sha256': phash(valid), 'query_artifact': str(raw_folder / 'fold-0.npz'),
                 'query_artifact_sha256': v.sha256(raw_folder / 'fold-0.npz'), 'nested_folds': events,
                 'new_teacher_fits': 3, 'model_fits': 3, 'teacher_train_sha256': hashlib.sha256(train.tobytes()).hexdigest()}
        cached = (raw_folder, classes, np.tile([.2, .3, .5], (18, 1)), original_folds)
        return bundle, fit, valid, audit, snapshot, out, cached

    def test_nested_teacher_independent_full_coverage_and_three_actual_claims(self):
        bundle, fit, valid, audit, snapshot, out, cached = self.fixture()
        with patch.object(v, 'read_oof', return_value=cached):
            result = v.honest_teacher_provenance(self.root, bundle, 'inner', fit, valid, audit, snapshot, out)
        np.testing.assert_allclose(result['train'], np.tile([.1, .2, .7], (len(fit), 1)))
        self.assertEqual(result['detail']['new_teacher_fits'], 3)

    def test_unregistered_nested_fit_is_rejected(self):
        bundle, fit, valid, audit, snapshot, out, cached = self.fixture()
        self.write('run/teacher-fits.json', '[]')
        with patch.object(v, 'read_oof', return_value=cached), self.assertRaisesRegex(ValueError, 'separately registered'):
            v.honest_teacher_provenance(self.root, bundle, 'inner', fit, valid, audit, snapshot, out)

    def test_source_sha_cannot_be_blindly_accepted(self):
        bundle, fit, valid, audit, snapshot, out, cached = self.fixture()
        path = Path(audit['nested_folds'][0]['started_evidence'])
        started = v.read_json(path)
        started['configuration']['source_hashes']['sprint_honest.py'] = 'changed'
        path.write_text(json.dumps(started), encoding='utf8')
        with patch.object(v, 'read_oof', return_value=cached), self.assertRaisesRegex(ValueError, 'scientific source SHA'):
            v.honest_teacher_provenance(self.root, bundle, 'inner', fit, valid, audit, snapshot, out)

    def reused_fixture(self):
        bundle, fit, valid, audit, snapshot, out, cached = self.fixture()
        historical = self.root / 'results/2026-10-24/HYP-088-01-inner-default-sprint'
        historical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out, historical)
        frozen = self.root / v.SUITE / 'HYP-088'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, frozen)
        for event in audit['nested_folds']:
            event['reused'] = True
            for key in ['started_evidence', 'outcome', 'artifact']:
                event[key] = str(historical / 'honest-raw-oof' / Path(event[key]).name)
        audit.update(protocol='H92_94_reused_honest_raw_teacher', reused_H88_teachers=True,
                     new_teacher_fits=0, model_fits=0,
                     historical_frozen_manifest=str(frozen / 'frozen.json'),
                     historical_frozen_manifest_sha256=v.sha256(frozen / 'frozen.json'))
        return bundle, fit, valid, audit, self.root / 'new-current-snapshot', self.root / 'consumer', cached

    def test_reuse_validates_historical_h88_sources_and_claims_without_new_fits(self):
        bundle, fit, valid, audit, snapshot, out, cached = self.reused_fixture()
        with patch.object(v, 'read_oof', return_value=cached):
            result = v.honest_teacher_provenance(self.root, bundle, 'inner', fit, valid, audit, snapshot, out)
        self.assertEqual(result['detail']['new_teacher_fits'], 0)
        self.assertTrue(result['detail']['reused_existing_H88_teachers'])
        self.assertFalse(out.exists())

    def test_reused_old_manifest_sha_cannot_be_replaced_by_current_sources(self):
        bundle, fit, valid, audit, snapshot, out, cached = self.reused_fixture()
        audit['historical_frozen_manifest_sha256'] = 'bad'
        with patch.object(v, 'read_oof', return_value=cached), self.assertRaisesRegex(ValueError, 'Historical H88 frozen manifest'):
            v.honest_teacher_provenance(self.root, bundle, 'inner', fit, valid, audit, snapshot, out)


class SimplexVerificationTests(unittest.TestCase):
    def test_active_face_solver_recovers_known_mixture(self):
        mse, weights = v.simplex_optimum([[0., 2., 3.], [2., 0., 3.]], [1., 1.])
        np.testing.assert_allclose(weights, [.5, .5, 0.], atol=1e-12)
        self.assertLess(mse, 1e-20)

    def test_boundary_optimum_keeps_best_vertex(self):
        mse, weights = v.simplex_optimum([[0., 1., 2.], [0., 1., 2.]], [-1., -1.])
        np.testing.assert_allclose(weights, [1., 0., 0.], atol=1e-12)
        self.assertAlmostEqual(mse, 1.)

    def test_invalid_shape_or_nonfinite_simplex_input_rejected(self):
        with self.assertRaises(ValueError):
            v.simplex_optimum([[np.nan, 1.]], [1.])
        with self.assertRaises(ValueError):
            v.simplex_optimum([[1., 2.]], [1., 2.])


class TargetTests(unittest.TestCase):
    controls = {'rmse': 40., 'macro_f1': .6}

    @staticmethod
    def run_record(name, score, stage='outer', verified=True):
        return {'directory': name + '-' + stage, 'metric': name, 'score': score, 'stage': stage, 'verified': verified}

    def test_both_original_outer_targets_required(self):
        result = v.target_summary([self.run_record('rmse', 32.), self.run_record('macro_f1', .72)], self.controls)
        self.assertTrue(result['both_20pct_targets_met'])

    def test_one_target_insufficient(self):
        result = v.target_summary([self.run_record('rmse', 31.), self.run_record('macro_f1', .71)], self.controls)
        self.assertFalse(result['both_20pct_targets_met'])

    def test_inner_or_invalid_run_cannot_satisfy_target(self):
        records = [self.run_record('rmse', 1., 'inner'), self.run_record('macro_f1', 1., verified=False)]
        result = v.target_summary(records, self.controls)
        self.assertFalse(result['both_20pct_targets_met'])
        self.assertIsNone(result['tasks']['rmse']['best_new_outer'])

    def test_h70_known_outer_improvement_is_below_twenty_percent(self):
        controls = {'rmse': 39.39219382766771, 'macro_f1': .6495311620074445}
        result = v.target_summary([self.run_record('rmse', 34.318489927675174)], controls)
        self.assertAlmostEqual(result['tasks']['rmse']['threshold'], 31.51375506213417)
        self.assertAlmostEqual(result['tasks']['rmse']['relative_gain'], .12879972926079947, places=12)
        self.assertFalse(result['both_20pct_targets_met'])


class AnalyticalBlendTests(unittest.TestCase):
    def test_interior_optimum(self):
        self.assertEqual(v.convex_weight([0., 0.], [2., 2.], [1., 1.]), .5)

    def test_boundary_optimum_not_a_new_model_gain(self):
        self.assertEqual(v.convex_weight([0., 0.], [2., 2.], [-1., -1.]), 1.)

    def test_identical_predictions_choose_first(self):
        self.assertEqual(v.convex_weight([1., 2.], [1., 2.], [0., 3.]), 1.)

    def test_nonfinite_or_unaligned_blend_rejected(self):
        with self.assertRaises(ValueError):
            v.convex_weight([np.nan], [1.], [0.])
        with self.assertRaises(ValueError):
            v.convex_weight([1., 2.], [1.], [0.])



if __name__ == '__main__':
    unittest.main()
