"""Read-only SPEC011 artifact audit. Never calls fit or changes experiment state.

Metric arithmetic and cache assembly are independent of the sprint worker/loader.
Trusted historical joblib is loaded only to predict an inner teacher's training
probabilities; no candidate model is deserialized. Configuration constructors are
read from each hash-verified frozen source, and are instantiated without fitting.
Recorded absence of refits is evidence, not proof against unlogged execution.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUITE = Path('results/2026-10-24/series-three-hours-v1')
REG_OUTER = Path('results/2026-10-24/HYP-069-01-regression-alternatives')
REG_INNER = Path('results/2026-10-24/HYP-062-03-regression-control-rewrite')
CLASS_OUTER = Path('results/2026-10-23/HYP-048-01-multiclass-combinations')
OUTER_COMPONENTS = {'raw': 'HYP-011-02-multiclass-none',
                    'smote': 'HYP-038-01-multiclass-spec006',
                    'frequency': 'HYP-017-01-multiclass-frequency'}
IGNORED_RECIPE = {'seed', 'random_state', 'random_seed', 'name', 'hypothesis',
                  'n_jobs', 'thread_count', 'threads', 'verbose', 'verbosity'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, value):
    root = Path(root).resolve()
    result = (root / value).resolve()
    require(result.is_relative_to(root), 'Artifact path escapes repository/source root')
    return result


def arrays(path, keys=None):
    # Access all requested arrays eagerly: object arrays are rejected here.
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in (saved.files if keys is None else keys)}


def indices(value, name, size=None):
    a = np.asarray(value)
    require(a.ndim == 1 and len(a) > 0 and a.dtype.kind in 'iu', name + ': integer vector required')
    require(len(np.unique(a)) == len(a) and np.all(a >= 0), name + ': duplicate/negative position')
    require(size is None or np.all(a < size), name + ': position out of bounds')
    return a


def partition(fit, valid, size=None, full=False):
    fit, valid = indices(fit, 'fit', size), indices(valid, 'valid', size)
    require(not np.intersect1d(fit, valid).size, 'Training and validation overlap')
    require(not full or len(fit) + len(valid) == size, 'Incomplete partition')
    return fit, valid


def metric(y, prediction, name):
    y, p = np.asarray(y), np.asarray(prediction)
    require(y.ndim == 1 and len(y) and p.shape == y.shape, 'Metric arrays must be aligned nonempty vectors')
    if name == 'rmse':
        require(y.dtype.kind in 'fiu' and p.dtype.kind in 'fiu', 'RMSE requires numeric arrays')
        y, p = y.astype(np.float64), p.astype(np.float64)
        require(np.isfinite(y).all() and np.isfinite(p).all(), 'Nonfinite regression value')
        value = float(np.sqrt(np.mean((y - p) ** 2)))
        require(np.isfinite(value), 'Nonfinite computed RMSE')
        return value
    require(name == 'macro_f1', 'Unknown metric')
    require(y.dtype.kind == 'U' and p.dtype.kind == 'U', 'Class labels must be Unicode arrays')
    labels = np.union1d(y, p)
    scores = []
    for label in labels:
        tp = np.count_nonzero((y == label) & (p == label))
        denominator = np.count_nonzero(y == label) + np.count_nonzero(p == label)
        scores.append(2 * tp / denominator if denominator else 0.)
    return float(np.mean(scores))


def close(actual, expected, name, tolerance=1e-12):
    require(np.isfinite(actual) and np.isfinite(expected)
            and np.isclose(actual, expected, atol=tolerance, rtol=tolerance), name + ': value mismatch')


def probabilities(value, classes, rows):
    p, classes = np.asarray(value), np.asarray(classes)
    require(classes.ndim == 1 and classes.dtype.kind == 'U' and len(classes)
            and len(np.unique(classes)) == len(classes), 'Invalid Unicode class order')
    require(p.dtype.kind in 'fiu' and p.shape == (rows, len(classes)), 'Probability shape/dtype mismatch')
    p = p.astype(float)
    require(np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1 + 1e-7), 'Invalid probability values')
    require(np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=1e-6), 'Probability rows do not sum to one')
    return p / p.sum(axis=1, keepdims=True)


def semantic(value):
    if isinstance(value, dict):
        return {str(k): semantic(v) for k, v in value.items() if k not in IGNORED_RECIPE}
    if isinstance(value, (list, tuple)):
        return [semantic(v) for v in value]
    return value


def data_digest(value):
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(np.asarray(value))
    digest = hashlib.sha256(json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


class Checks:
    def __init__(self):
        self.rows = []

    def run(self, name, action):
        try:
            detail = action()
            self.rows.append({'name': name, 'passed': True, 'detail': detail})
            return detail
        except Exception as exc:
            self.rows.append({'name': name, 'passed': False,
                              'error': type(exc).__name__ + ': ' + str(exc)})
            return None


def verify_hashes(data_directory, expected):
    aliases = {'train': 'train.csv', 'test': 'test.csv', 'sample': 'sample_submission.csv',
               'dictionary': 'data_dictionary.csv', 'task': 'task.md'}
    require(isinstance(expected, dict) and len(expected) >= 4, 'Missing data hash evidence')
    actual = {}
    for key, digest in expected.items():
        path = safe_path(data_directory, aliases.get(key, key))
        actual[key] = sha256(path)
        require(actual[key] == digest, 'Actual input SHA mismatch: ' + key)
    require({'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'}
            <= {aliases.get(k, k) for k in expected}, 'Required data hashes absent')
    return actual


def verify_sources(directory, evidence):
    frozen = read_json(directory / 'frozen.json')
    require(frozen.get('spec') == 'SPEC011' and frozen.get('sources'), 'Invalid frozen manifest')
    require(evidence['sources'] == frozen['sources'], 'Attempt/frozen source manifests differ')
    root = directory / 'source'
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*.py')}
    require(actual_files == set(frozen['sources']), 'Source snapshot Python file set differs from manifest')
    for name, expected in frozen['sources'].items():
        require(sha256(safe_path(root, name)) == expected, 'Frozen source SHA mismatch: ' + name)
    require(datetime.fromisoformat(frozen['created']) <= datetime.fromisoformat(evidence['started']),
            'Sources were not frozen before attempt')
    return {'files': len(frozen['sources']), 'sha256_exact': True}


def expected_configuration(directory, request, recorded):
    """Match a constructor imported by the frozen worker, without running fit.

    New frozen hypotheses work without an ID range hardcoded in this verifier.
    Class dispatch is also checked explicitly for the original six hypotheses.
    """
    known = {70: 'TemporalEntityLinker', 71: 'RatingNormalizedRegressor',
             72: 'TeacherResidualF1Classifier', 73: 'ECOCClassifier',
             74: 'ContextualTeacherCorrector', 75: 'MacroF1Decision'}
    worker = ast.parse((directory / 'source/experiments/sprint_worker.py').read_text(encoding='utf8'))
    imported = [(node.module, alias.name) for node in ast.walk(worker)
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('sprint_')
                for alias in node.names]
    target = {k: v for k, v in recorded.items() if k not in ('teacher_train_sha256', 'teacher_classes')}
    matches, errors = [], []
    for module_name, class_name in imported:
        if request['number'] in known and class_name != known[request['number']]:
            continue
        path = directory / 'source' / (module_name.replace('.', '/') + '.py')
        try:
            module_id = '_sprint_verify_' + hashlib.sha256(str(path).encode()).hexdigest()
            spec = importlib.util.spec_from_file_location(module_id, path)
            module = importlib.util.module_from_spec(spec)
            previous_bytecode = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                spec.loader.exec_module(module)
            finally:
                sys.dont_write_bytecode = previous_bytecode
            cls = getattr(module, class_name)
            constructor = cls(**request['parameters'])
            config = constructor.configuration
            config = config() if callable(config) else config
            if config == target:
                tree = ast.parse(path.read_text(encoding='utf8'))
                definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
                audit_explicit, audit_spread = set(), False
                for node in ast.walk(definition):
                    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                            and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                    and t.value.id == 'self' and t.attr == 'audit_' for t in node.targets)):
                        continue
                    for key, expression in zip(node.value.keys, node.value.values):
                        if key is None and 'configuration' in ast.unparse(expression):
                            audit_spread = True
                        elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                            audit_explicit.add(key.value)
                matches.append({'constructor': class_name, 'audit_spreads_configuration': audit_spread,
                                'audit_explicit_fields': sorted(audit_explicit)})
        except Exception as exc:
            errors.append(class_name + ': ' + type(exc).__name__ + ': ' + str(exc))
    require(len(matches) == 1, 'Configuration does not uniquely match frozen constructor/defaults: ' + str(errors))
    return {**matches[0], 'exact_frozen_defaults_and_parameters': True, 'fits': 0}


def audit_configuration(configuration, audit, constructor):
    """Only compare fields that the frozen code claims are configuration.

    A learned audit may deliberately overwrite a configuration field with a
    statistic (H76 training_pairs), or use a name for nested fitted-state audit
    (H78 decision). Frozen AST tells these apart without ignoring real changes.
    """
    plain = {k: v for k, v in configuration.items() if k not in ['teacher_train_sha256', 'teacher_classes']}
    if 'configuration' in audit:
        require(audit['configuration'] == plain, 'Audit configuration differs')
        return {'full_configuration_in_audit': True}
    inherited = set(plain) - set(constructor['audit_explicit_fields']) if constructor['audit_spreads_configuration'] else set()
    for key in inherited:
        require(key in audit and audit[key] == plain[key], 'Inherited audit/configuration differ: ' + key)
    return {'full_configuration_in_audit': False, 'inherited_configuration_fields_checked': sorted(inherited),
            'explicit_audit_fields_have_diagnostic_semantics': constructor['audit_explicit_fields']}


def cache_key(X, y, positions, config):
    digest = hashlib.sha256(json.dumps({'version': 2, 'task': 'multiclass', 'config': config,
        'columns': list(X.columns), 'dtypes': list(map(str, X.dtypes)),
        'split': 'SGKF3-seed42-rawgroups'}, sort_keys=True).encode())
    for value in (pd.util.hash_pandas_object(X, index=False).to_numpy(),
                  pd.util.hash_pandas_object(pd.Series(np.asarray(y)), index=False).to_numpy(),
                  np.asarray(positions, dtype=np.int64)):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def read_oof(root, bundle, outer_fit, component, audit):
    from sklearn.model_selection import StratifiedGroupKFold
    X, y = bundle.X.iloc[outer_fit], np.asarray(bundle.y)[outer_fit]
    config = ({'policy': 'legacy'} if component == 'raw' else {'component': component,
        'protocol': 'H38-bounded-smotenc-v1' if component == 'smote' else 'H17-tagged-frequency-native-v1',
        'parameters': {}})
    key = cache_key(X, y, outer_fit, config)
    series = 'series-twenty-hypotheses-v1' if component == 'raw' else 'series-combinations-v1'
    folder = root / 'results/2026-10-23' / series / 'cache' / key
    metadata = read_json(folder / 'metadata.json')
    require(metadata['key'] == key and metadata['config'] == config, 'Teacher cache metadata/key mismatch')
    require(audit['key'] == key and audit['config'] == config and audit['model_fits'] == 0,
            'Teacher cache audit mismatch')
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    expected = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(X, y, groups))
    paths = [folder / f'fold-{i}.npz' for i in range(3)]
    require([safe_path(root, p) for p in audit['artifacts']] == paths, 'Teacher artifact paths mismatch')
    classes = np.unique(y).astype(str)
    oof, seen, folds = np.full((len(y), len(classes)), np.nan), set(), []
    for path, (want_fit, want_valid) in zip(paths, expected):
        saved = arrays(path)
        fit, valid = partition(saved['fit'], saved['valid'], len(y), full=True)
        require(np.array_equal(fit, want_fit) and np.array_equal(valid, want_valid), 'OOF local split differs')
        require(not set(groups[fit]) & set(groups[valid]), 'Raw duplicate groups overlap teacher fold')
        require(not seen & set(map(int, valid)), 'OOF validation rows repeated')
        seen.update(map(int, valid))
        require(np.array_equal(indices(saved['fit_positions'], 'OOF global fit', len(bundle.y)), outer_fit[fit])
                and np.array_equal(indices(saved['valid_positions'], 'OOF global valid', len(bundle.y)), outer_fit[valid]), 'OOF global position alignment mismatch')
        require(saved['classes'].dtype.kind == 'U' and np.array_equal(saved['classes'], classes), 'OOF teacher class order mismatch')
        oof[valid] = probabilities(saved['probabilities'], classes, len(valid))
        folds.append(saved)
    require(len(seen) == len(y), 'OOF coverage incomplete')
    return folder, classes, oof, folds


def stack_probability(parts, state, classes):
    require(state['classes'] == classes.tolist() and state['component_order'] == ['raw', 'smote', 'frequency']
            and state['clip'] == 1e-6, 'H48 frozen meta protocol mismatch')
    values = np.concatenate([np.log(np.clip(p, 1e-6, 1.)) for p in parts], axis=1)
    coef, intercept = np.asarray(state['coef'], float), np.asarray(state['intercept'], float)
    require(coef.shape == (len(classes), values.shape[1]) and intercept.shape == (len(classes),), 'H48 coefficient dimensions mismatch')
    require(np.isfinite(coef).all() and np.isfinite(intercept).all(), 'H48 coefficients nonfinite')
    z = values @ coef.T + intercept
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def class_teacher(root, bundle, stage, audit):
    """Rebuild cache provenance without using experiments.sprint_cache."""
    context = root / CLASS_OUTER
    verify_hashes(root / 'data/multiclass', read_json(context / 'evidence.json')['hashes'])
    split = arrays(context / 'splits.npz')
    outer_fit, outer_valid = partition(split['fit'], split['valid'], len(bundle.y), full=True)
    if stage == 'inner':
        require(audit['teacher'] == 'raw_fold_0' and audit['model_fits'] == 0
                and audit['independent_valid_predictions'] is True and audit['H48_meta_state_used'] is False
                and audit['training_predictions'] == 'in_sample_raw_teacher', 'Incorrect inner teacher independence audit')
        folder, classes, _, folds = read_oof(root, bundle, outer_fit, 'raw', audit['component'])
        model_path = folder / 'fold-0-model.joblib'
        require(safe_path(root, audit['model_artifact']) == model_path, 'Unexpected teacher model artifact')
        # Trusted historical repository model; inference only, never fit.
        import joblib
        model = joblib.load(model_path)
        require(np.array_equal(np.asarray(model.classes_).astype(str), classes), 'Raw teacher model class order mismatch')
        fit, valid = folds[0]['fit_positions'], folds[0]['valid_positions']
        train = probabilities(model.predict_proba(bundle.X.iloc[fit]), classes, len(fit))
        valid_p = probabilities(model.predict_proba(bundle.X.iloc[valid]), classes, len(valid))
        require(np.allclose(valid_p, folds[0]['probabilities'], atol=1e-7, rtol=1e-7), 'Historical teacher prediction/cache mismatch')
        note = 'Held-out raw fold0; training teacher predictions in-sample; proxy is not full H48 validation'
        artifacts = {str(model_path.relative_to(root)): sha256(model_path)}
    else:
        require(audit['teacher'] == 'H48' and audit['protocol'] == 'original_outer_only'
                and audit['model_fits'] == 0 and audit['independent_training_oof'] is False
                and audit['training_meta_predictions'] == 'meta_in_sample'
                and audit['inner_validation_available'] is False, 'Incorrect H48 teacher independence audit')
        trains, valids, artifacts = [], [], {}
        for component in ['raw', 'smote', 'frequency']:
            folder, classes, oof, folds = read_oof(root, bundle, outer_fit, component, audit['components'][component])
            source = root / 'results/2026-10-23' / OUTER_COMPONENTS[component]
            verify_hashes(root / 'data/multiclass', read_json(source / 'evidence.json')['hashes'])
            saved = arrays(source / 'validation_predictions.npz', ['positions', 'classes', 'probabilities'])
            require(np.array_equal(saved['positions'], outer_valid) and np.array_equal(saved['classes'], classes),
                    'Original component outer class/row alignment differs')
            trains.append(oof)
            valids.append(probabilities(saved['probabilities'], classes, len(outer_valid)))
            artifacts[str((source / 'validation_predictions.npz').relative_to(root))] = sha256(source / 'validation_predictions.npz')
        state_path = context / 'learned-state.json'
        require(safe_path(root, audit['meta_state_artifact']) == state_path, 'H48 meta state path differs')
        state = read_json(state_path)['meta_state']
        train, valid_p = stack_probability(trains, state, classes), stack_probability(valids, state, classes)
        saved = arrays(context / 'validation_predictions.npz', ['positions', 'classes', 'probabilities'])
        require(np.array_equal(saved['positions'], outer_valid) and np.array_equal(saved['classes'], classes)
                and np.allclose(valid_p, saved['probabilities'], atol=1e-12, rtol=1e-12), 'Reconstructed H48 differs from original saved outer')
        fit, valid = outer_fit, outer_valid
        note = 'Base OOF plus in-sample H48 meta teacher for training; original outer repeatedly viewed/exploratory'
        artifacts[str(state_path.relative_to(root))] = sha256(state_path)
    return {'fit': fit, 'valid': valid, 'classes': classes, 'train': train,
            'reference': classes[valid_p.argmax(axis=1)], 'note': note, 'artifacts': artifacts}


def target_summary(runs, controls):
    """Only validated ORIGINAL outer scores can satisfy the two task thresholds."""
    result = {'scope': 'original_outer_only_exploratory', 'tasks': {}}
    for name in ['rmse', 'macro_f1']:
        control = controls[name]
        threshold = control * (.8 if name == 'rmse' else 1.2)
        candidates = [r for r in runs if r.get('verified') and r.get('stage') == 'outer' and r.get('metric') == name]
        best = (min(candidates, key=lambda r: r['score']) if name == 'rmse' else max(candidates, key=lambda r: r['score'])) if candidates else None
        met = bool(best and (best['score'] <= threshold if name == 'rmse' else best['score'] >= threshold))
        result['tasks'][name] = {'control': control, 'threshold': threshold,
            'best_new_outer': {k: best[k] for k in ['directory', 'score']} if best else None,
            'relative_gain': ((control - best['score']) / control if name == 'rmse' else (best['score'] - control) / control) if best else None,
            'target_met': met}
    result['both_20pct_targets_met'] = all(v['target_met'] for v in result['tasks'].values())
    return result


def convex_weight(first, second, target):
    first, second, target = (np.asarray(a, float) for a in [first, second, target])
    require(first.ndim == 1 and len(first) and first.shape == second.shape == target.shape
            and all(np.isfinite(a).all() for a in [first, second, target]), 'Invalid analytic blend arrays')
    direction = first - second
    squared_norm = np.dot(direction, direction)
    return float(np.clip(np.dot(target - second, direction) / squared_norm, 0., 1.)) if squared_norm else 1.


def verify_analytical_run(root, out, prior_runs):
    """Audit H79's cached-prediction parameter fit, outside sprint_run registry."""
    checks = Checks()
    report = {'directory': str(out.relative_to(root)), 'hypothesis': 'HYP-079', 'stage': 'inner',
              'attempt_kind': 'analytical_cached_combination', 'checks': checks.rows, 'verified': False}
    try:
        evidence, result = read_json(out / 'evidence.json'), read_json(out / 'metrics.json')
        report.update(status=result['status'], metric=result['metric'], score=result['score'], reference_score=result['reference_score'])
        require(result['status'] == 'completed' and result['hypothesis'] == 'HYP-079'
                and result['stage'] == 'inner' and result['metric'] == 'rmse', 'Analytic result scope differs')
        config = evidence['configuration']
        require(config['strategy'] == 'H70_H76_convex_temporal_combination'
                and config['selection'] == 'analytic_least_squares_inner_only', 'Unknown analytical protocol')
        require(config['sources'] == ['results/2026-10-24/HYP-070-03-inner-high-sprint',
                                       'results/2026-10-24/HYP-076-01-inner-default-sprint'], 'Analytic source selection differs')
        def code_check():
            require(sha256(out / 'source.py') == evidence['code_sha256'], 'Analytic frozen source SHA mismatch')
            tree = ast.parse((out / 'source.py').read_text(encoding='utf8'))
            require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                            and n.func.attr in ['fit', 'partial_fit', 'fit_transform'] for n in ast.walk(tree)),
                    'Analytic source unexpectedly contains learner fit calls')
            return {'source_sha256': evidence['code_sha256'], 'learner_fit_calls_in_source': 0,
                    'initial_evidence_status': evidence['status'], 'terminal_status_from_metrics_and_claim': result['status']}
        checks.run('analytical_source_snapshot_no_learner_fit', code_check)
        sources = [safe_path(root, p) for p in config['sources']]
        original = [arrays(p / 'validation_predictions.npz') for p in sources]
        saved = arrays(out / 'validation_predictions.npz')
        a, b = original

        def source_check():
            actual_hashes = {name: sha256(source / 'validation_predictions.npz') for name, source in zip(config['sources'], sources)}
            require(actual_hashes == evidence['source_prediction_hashes'] == result['source_prediction_hashes'], 'Analytic input SHA mismatch')
            for source in sources:
                matches = [r for r in prior_runs if safe_path(root, r['directory']) == source]
                require(len(matches) == 1 and matches[0].get('verified') is True and matches[0]['stage'] == 'inner'
                        and matches[0]['metric'] == 'rmse', 'Analytic source lacks independently verified inner provenance')
            require(np.array_equal(indices(a['positions'], 'first source positions'), indices(b['positions'], 'second source positions'))
                    and np.array_equal(a['y'], b['y']), 'Analytic sources target/position mismatch')
            require(np.array_equal(indices(saved['positions'], 'analytic positions'), a['positions'])
                    and np.array_equal(saved['y'], a['y']) and np.array_equal(saved['reference'], a['prediction']), 'Saved analytic target/position/reference mismatch')
            return actual_hashes
        checks.run('analytical_source_hashes_and_inner_provenance', source_check)

        def arithmetic_check():
            weight = convex_weight(a['prediction'], b['prediction'], a['y'])
            close(weight, result['weight_h70'], 'Analytic H70 weight')
            close(1 - weight, result['weight_h76'], 'Analytic H76 weight')
            expected = weight * a['prediction'] + (1 - weight) * b['prediction']
            require(np.array_equal(saved['prediction'], expected), 'Analytic predictions differ from exact convex blend')
            score, reference = metric(saved['y'], saved['prediction'], 'rmse'), metric(saved['y'], a['prediction'], 'rmse')
            close(score, result['score'], 'Analytic RMSE')
            close(reference, result['reference_score'], 'Analytic compatible H70 control')
            require(result['eligible_for_outer'] == bool(0 < weight < 1 and score < reference), 'Analytic outer eligibility differs')
            require(result['old_model_refits'] == 0, 'Analytic run records old learner refit')
            return {'weight_h70': weight, 'weight_h76': 1 - weight, 'score': score,
                    'control': reference, 'eligible_for_outer': result['eligible_for_outer'],
                    'scope': 'inner_selected_combination_not_independent_outer_score'}
        checks.run('analytical_independent_optimum_and_metric', arithmetic_check)

        def claim_check():
            records = read_json(out / 'model-fit-history.json')
            require(len(records) == 1, 'Analytic parameter fit claim not unique')
            claim = records[0]
            identity = {'recipe': semantic(config), 'X_sha256': data_digest(pd.DataFrame({'h70': a['prediction'], 'h76': b['prediction']})),
                        'y_sha256': data_digest(a['y'].reshape(-1, 1))}
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
            require(fingerprint == result['fingerprint'] == claim['fingerprint']
                    and all(claim[k] == value for k, value in identity.items()), 'Analytic actual configuration/data identity differs')
            require(all(claim[key] == value for key, value in result.items() if key != 'finished'), 'Analytic claim/result outcome differs')
            require(evidence.get('ledger_sha256') == claim['ledger_sha256'], 'Analytic ledger snapshot evidence differs')
            require(datetime.fromisoformat(evidence['started']) <= datetime.fromisoformat(claim['started'])
                    <= datetime.fromisoformat(result['finished']) <= datetime.fromisoformat(claim['finished']), 'Analytic pre-fit/terminal timestamp order differs')
            return {'fingerprint': fingerprint, 'parameter_fits': 1, 'learner_fits': 0}
        checks.run('analytical_parameter_fit_identity', claim_check)
        report['verified'] = all(row['passed'] for row in checks.rows)
    except Exception as exc:
        checks.rows.append({'name': 'required_analytical_artifact_or_structure', 'passed': False,
                            'error': type(exc).__name__ + ': ' + str(exc)})
    return report


def verify_run(root, hypothesis_directory, event, bundles, teachers):
    checks = Checks()
    report = {'directory': event.get('directory'), 'hypothesis': event.get('hypothesis'),
              'status': event.get('status'), 'stage': event.get('stage'), 'attempt_kind': 'sprint_worker', 'checks': checks.rows}
    try:
        out = safe_path(root, event['directory'])
        evidence, request, result = [read_json(out / name) for name in ['evidence.json', 'request.json', 'metrics.json']]
        checks.run('attempt_evidence_exact', lambda: require(evidence == event, 'Registry/evidence differ'))
        checks.run('source_snapshot', lambda: verify_sources(hypothesis_directory, evidence))
        frozen = read_json(hypothesis_directory / 'frozen.json')

        def request_check():
            require(request['number'] == int(event['hypothesis'].split('-')[1]) and request['stage'] == event['stage'], 'Request hypothesis/stage differs')
            require(request['parameters'] == event['parameters'] == frozen['variants'][event['variant']], 'Request parameters were not frozen')
            require(safe_path(root, request['repo_root']) == root and safe_path(root, request['output']) == out,
                    'Request repository/output differs')
            require(safe_path(root, request['history']) == hypothesis_directory / 'model-fit-history.json', 'Request fit history differs')
        checks.run('frozen_request', request_check)
        require(result['status'] == event['status'], 'Registry/result status differs')
        if result['status'] != 'completed':
            report.update(verified=False, reason='No completed metric; not counted as negative scientific outcome')
            return report
        report.update(metric=result['metric'], score=result['score'], reference_score=result['reference_score'])
        task = 'regression' if result['metric'] == 'rmse' else 'multiclass'
        key = (task, event['stage'] == 'inner' and task == 'regression')
        if key not in bundles:
            if key[1]:
                from research_data import load_task
            else:
                from automl_data import load_task
            bundles[key] = load_task(root / 'data' / task / 'task.md')
        bundle = bundles[key]
        data = read_json(out / 'data.json')
        checks.run('input_file_sha256', lambda: verify_hashes(root / 'data' / task, data['hashes']))
        target_column = getattr(bundle, 'target_column', getattr(bundle, 'target', None))
        checks.run('features_exclude_id_target', lambda: require(data['id_excluded_from_X'] is True and data['target_excluded_from_X'] is True
            and target_column is not None and bundle.id_column not in bundle.X and target_column not in bundle.X, 'ID/target feature exclusion failed'))
        split, saved, configuration, audit, teacher_audit = [arrays(out / 'splits.npz'), arrays(out / 'validation_predictions.npz'),
            read_json(out / 'configuration.json'), read_json(out / 'audit.json'), read_json(out / 'teacher-audit.json')]
        fit, valid = partition(split['fit'], split['valid'], len(bundle.y))
        checks.run('position_and_target_alignment', lambda: require(np.array_equal(indices(saved['positions'], 'prediction positions', len(bundle.y)), valid)
            and np.array_equal(np.asarray(bundle.y)[valid].astype(str) if task == 'multiclass' else np.asarray(bundle.y)[valid], saved['y'])
            and result['fit_rows'] == len(fit) and result['valid_rows'] == len(valid), 'Saved positions/targets/row counts differ'))

        def metrics_check():
            score, reference = metric(saved['y'], saved['prediction'], result['metric']), metric(saved['y'], saved['reference'], result['metric'])
            close(score, result['score'], 'Score')
            close(reference, result['reference_score'], 'Reference score')
            gain = (reference - score) / reference if task == 'regression' else (score - reference) / reference
            close(gain, result['relative_gain'], 'Relative gain')
            for field in ['status', 'score', 'reference_score', 'relative_gain', 'fingerprint', 'metric', 'fit_rows', 'valid_rows']:
                require(event[field] == result[field], 'Registry/result differ: ' + field)
            return {'score': score, 'reference_score': reference, 'relative_gain': gain}
        checks.run('independent_metric_arithmetic', metrics_check)
        # Do not execute snapshot constructors whose bytes failed integrity checking.
        constructor = None
        if any(row['name'] == 'source_snapshot' and row['passed'] for row in checks.rows):
            constructor = checks.run('actual_configuration_vs_frozen', lambda: expected_configuration(hypothesis_directory, request, configuration))
        else:
            checks.run('actual_configuration_vs_frozen', lambda: require(False, 'Skipped unsafe mutated snapshot'))

        checks.run('fit_audit_configuration', lambda: audit_configuration(configuration, audit, constructor)
                   if constructor else require(False, 'Frozen constructor unavailable'))

        def fit_identity():
            records = read_json(hypothesis_directory / 'model-fit-history.json')
            require(len({r['fingerprint'] for r in records}) == len(records), 'Duplicate fit identities')
            identity = {'recipe': semantic(configuration), 'X_sha256': data_digest(bundle.X.iloc[fit]),
                        'y_sha256': data_digest(np.asarray(bundle.y)[fit].reshape(-1, 1))}
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
            require(fingerprint == result['fingerprint'], 'Actual configuration/data fit identity differs')
            matches = [r for r in records if r['fingerprint'] == fingerprint]
            require(len(matches) == 1, 'Missing unique registered fit')
            claim = matches[0]
            require(all(claim[k] == v for k, v in identity.items()) and claim['status'] == 'completed'
                    and claim['stage'] == event['stage'] and claim['hypothesis'] == event['hypothesis'], 'Fit claim mismatch')
            require(datetime.fromisoformat(event['started']) <= datetime.fromisoformat(claim['started'])
                    <= datetime.fromisoformat(claim['finished']) <= datetime.fromisoformat(event['finished']), 'Fit claim lifecycle timestamps differ')
            require(claim.get('ledger_sha256') and event.get('ledger_sha256'), 'Missing pre-fit ledger evidence')
            return {'fingerprint': fingerprint, 'ledger_read_is_recorded_not_independently_observable': True}
        checks.run('fit_identity_registration', fit_identity)
        checks.run('recorded_old_refits_zero', lambda: require(result['old_baseline_refits'] == 0 and
            teacher_audit.get('old_model_refits', teacher_audit.get('model_fits')) == 0, 'Recorded old model refit'))

        if task == 'regression':
            def regression_reference():
                source = root / (REG_INNER if event['stage'] == 'inner' else REG_OUTER)
                require(safe_path(root, teacher_audit['baseline_source']) == source, 'Wrong regression reference provenance')
                expected = arrays(source / 'splits.npz')
                reference = arrays(source / 'validation_predictions.npz', ['positions', 'prediction'])
                require(np.array_equal(fit, expected['fit']) and np.array_equal(valid, expected['valid'])
                        and np.array_equal(valid, reference['positions']) and np.array_equal(saved['reference'], reference['prediction']),
                        'Regression reference/splits differ from compatible saved control')
                if event['stage'] == 'inner':
                    year = pd.to_numeric(bundle.X['Year_Factor']).to_numpy()
                    require(np.all(year[fit] < 4) and np.all(year[valid] == 4), 'Incorrect inner temporal split')
                return {'source': str(source.relative_to(root)), 'prediction_sha256': sha256(source / 'validation_predictions.npz')}
            checks.run('compatible_saved_regression_reference', regression_reference)
        else:
            def class_checks():
                teacher_key = event['stage']
                if teacher_key not in teachers:
                    teachers[teacher_key] = class_teacher(root, bundle, teacher_key, teacher_audit)
                teacher = teachers[teacher_key]
                # Re-validate each attempt's own provenance even when numerical teacher is cached.
                previous_audit = teachers.get(('audit', teacher_key))
                if previous_audit is None:
                    teachers[('audit', teacher_key)] = teacher_audit
                else:
                    require(previous_audit == teacher_audit, 'Per-run teacher provenance differs')
                require(np.array_equal(fit, teacher['fit']) and np.array_equal(valid, teacher['valid'])
                        and np.array_equal(saved['reference'], teacher['reference']), 'Teacher split/reference mismatch')
                classes = np.unique(np.asarray(bundle.y)[fit]).astype(str)
                require(np.array_equal(saved['classes'], classes) and np.array_equal(teacher['classes'], classes)
                        and configuration['teacher_classes'] == classes.tolist(), 'Candidate/teacher/training class order differs')
                p = probabilities(saved['probabilities'], saved['classes'], len(valid))
                require(np.array_equal(saved['prediction'], classes[p.argmax(axis=1)]), 'Prediction differs from probability argmax')
                require(configuration['teacher_train_sha256'] == hashlib.sha256(teacher['train'].tobytes()).hexdigest(), 'Training teacher probability SHA mismatch')
                if event['stage'] == 'inner':
                    require(result['inner_is_full_H48_pipeline_validation'] is False, 'Inner proxy misrepresented as H48 pipeline validation')
                report['teacher_scope'] = teacher['note']
                return {'artifacts': teacher['artifacts'], 'teacher_class_order_exact': True, 'teacher_train_sha256_exact': True}
            checks.run('teacher_provenance_class_order_and_predictions', class_checks)
        report['verified'] = all(row['passed'] for row in checks.rows)
    except Exception as exc:
        checks.rows.append({'name': 'required_artifact_or_structure', 'passed': False,
                            'error': type(exc).__name__ + ': ' + str(exc)})
        report['verified'] = False
    return report


def verify_suite(root=ROOT, suite=SUITE):
    root = Path(root).resolve()
    suite = safe_path(root, suite)
    runs, bundles, teachers, registry_errors = [], {}, {}, []
    for directory in sorted(suite.glob('HYP-*')):
        try:
            events = read_json(directory / 'attempts.json')
            require(isinstance(events, list), 'Attempt registry must be list')
            require(len({(r['stage'], r['variant']) for r in events}) == len(events), 'Repeated stage/variant attempt')
            require(sum(r['stage'] == 'outer' for r in events) <= 1, 'More than one outer attempt per hypothesis')
            for event in events:
                runs.append(verify_run(root, directory, event, bundles, teachers))
        except Exception as exc:
            registry_errors.append({'directory': str(directory.relative_to(root)), 'error': type(exc).__name__ + ': ' + str(exc)})
    for analytical in sorted((root / 'results/2026-10-24').glob('HYP-079-*-inner-combination-sprint')):
        runs.append(verify_analytical_run(root, analytical, runs))
    require(runs or registry_errors, 'No registered sprint attempts found')
    global_checks = Checks()
    controls = {}
    for name, source in [('rmse', REG_OUTER), ('macro_f1', CLASS_OUTER)]:
        def control_check(name=name, source=source):
            from automl_data import load_task
            task = 'regression' if name == 'rmse' else 'multiclass'
            bundle = load_task(root / 'data' / task / 'task.md')
            saved = arrays(root / source / 'validation_predictions.npz', ['positions', 'prediction'])
            split = arrays(root / source / 'splits.npz')
            partition(split['fit'], split['valid'], len(bundle.y), full=True)
            require(np.array_equal(saved['positions'], split['valid']), 'Original control split mismatch')
            y = np.asarray(bundle.y)[saved['positions']]
            if name == 'macro_f1':
                y = y.astype(str)
            score = metric(y, saved['prediction'], name)
            stored = read_json(root / source / 'metrics.json')
            # Earlier metrics use either a flat score or task metric field.
            wanted = stored.get('score', stored.get(name))
            if wanted is None and isinstance(stored.get('metrics'), dict):
                wanted = stored['metrics'][name]
            close(score, wanted, 'Original control score')
            controls[name] = score
            return {'source': str(source), 'score': score, 'predictions_sha256': sha256(root / source / 'validation_predictions.npz')}
        global_checks.run('original_' + name + '_control', control_check)
    completed = [r for r in runs if r['status'] == 'completed']
    summary = {'spec': 'SPEC011', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'verifier': str(Path(__file__).relative_to(ROOT)), 'verifier_sha256': sha256(__file__),
        'model_fits_performed': 0, 'runs': runs, 'registry_errors': registry_errors, 'global_checks': global_checks.rows,
        'registered_attempts': len(runs), 'completed_attempts': len(completed),
        'registered_sprint_attempts': sum(r['attempt_kind'] == 'sprint_worker' for r in runs),
        'analytical_attempts': sum(r['attempt_kind'] == 'analytical_cached_combination' for r in runs),
        'verified_completed_attempts': sum(r.get('verified', False) for r in completed),
        'all_attempts_terminal': all(r['status'] in ['completed', 'failed', 'timeout', 'skipped'] for r in runs),
        'all_completed_runs_verified': bool(completed) and all(r.get('verified') for r in completed) and not registry_errors,
        'limitations': ['No model is fitted by this verifier; historical raw teacher inference may be executed.',
            'Old-refit absence and pre-fit ledger reads are checked against recorded claims, not a runtime trace.',
            'Inner classification is raw fold0 proxy; H48 training meta teacher is in-sample.',
            'An empty custom-objective loss history does not establish zero updates or convergence; LightGBM may copy bound objective objects.',
            'Original outer has already been viewed; results are exploratory and say nothing about hidden platform labels.']}
    summary['passed'] = summary['all_completed_runs_verified'] and summary['all_attempts_terminal'] and all(r['passed'] for r in global_checks.rows)
    summary['target'] = target_summary(runs, controls) if set(controls) == {'rmse', 'macro_f1'} else {'both_20pct_targets_met': False, 'error': 'Original control verification incomplete'}
    return summary


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--output', type=Path, help='Unique historical result path; existing results are never overwritten')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = safe_path(args.root, args.output or args.suite / ('verification-' + stamp + '.json'))
    require(output.name != 'verification.json', 'verification.json is reserved for the latest-result pointer; choose a unique result path')
    require(not output.exists(), 'Verification result already exists; choose a new append-only output path')
    result = verify_suite(args.root, args.suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pointer = {'latest_verification': str(output.relative_to(Path(args.root).resolve())),
               'sha256': sha256(output), 'created_utc': result['created_utc'], 'passed': result['passed']}
    (safe_path(args.root, args.suite) / 'verification.json').write_text(json.dumps(pointer, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: result[k] for k in ['passed', 'registered_attempts', 'completed_attempts', 'verified_completed_attempts', 'target']}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
