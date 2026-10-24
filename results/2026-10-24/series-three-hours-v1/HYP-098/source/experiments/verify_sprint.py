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


def source_data_hashes(root, task, source):
    evidence = read_json(source / 'evidence.json')
    hashes = evidence.get('hashes') or evidence.get('data', {}).get('hashes')
    return verify_hashes(root / 'data' / task, hashes)


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


def expected_configuration(directory, request, recorded, verified_extra_fields=(), verified_constructor_parameters=None):
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
    imported = list(dict.fromkeys(imported))
    target = {k: v for k, v in recorded.items() if k not in {'teacher_train_sha256', 'teacher_classes', *verified_extra_fields}}
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
            constructor = cls(**{**request['parameters'], **(verified_constructor_parameters or {})})
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


def audit_configuration(configuration, audit, constructor, verified_extra_fields=()):
    """Only compare fields that the frozen code claims are configuration.

    A learned audit may deliberately overwrite a configuration field with a
    statistic (H76 training_pairs), or use a name for nested fitted-state audit
    (H78 decision). Frozen AST tells these apart without ignoring real changes.
    """
    plain = {k: v for k, v in configuration.items() if k not in {'teacher_train_sha256', 'teacher_classes', *verified_extra_fields}}
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


def simplex_optimum(predictions, target):
    """Enumerate active faces and solve equality-constrained quadratic systems.

    This independently checks SLSQP's numeric parameter result; no estimator fit.
    """
    import itertools
    p, y = np.asarray(predictions, float), np.asarray(target, float)
    require(p.ndim == 2 and p.shape[0] == len(y) and y.ndim == 1 and len(y)
            and p.shape[1] <= 8 and np.isfinite(p).all() and np.isfinite(y).all(), 'Invalid simplex verification arrays')
    best = None
    for size in range(1, p.shape[1] + 1):
        for active in itertools.combinations(range(p.shape[1]), size):
            x = p[:, active]
            gram, rhs = x.T @ x / len(y), x.T @ y / len(y)
            system = np.block([[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            solution = np.linalg.lstsq(system, np.r_[rhs, 1.], rcond=None)[0][:size]
            if np.any(solution < -1e-8) or abs(solution.sum() - 1.) > 1e-7:
                continue
            weights = np.zeros(p.shape[1])
            weights[list(active)] = np.maximum(solution, 0.)
            weights /= weights.sum()
            objective = float(np.mean((p @ weights - y) ** 2))
            if best is None or objective < best[0]:
                best = (objective, weights)
    require(best is not None, 'No feasible simplex solution')
    return best


def verify_multiblend(root, out, prior_runs):
    checks = Checks()
    report = {'directory': str(out.relative_to(root)), 'hypothesis': 'HYP-091', 'status': 'unreadable', 'stage': 'unknown',
              'attempt_kind': 'analytical_cached_combination', 'checks': checks.rows, 'verified': False}
    try:
        evidence, result = read_json(out / 'evidence.json'), read_json(out / 'metrics.json')
        stage, configuration = result['stage'], evidence['configuration']
        report.update(stage=stage, status=result['status'], metric=result['metric'], score=result['score'], reference_score=result['reference_score'])
        require(stage in ['inner', 'outer'] and evidence['stage'] == stage and result['status'] == 'completed'
                and result['metric'] == 'rmse' and configuration == {'strategy': 'new_temporal_transport_conditional_uncertainty_pool',
                    'components': ['H70', 'H83', 'H85'], 'optimizer': 'inner_only_SLSQP_original_MSE_simplex',
                    'maxiter': 200, 'ftol': 1e-12, 'weight_pruning': 1e-8}, 'H91 frozen scope/configuration differs')
        def snapshot_check():
            require(sha256(out / 'source.py') == evidence['code_sha256'], 'H91 source SHA differs')
            tree = ast.parse((out / 'source.py').read_text(encoding='utf8'))
            require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                            and n.func.attr in ['fit', 'fit_transform', 'partial_fit'] for n in ast.walk(tree)), 'H91 source unexpectedly fits learners')
            return {'source_sha256': evidence['code_sha256'], 'learner_fit_calls': 0}
        checks.run('multiblend_frozen_source', snapshot_check)
        names = (['HYP-070-03-inner-high-sprint', 'HYP-083-01-inner-default-sprint', 'HYP-085-01-inner-default-sprint'] if stage == 'inner' else
                 ['HYP-070-04-outer-high-sprint', 'HYP-083-02-outer-default-sprint', 'HYP-085-02-outer-default-sprint'])
        sources = [root / 'results/2026-10-24' / name for name in names]
        source_arrays = [arrays(source / 'validation_predictions.npz') for source in sources]
        saved = arrays(out / 'validation_predictions.npz')
        p = np.column_stack([item['prediction'] for item in source_arrays])
        y = source_arrays[0]['y']
        weights = np.asarray(result['weights'], float)
        def source_check():
            actual = {str((source / 'validation_predictions.npz').relative_to(root)): sha256(source / 'validation_predictions.npz') for source in sources}
            expected = {str(Path(key)): value for key, value in evidence['source_sha256'].items()}
            require(actual == expected, 'H91 component source SHA mismatch')
            for source, item in zip(sources, source_arrays):
                matches = [r for r in prior_runs if safe_path(root, r['directory']) == source]
                require(len(matches) == 1 and matches[0].get('verified') is True and matches[0]['stage'] == stage, 'H91 component lacks independently verified compatible stage')
                require(np.array_equal(indices(item['positions'], 'H91 component positions'), source_arrays[0]['positions'])
                        and np.array_equal(item['y'], y), 'H91 component rows/targets differ')
            require(np.array_equal(indices(saved['positions'], 'H91 positions'), source_arrays[0]['positions'])
                    and np.array_equal(saved['y'], y) and np.array_equal(saved['reference'], p[:, 0]), 'H91 saved target/reference/position mismatch')
            return actual
        checks.run('multiblend_component_provenance', source_check)
        def arithmetic_check():
            require(weights.shape == (3,) and np.isfinite(weights).all() and np.all(weights >= 0)
                    and np.all(weights <= 1) and abs(weights.sum() - 1.) < 1e-12, 'H91 weights not on simplex')
            require(np.array_equal(saved['prediction'], p @ weights), 'H91 exact frozen weighted prediction mismatch')
            score, reference = metric(y, saved['prediction'], 'rmse'), metric(y, p[:, 0], 'rmse')
            close(score, result['score'], 'H91 RMSE')
            close(reference, result['reference_score'], 'H91 H70 reference')
            require(result['eligible_for_outer'] == bool(np.count_nonzero(weights) > 1 and score < reference)
                    and result['old_model_refits'] == result['learner_fits'] == 0
                    and result['parameter_fits'] == int(stage == 'inner'), 'H91 eligibility/fit accounting mismatch')
            return {'score': score, 'reference_H70': reference, 'relative_gain_vs_H70': (reference - score) / reference,
                    'weights': weights.tolist(), 'learner_fits': 0}
        checks.run('multiblend_exact_prediction_and_metric', arithmetic_check)
        def selection_check():
            if stage == 'inner':
                optimal_mse, exact_weights = simplex_optimum(p, y)
                stored_mse = float(np.mean((p @ weights - y) ** 2))
                require(stored_mse <= optimal_mse + 1e-7 * max(1., optimal_mse), 'H91 SLSQP result is worse than independently enumerated simplex optimum')
                records = read_json(out / 'model-fit-history.json')
                require(len(records) == 1, 'H91 inner parameter fit not uniquely registered')
                verify_claim(records[0], configuration, pd.DataFrame(p, columns=configuration['components']), y, 'inner')
                require(records[0]['fingerprint'] == result['fingerprint'] and records[0]['ledger_sha256'] == evidence['ledger_sha256'], 'H91 parameter claim identity differs')
                return {'independent_optimal_mse': optimal_mse, 'independent_weights': exact_weights.tolist()}
            inner = root / 'results/2026-10-24/HYP-091-01-inner-multiblend-sprint'
            selected = read_json(inner / 'metrics.json')
            require(np.array_equal(weights, np.asarray(selected['weights'])) and selected['eligible_for_outer'] is True
                    and result['fingerprint'] is None and not (out / 'model-fit-history.json').exists(), 'H91 outer weights were refitted or differ from frozen inner')
            matches = [r for r in prior_runs if safe_path(root, r['directory']) == inner]
            require(len(matches) == 1 and matches[0].get('verified') is True, 'H91 inner selection not independently verified')
            outer_fit_started = read_json(sources[2] / 'evidence.json')['started']
            require(datetime.fromisoformat(selected['finished']) < datetime.fromisoformat(outer_fit_started)
                    <= datetime.fromisoformat(evidence['started']) and sha256(inner / 'source.py') == evidence['code_sha256'],
                    'H91 weights/source not frozen before H85 outer fit')
            return {'inner_weights_exact': True, 'weights_frozen_before_H85_outer_fit': True,
                    'comparison_reference': 'H70; 20pct target separately uses H69 original outer'}
        checks.run('multiblend_independent_selection_and_freeze', selection_check)
        report['verified'] = all(row['passed'] for row in checks.rows)
    except Exception as exc:
        checks.rows.append({'name': 'required_multiblend_artifact_or_structure', 'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)})
    return report


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


def inherited_overlay(root, stage, fit, valid, saved, configuration, audit, teacher_audit, prior_runs=None, use_transport=True):
    source = root / ('results/2026-10-24/HYP-070-03-inner-high-sprint' if stage == 'inner'
                     else 'results/2026-10-24/HYP-070-04-outer-high-sprint')
    require(safe_path(root, teacher_audit['inherited_baseline_source']) == source, 'Incorrect inherited H70 source')
    inherited = arrays(source / 'validation_predictions.npz')
    split = arrays(source / 'splits.npz')
    require(np.array_equal(fit, split['fit']) and np.array_equal(valid, split['valid'])
            and np.array_equal(indices(inherited['positions'], 'inherited positions'), valid)
            and np.array_equal(inherited['y'], saved['y']), 'Inherited H70 source split/target mismatch')
    actual = {'transport_state_sha256': sha256(source / 'audit.json'),
              'inherited_prediction_sha256': hashlib.sha256(inherited['prediction'].tobytes()).hexdigest()}
    require(all(configuration[k] == digest == teacher_audit[k] for k, digest in actual.items()), 'Overlay inherited source SHA mismatch')
    state = read_json(source / 'audit.json')
    require(audit['base_refit'] is False, 'Inherited H70 baseline was refitted')
    if use_transport:
        require(audit['rating_beta'] == state['rating_beta'] and audit['group_rating_betas'] == state['group_rating_betas']
                and audit['transport_state_fitted_here'] is False and audit['transport_state_supplied'] is True,
                'Overlay transport state differs or was refitted')
    else:
        require('rating_beta' not in audit and 'group_rating_betas' not in audit,
                'Prediction-only expert unexpectedly reports transport beta')
    if prior_runs is not None:
        matches = [r for r in prior_runs if safe_path(root, r['directory']) == source]
        require(len(matches) == 1 and matches[0].get('verified') is True, 'Inherited H70 source not independently verified')
    source_score = metric(inherited['y'], inherited['prediction'], 'rmse')
    score = metric(saved['y'], saved['prediction'], 'rmse')
    return {'verified_extra_fields': sorted(actual), 'source': str(source.relative_to(root)), 'hashes': actual,
            'incremental_H70_control_score': source_score, 'overlay_score': score,
            'incremental_relative_gain_vs_H70': (source_score - score) / source_score,
            'incremental_improvement_over_H70': score < source_score,
            'worker_reference_remains_H69': True, 'transport_state_used': use_transport}


def verify_claim(record, configuration, X, y, stage):
    identity = {'recipe': semantic(configuration), 'X_sha256': data_digest(X),
                'y_sha256': data_digest(np.asarray(y).reshape(-1, 1))}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    require(all(record.get(k) == value for k, value in identity.items()) and record['fingerprint'] == fingerprint
            and record['status'] == 'completed' and record['stage'] == stage and record.get('ledger_sha256'),
            'Actual learner configuration/data/registered fit identity mismatch')
    return fingerprint


def calibration_provenance(root, stage, configuration, audit, teacher_audit, classes, snapshot, prior_runs):
    source = root / ('results/2026-10-24/HYP-078-01-inner-default-sprint/audit.json' if stage == 'inner'
                     else 'results/2026-10-24/HYP-078-02-outer-default-sprint/audit.json')
    require(safe_path(root, teacher_audit['calibration_state_source']) == source
            and teacher_audit['calibration_state_sha256'] == sha256(source), 'Calibration state source/SHA mismatch')
    state = read_json(source)
    theta = np.asarray(state['theta'], float)
    require(theta.shape == (len(classes) + 1,) and np.isfinite(theta).all()
            and configuration['calibration_state'] == {'theta': state['theta']}, 'Calibration theta differs from fixed H78 state')
    require(audit['learner_fits'] == 0 and audit['calibration_refitted'] is False
            and audit['query_target_used'] is False and configuration['query_labels_used'] is False,
            'Cohort calibration/query-label audit mismatch')
    tree = ast.parse((snapshot / 'source/sprint_cohort.py').read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CohortMacroDecision')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ['predict', 'predict_proba', '_p']]
    require(all(all(arg.arg not in ['y', 'target', 'labels', 'query_labels'] for arg in n.args.args) for n in methods)
            and not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ['fit', 'fit_transform', 'partial_fit']
                        for method in methods for n in ast.walk(method)), 'Query code exposes labels or learner fitting')
    matches = [r for r in prior_runs if safe_path(root, r['directory']) == source.parent]
    require(len(matches) == 1 and matches[0].get('verified') is True, 'H78 calibration source is not independently verified')
    return {'constructor_parameters': {'calibration_state': {'theta': state['theta']}},
            'source_sha256': sha256(source), 'query_labels_absent_from_query_API_and_no_fit_calls': True,
            'query_cohort_statistics_used': True, 'batch_invariant': False}


def expert_frame(X):
    """Independent H29/H30/H31 subset preprocessing; no transformer/learner fit."""
    frame = X.copy()
    def numeric(column):
        return pd.to_numeric(frame[column], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
    for column in ['direction_max_wind_speed', 'direction_peak_wind_speed']:
        if column in frame:
            angles = np.deg2rad(numeric(column).to_numpy())
            frame[column + '__sin'], frame[column + '__cos'] = np.sin(angles), np.cos(angles)
    months = 'january february march april may june july august september october november december'.split()
    monthly = next(([m + suffix for m in months] for suffix in ['_avg_temp_c', '_avg_temp']
                    if all(m + suffix in frame for m in months)), [])
    if monthly:
        matrix = np.column_stack([numeric(c).fillna(float(numeric(c).median()) if numeric(c).notna().any() else 0.).to_numpy() for c in monthly])
        angles = 2 * np.pi * np.arange(12) / 12
        for harmonic in [1, 2]:
            frame[f'monthly_temp__sin{harmonic}'] = matrix @ np.sin(harmonic * angles) / 6
            frame[f'monthly_temp__cos{harmonic}'] = matrix @ np.cos(harmonic * angles) / 6
    for column in ['floor_area', 'energy_star_rating']:
        if column not in frame:
            continue
        value = numeric(column)
        finite = value.dropna()
        medians = np.full(len(frame), float(finite.median()) if len(finite) else 0.)
        iqrs = np.full(len(frame), float(finite.quantile(.75) - finite.quantile(.25)) if len(finite) else 0.)
        # Broad facility fallback first, more specific state/facility overrides it.
        for columns in [['facility_type'], ['State_Factor', 'facility_type']]:
            if not all(c in frame for c in columns):
                continue
            key = pd.Series('', index=frame.index, dtype=object)
            for c in columns:
                tokens = ('V:' + frame[c].astype('string')).fillna('M:').astype(str)
                key = key + tokens.str.len().astype(str) + ':' + tokens
            groups = pd.DataFrame({'key': key.to_numpy(), 'value': value.to_numpy()}).groupby('key', sort=False)['value']
            eligible = groups.count() >= 100
            median_map, iqr_map = groups.median()[eligible], (groups.quantile(.75) - groups.quantile(.25))[eligible]
            local_median = key.map(median_map).to_numpy(float)
            local_iqr = key.map(iqr_map).to_numpy(float)
            present = np.isfinite(local_median)
            medians[present], iqrs[present] = local_median[present], local_iqr[present]
        frame[column + '__group_relative'] = (value.to_numpy() - medians) / np.maximum(iqrs, 1e-6)
    categorical = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
    for column in frame:
        frame[column] = (('V:' + frame[column].astype('string')).fillna('M:').astype(str)
                         if column in categorical else pd.to_numeric(frame[column], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan))
    return frame, categorical


def missing_expert_provenance(bundle, fit, valid, saved, configuration, audit, out, stage, inherited_prediction, snapshot=None):
    rating = pd.to_numeric(bundle.X.iloc[fit]['energy_star_rating'], errors='coerce').to_numpy(float)
    mask = ~np.isfinite(rating)
    local = np.flatnonzero(mask)
    require(np.array_equal(np.asarray(audit['expert_fit_mask']), mask)
            and np.array_equal(np.asarray(audit['expert_fit_positions']), local)
            and np.array_equal(np.asarray(audit['expert_fit_original_indices']), bundle.X.iloc[fit].index[local])
            and audit['expert_fit_rows'] == len(local) and audit['removed_rows'] == 0,
            'Missing-rating actual training subset/mask/indices mismatch')
    require(audit['expert_fit_count'] == int(len(local) >= 300) and audit['base_refit'] is False,
            'Missing-rating expert fit count/base refit mismatch')
    if len(local) >= 300:
        records = read_json(out / 'expert-fits.json')
        require(len(records) == 1, 'Missing expert must have exactly one actual learner claim')
        if snapshot is not None:
            launcher = ast.parse((snapshot / 'source/experiments/sprint_run.py').read_text(encoding='utf8'))
            declarations = [node for node in ast.walk(launcher) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == 'env' and node.func.attr == 'update']
            require(any(all(any(keyword.arg == name and isinstance(keyword.value, ast.Constant) and str(keyword.value.value) == '4'
                                    for keyword in node.keywords) for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'])
                        for node in declarations), 'H90 frozen worker BLAS environment is not explicitly four threads')
        # The launcher fixes BLAS=4. Fourier GEMV reductions differ at last bits
        # under another BLAS thread count, which changes pandas' exact X digest.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            frame, categorical = expert_frame(bundle.X.iloc[fit].iloc[local])
        plain = {k: v for k, v in configuration.items() if k not in ['transport_state_sha256', 'inherited_prediction_sha256']}
        expected = {**plain, 'stage': 'missing_rating_actual_learner', 'training_subset_rows': len(local), 'feature_columns': list(frame.columns)}
        verify_claim(records[0], expected, frame, np.asarray(bundle.y)[fit][local], stage)
        require(audit['categorical_columns'] == categorical and audit['feature_audit'] == {'numbers': [29, 30, 31], 'fit_rows': len(local), 'target_used': False},
                'Missing expert subset-only feature preprocessing audit mismatch')
    query_mask = ~np.isfinite(pd.to_numeric(bundle.X.iloc[valid]['energy_star_rating'], errors='coerce').to_numpy(float))
    require(np.array_equal(saved['prediction'][~query_mask], inherited_prediction[~query_mask]),
            'Known-rating rows changed instead of keeping inherited H70')
    prediction_audit = read_json(out / 'prediction-audit.json')
    require(prediction_audit['query_missing_rows'] == int(query_mask.sum())
            and prediction_audit['expert_applied_rows'] == int(query_mask.sum() if len(local) >= 300 else 0)
            and prediction_audit['known_rating_unchanged'] is True, 'Missing expert query gate audit mismatch')
    return {'actual_subset_rows': len(local), 'actual_learner_claims': int(len(local) >= 300),
            'X_y_and_configuration_fingerprints_exact': True, 'known_rating_H70_predictions_exact': True,
            'H70_transport_beta_used': False, 'feature_reconstruction_BLAS_threads': 4}


def honest_teacher_provenance(root, bundle, stage, fit, valid, audit, snapshot, out):
    """Assemble H88 independently of its provider; never load or fit its models."""
    from sklearn.model_selection import StratifiedGroupKFold
    reused = audit['protocol'] in ['H92_94_reused_honest_raw_teacher', 'H92_95_reused_honest_raw_teacher']
    require(audit['teacher'] == 'raw_xgboost_grouped_oof' and (audit['protocol'] == 'H88_honest_raw_context' or reused)
            and audit['independent_training_oof'] is True and audit['H48_meta_state_used'] is False
            and audit['old_model_refits'] == 0, 'H88 honest teacher scope/refit audit mismatch')
    evidence_out, evidence_snapshot = out, snapshot
    if reused:
        require(audit['reused_H88_teachers'] is True and audit['new_teacher_fits'] == audit['model_fits'] == 0
                and not (out / 'teacher-fits.json').exists(), 'Honest reuse unexpectedly creates teacher claims/fits')
        evidence_out = root / 'results/2026-10-24/HYP-088-01-inner-default-sprint'
        evidence_snapshot = root / SUITE / 'HYP-088'
        if stage == 'inner':
            manifest = evidence_snapshot / 'frozen.json'
            require(safe_path(root, audit['historical_frozen_manifest']) == manifest
                    and audit['historical_frozen_manifest_sha256'] == sha256(manifest), 'Historical H88 frozen manifest path/SHA differs')
    context_split = arrays(root / CLASS_OUTER / 'splits.npz')
    outer_fit, outer_valid = partition(context_split['fit'], context_split['valid'], len(bundle.y), full=True)
    raw_folder, classes, original_oof, original_folds = read_oof(root, bundle, outer_fit, 'raw', audit['raw_component'])
    position_hash = lambda values: hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()
    require(audit['fit_positions_sha256'] == position_hash(fit) and audit['valid_positions_sha256'] == position_hash(valid),
            'H88 global fit/query position hashes differ')
    query_path = safe_path(root, audit['query_artifact'])
    require(sha256(query_path) == audit['query_artifact_sha256'], 'H88 query artifact SHA mismatch')
    if stage == 'outer':
        require(np.array_equal(fit, outer_fit) and np.array_equal(valid, outer_valid)
                and query_path == root / 'results/2026-10-23' / OUTER_COMPONENTS['raw'] / 'validation_predictions.npz'
                and audit['new_teacher_fits'] == audit['model_fits'] == 0 and not audit['nested_folds'],
                'H88 outer must reuse exact original raw OOF/query with zero new fits')
        query = arrays(query_path, ['positions', 'classes', 'probabilities'])
        require(np.array_equal(query['positions'], valid) and np.array_equal(query['classes'], classes), 'H88 outer RAW query alignment differs')
        train, query_p = original_oof, probabilities(query['probabilities'], classes, len(valid))
    else:
        first = original_folds[0]
        require(np.array_equal(fit, first['fit_positions']) and np.array_equal(valid, first['valid_positions']), 'H88 inner context differs from original fold0')
        require(query_path == raw_folder / 'fold-0.npz' and np.array_equal(arrays(query_path, ['valid_positions'])['valid_positions'], valid),
                'H88 inner query source differs from original held-out raw fold0')
        query_p = probabilities(first['probabilities'], classes, len(valid))
        X, y = bundle.X.iloc[fit], np.asarray(bundle.y)[fit]
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        expected = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(X, y, groups))
        train = np.full((len(fit), len(classes)), np.nan)
        seen = np.zeros(len(fit), dtype=bool)
        events = audit['nested_folds']
        if not events and audit.get('reused_historical_nested_cache'):
            _, nested_classes, train, _ = read_oof(root, bundle, fit, 'raw', audit['reused_historical_nested_cache'])
            require(np.array_equal(classes, nested_classes) and audit['new_teacher_fits'] == 0, 'H88 reused nested historical class/refit mismatch')
        else:
            records = read_json(evidence_out / 'teacher-fits.json')
            require(len(events) == 3 and len(records) == 3, 'H88 needs exactly three separately registered nested fits')
            frozen = read_json(evidence_snapshot / 'frozen.json')['sources']
            for number, (event, (local_fit, local_valid)) in enumerate(zip(events, expected)):
                partition(local_fit, local_valid, len(fit), full=True)
                require(not set(groups[local_fit]) & set(groups[local_valid]) and not seen[local_valid].any(), 'H88 nested groups/OOF rows overlap')
                started_path, outcome_path, artifact = [safe_path(root, event[key]) for key in ['started_evidence', 'outcome', 'artifact']]
                require(started_path.parent == outcome_path.parent == artifact.parent == evidence_out / 'honest-raw-oof', 'H88 nested artifacts escape owned run directory')
                started, outcome, saved = read_json(started_path), read_json(outcome_path), arrays(artifact)
                config = started['configuration']
                require(started['status'] == 'started' and started['classes'] == classes.tolist()
                        and outcome['status'] == 'completed' and event['fold'] == number
                        and outcome['fit_rows'] == event['fit_rows'] == len(local_fit)
                        and outcome['valid_rows'] == event['valid_rows'] == len(local_valid)
                        and outcome['prediction_sha256'] == event['prediction_sha256'] == sha256(artifact), 'H88 nested lifecycle or prediction SHA mismatch')
                require(np.array_equal(saved['fit'], local_fit) and np.array_equal(saved['valid'], local_valid)
                        and np.array_equal(saved['fit_positions'], fit[local_fit]) and np.array_equal(saved['valid_positions'], fit[local_valid])
                        and np.array_equal(saved['classes'], classes), 'H88 nested local/global positions/classes differ')
                wanted_parameters = {'n_estimators': 600, 'max_depth': 6, 'learning_rate': .05, 'reg_lambda': 1,
                                     'tree_method': 'hist', 'device': 'cpu', 'objective': 'multi:softprob'}
                require(config['parameters'] == wanted_parameters and config['component'] == 'raw'
                        and config['encoder'] == {'native': False, 'scale': False} and config['parameters_override'] == {}
                        and config['threads'] == 2 and config['nested_fold'] == number and config['stage'] == 'inner', 'H88 raw model/encoder parameters differ')
                require(config['fit_positions'] == fit[local_fit].tolist() and config['valid_positions'] == fit[local_valid].tolist()
                        and config['fit_positions_sha256'] == position_hash(fit[local_fit])
                        and config['valid_positions_sha256'] == position_hash(fit[local_valid]) and config['data_hashes'] == bundle.hashes,
                        'H88 pre-fit position/data hashes differ')
                require(set(config['source_hashes']) == {'automl_class_combo.py', 'automl_models.py', 'automl_combo_memory.py',
                                                        'sprint_contextual.py', 'sprint_honest.py', 'sprint_crossfit_cache.py'},
                        'H88 scientific source hash inventory incomplete')
                for filename, digest in config['source_hashes'].items():
                    matching = [value for key, value in frozen.items() if Path(key).name == filename]
                    require(len(matching) == 1 and matching[0] == digest, 'H88 nested scientific source SHA differs: ' + filename)
                require(config['X_sha256'] == data_digest(X.iloc[local_fit])
                        and config['y_sha256'] == data_digest(y[local_fit].reshape(-1, 1)), 'H88 pre-fit raw X/y digest differs')
                matches = [claim for claim in records if claim['recipe'] == semantic(config)]
                require(len(matches) == 1, 'H88 missing/duplicate nested actual fit claim')
                verify_claim(matches[0], config, X.iloc[local_fit], y[local_fit], 'inner')
                require(datetime.fromisoformat(started['started']) <= datetime.fromisoformat(matches[0]['started'])
                        <= datetime.fromisoformat(matches[0]['finished']) <= datetime.fromisoformat(outcome['finished']), 'H88 evidence was not saved before actual fit')
                split_artifact = arrays(artifact.parent / f'fold-{number}.splits.npz')
                require(all(np.array_equal(split_artifact[key], saved[key]) for key in ['fit', 'valid', 'fit_positions', 'valid_positions']),
                        'H88 pre-fit split differs from completed split')
                train[local_valid] = probabilities(saved['probabilities'], classes, len(local_valid))
                seen[local_valid] = True
            require(seen.all() and audit['new_teacher_fits'] == sum(not event['reused'] for event in events)
                    and audit['model_fits'] == audit['new_teacher_fits'], 'H88 incomplete OOF coverage/actual new fit count')
    # Provider normalizes the assembled matrix once more at its public boundary.
    train = probabilities(train, classes, len(fit))
    require(audit['teacher_train_sha256'] == hashlib.sha256(train.tobytes()).hexdigest(), 'H88 independently assembled training teacher SHA differs')
    return {'train': train, 'query': query_p, 'classes': classes,
            'detail': {'new_teacher_fits': audit['new_teacher_fits'], 'independent_training_oof': True,
                       'H48_meta_state_used': False, 'raw_query_is_separate_from_H48_comparison': True,
                       'reused_existing_H88_teachers': reused}}


def verify_run(root, hypothesis_directory, event, bundles, teachers, prior_runs=None):
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
        verified_extra_fields = []
        verified_constructor_parameters = {}
        if request['number'] in [81, 86, 90]:
            overlay = checks.run('inherited_H70_overlay_provenance_and_incremental_comparison',
                                lambda: inherited_overlay(root, event['stage'], fit, valid, saved, configuration, audit, teacher_audit, prior_runs,
                                                          use_transport=request['number'] != 90))
            if overlay is not None:
                verified_extra_fields = overlay['verified_extra_fields']
                report['incremental_H70_comparison'] = {k: overlay[k] for k in ['incremental_H70_control_score',
                    'overlay_score', 'incremental_relative_gain_vs_H70', 'incremental_improvement_over_H70']}
            if request['number'] == 90:
                source = safe_path(root, teacher_audit['inherited_baseline_source'])
                inherited = arrays(source / 'validation_predictions.npz', ['prediction'])['prediction']
                checks.run('missing_rating_actual_subset_learner_claim', lambda: missing_expert_provenance(
                    bundle, fit, valid, saved, configuration, audit, out, event['stage'], inherited, hypothesis_directory))
        if request['number'] == 87:
            calibrated = checks.run('frozen_H78_calibration_and_label_free_query', lambda: calibration_provenance(
                root, event['stage'], configuration, audit, teacher_audit, np.asarray(configuration['teacher_classes']), hypothesis_directory, prior_runs))
            if calibrated is not None:
                verified_constructor_parameters = calibrated['constructor_parameters']
        # Do not execute snapshot constructors whose bytes failed integrity checking.
        constructor = None
        if any(row['name'] == 'source_snapshot' and row['passed'] for row in checks.rows):
            constructor = checks.run('actual_configuration_vs_frozen', lambda: expected_configuration(hypothesis_directory, request, configuration, verified_extra_fields,
                                                                                                      verified_constructor_parameters))
        else:
            checks.run('actual_configuration_vs_frozen', lambda: require(False, 'Skipped unsafe mutated snapshot'))

        checks.run('fit_audit_configuration', lambda: audit_configuration(configuration, audit, constructor, verified_extra_fields)
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
                source_hashes = source_data_hashes(root, 'regression', source)
                expected = arrays(source / 'splits.npz')
                reference = arrays(source / 'validation_predictions.npz', ['positions', 'prediction'])
                require(np.array_equal(fit, expected['fit']) and np.array_equal(valid, expected['valid'])
                        and np.array_equal(valid, reference['positions']) and np.array_equal(saved['reference'], reference['prediction']),
                        'Regression reference/splits differ from compatible saved control')
                if event['stage'] == 'inner':
                    year = pd.to_numeric(bundle.X['Year_Factor']).to_numpy()
                    require(np.all(year[fit] < 4) and np.all(year[valid] == 4), 'Incorrect inner temporal split')
                return {'source': str(source.relative_to(root)), 'prediction_sha256': sha256(source / 'validation_predictions.npz'),
                        'source_data_sha256': source_hashes}
            checks.run('compatible_saved_regression_reference', regression_reference)
        else:
            def class_checks():
                teacher_key = event['stage']
                honest_protocol = teacher_audit.get('protocol') in ['H88_honest_raw_context',
                    'H92_94_reused_honest_raw_teacher', 'H92_95_reused_honest_raw_teacher']
                comparison_audit = (teacher_audit['comparison_baseline_audit'] if honest_protocol else
                                    {k: v for k, v in teacher_audit.items() if k not in ['calibration_state_source', 'calibration_state_sha256']})
                if teacher_key not in teachers:
                    teachers[teacher_key] = class_teacher(root, bundle, teacher_key, comparison_audit)
                teacher = teachers[teacher_key]
                # Re-validate each attempt's own provenance even when numerical teacher is cached.
                previous_audit = teachers.get(('audit', teacher_key))
                if previous_audit is None:
                    teachers[('audit', teacher_key)] = comparison_audit
                else:
                    require(previous_audit == comparison_audit, 'Per-run teacher provenance differs')
                require(np.array_equal(fit, teacher['fit']) and np.array_equal(valid, teacher['valid'])
                        and np.array_equal(saved['reference'], teacher['reference']), 'Teacher split/reference mismatch')
                classes = np.unique(np.asarray(bundle.y)[fit]).astype(str)
                require(np.array_equal(saved['classes'], classes) and np.array_equal(teacher['classes'], classes)
                        and configuration['teacher_classes'] == classes.tolist(), 'Candidate/teacher/training class order differs')
                p = probabilities(saved['probabilities'], saved['classes'], len(valid))
                require(np.array_equal(saved['prediction'], classes[p.argmax(axis=1)]), 'Prediction differs from probability argmax')
                train_probabilities = teacher['train']
                if honest_protocol:
                    require(teacher_audit['comparison_baseline'] == 'cached_H48_outer_or_RAW_inner', 'H88 comparison reference provenance differs')
                    if teacher_audit.get('reused_H88_teachers') is True:
                        h88_source = root / 'results/2026-10-24/HYP-088-01-inner-default-sprint'
                        matches = [r for r in prior_runs if safe_path(root, r['directory']) == h88_source]
                        require(len(matches) == 1 and matches[0].get('verified') is True, 'Reused H88 teacher source not independently verified')
                    honest = honest_teacher_provenance(root, bundle, event['stage'], fit, valid, teacher_audit, hypothesis_directory, out)
                    require(np.array_equal(honest['classes'], classes), 'H88 honest teacher class order mismatch')
                    train_probabilities = honest['train']
                    report['honest_teacher'] = honest['detail']
                require(configuration['teacher_train_sha256'] == hashlib.sha256(train_probabilities.tobytes()).hexdigest(), 'Training teacher probability SHA mismatch')
                if event['stage'] == 'inner':
                    require(result['inner_is_full_H48_pipeline_validation'] is False, 'Inner proxy misrepresented as H48 pipeline validation')
                report['teacher_scope'] = teacher['note']
                if honest_protocol:
                    report['teacher_scope'] = 'H88 grouped OOF RAW training teacher; held-out RAW query; separate cached H48 outer or RAW inner comparison'
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
                runs.append(verify_run(root, directory, event, bundles, teachers, runs))
        except Exception as exc:
            registry_errors.append({'directory': str(directory.relative_to(root)), 'error': type(exc).__name__ + ': ' + str(exc)})
    for analytical in sorted((root / 'results/2026-10-24').glob('HYP-079-*-inner-combination-sprint')):
        runs.append(verify_analytical_run(root, analytical, runs))
    for analytical in sorted((root / 'results/2026-10-24').glob('HYP-091-*-*-multiblend-sprint')):
        runs.append(verify_multiblend(root, analytical, runs))
    diagnostic_checks = Checks()
    for diagnostic in sorted((root / 'results/2026-10-24').glob('HYP-*-*-registration-preflight-failure')):
        def inspect_preflight(diagnostic=diagnostic):
            recorded = read_json(diagnostic / 'metrics.json')
            require(recorded['status'] == 'blocked_preflight' and recorded['scientific_fits'] == recorded['fit_claims'] == 0
                    and recorded['model_configuration_changed'] is False
                    and {p.name for p in diagnostic.iterdir()} == {'metrics.json', 'report.md'},
                    'Pre-registration technical failure contains scientific fits/claims/artifacts')
            return {'directory': str(diagnostic.relative_to(root)), 'scientific_status': 'not_evaluated',
                    'recorded_fits': 0, 'scientific_artifacts_present': False,
                    'unlogged_fit_absence_not_proven': True}
        diagnostic_checks.run(str(diagnostic.relative_to(root)), inspect_preflight)
    require(runs or registry_errors, 'No registered sprint attempts found')
    global_checks = Checks()
    controls = {}
    for name, source in [('rmse', REG_OUTER), ('macro_f1', CLASS_OUTER)]:
        def control_check(name=name, source=source):
            from automl_data import load_task
            task = 'regression' if name == 'rmse' else 'multiclass'
            bundle = load_task(root / 'data' / task / 'task.md')
            source_hashes = source_data_hashes(root, task, root / source)
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
            return {'source': str(source), 'score': score, 'predictions_sha256': sha256(root / source / 'validation_predictions.npz'),
                    'source_data_sha256': source_hashes}
        global_checks.run('original_' + name + '_control', control_check)
    completed = [r for r in runs if r['status'] == 'completed']
    summary = {'spec': 'SPEC011', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'verifier': str(Path(__file__).relative_to(ROOT)), 'verifier_sha256': sha256(__file__),
        'model_fits_performed': 0, 'runs': runs, 'registry_errors': registry_errors, 'global_checks': global_checks.rows,
        'pre_registration_diagnostic_checks': diagnostic_checks.rows,
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
    summary['passed'] = (summary['all_completed_runs_verified'] and summary['all_attempts_terminal']
                         and all(r['passed'] for r in global_checks.rows + diagnostic_checks.rows))
    summary['checks_total'] = sum(len(run['checks']) for run in runs) + len(global_checks.rows) + len(diagnostic_checks.rows)
    summary['checks_passed'] = sum(check['passed'] for run in runs for check in run['checks']) + sum(check['passed'] for check in global_checks.rows + diagnostic_checks.rows)
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
