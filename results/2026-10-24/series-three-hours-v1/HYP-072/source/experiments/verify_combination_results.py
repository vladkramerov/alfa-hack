"""Independent SPEC-008 artifact verification and reporting; never fits a model.

Frozen source *copies* are checked against the pre-fit manifest. Current root
production modules may legitimately change during a later winner promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold

from automl_data import load_task, split_key

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'results/2026-10-23/series-combinations-v1'
BASELINE = {'regression': 41.480963027907194, 'multiclass': .6286047899141216}
TRAIN_SHA = {'regression': '096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775',
             'multiclass': '171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0'}
SPLIT_KEYS = {'regression': '68ef03446e22982a2ce32859ef7bdf4139ff943c133f1c1de95394f92d7f66fe',
              'multiclass': '239200bc6da7d2190167f04a816e82fd7fc2c29ec36eca7f72c29ff1a19fef32'}
OUTER_COMPONENTS = {
    'regression': {'raw': 'HYP-012-01-regression-log1p_target', 'experts': 'HYP-035-01-regression-spec006',
                   'season': 'HYP-030-01-regression-spec006', 'relative': 'HYP-031-01-regression-spec006'},
    'multiclass': {'raw': 'HYP-011-02-multiclass-none', 'smote': 'HYP-038-01-multiclass-spec006',
                   'frequency': 'HYP-017-01-multiclass-frequency'}}
TERMINAL = {'completed', 'failed', 'timeout'}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temporary.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def score_saved_predictions(task, artifact, truth, valid_positions, expected_classes=None):
    """Recompute primary score, rejecting altered truth/order/probability labels."""
    truth, valid_positions = np.asarray(truth), np.asarray(valid_positions)
    with np.load(artifact, allow_pickle=False) as saved:
        _require(np.array_equal(saved['positions'], valid_positions), 'Saved validation positions mismatch')
        _require(np.array_equal(saved['y'].astype(str), truth.astype(str)), 'Saved truth differs from original targets')
        prediction = saved['prediction']
        _require(prediction.shape == truth.shape == (len(valid_positions),), 'Prediction shape mismatch')
        if task == 'regression':
            _require(np.isfinite(prediction).all(), 'Nonfinite regression prediction')
            _require(np.array_equal(saved['y'], truth), 'Numeric saved truth mismatch')
            value = float(np.sqrt(mean_squared_error(truth, prediction)))
        else:
            classes, probabilities = saved['classes'], saved['probabilities']
            _require(np.array_equal(classes, expected_classes), 'Classification classes/order mismatch')
            _require(probabilities.shape == (len(valid_positions), len(classes)), 'Probability shape mismatch')
            _require(np.isfinite(probabilities).all() and np.all(probabilities >= 0), 'Invalid probabilities')
            _require(np.allclose(probabilities.sum(axis=1), 1., atol=1e-10, rtol=0), 'Probability normalization mismatch')
            _require(np.array_equal(prediction, classes[np.argmax(probabilities, axis=1)]), 'Saved labels differ from argmax')
            value = float(f1_score(truth, prediction, average='macro', zero_division=0))
    _require(np.isfinite(value), 'Nonfinite recomputed score')
    return value


def promotable_best(table, verification_passed):
    """Export frozen state only for verified, strictly improving task winners."""
    if not verification_passed:
        return {}
    best = {}
    for task in BASELINE:
        rows = [row for row in table if row['task'] == task and row['status'] == 'completed'
                and row.get('verified') and row.get('improved')]
        if rows:
            winner = (min if task == 'regression' else max)(rows, key=lambda row: row['score'])
            best[task] = {key: winner[key] for key in ('hypothesis', 'task', 'metric', 'score', 'candidate',
                                                       'learned_state', 'directory', 'improved', 'verified')}
    return best


def _folds(task, X, y):
    if task == 'regression':
        years = pd.to_numeric(X['Year_Factor'], errors='raise').to_numpy()
        values = sorted(np.unique(years))
        return [(np.flatnonzero(years < value), np.flatnonzero(years == value)) for value in values[-2:]]
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    return list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, y, groups))


def _align(probabilities, source, classes):
    source, classes = list(source), list(classes)
    _require(set(source) == set(classes), 'Source class universe mismatch')
    result = np.asarray(probabilities, dtype=float)[:, [source.index(label) for label in classes]]
    _require(np.isfinite(result).all() and np.all(result >= 0), 'Invalid component probabilities')
    _require(np.all(result.sum(axis=1) > 0), 'Zero component probability row')
    return result / result.sum(axis=1, keepdims=True)


def _calibrate(probabilities, classes, mapping):
    weights = np.asarray([mapping[str(label)] for label in classes], dtype=float)
    _require(np.isfinite(weights).all() and np.all(weights > 0), 'Invalid learned multipliers')
    result = probabilities * weights
    return result / result.sum(axis=1, keepdims=True)


def _choose_multipliers(y, probabilities, classes):
    factors = np.ones(len(classes))
    score = float(f1_score(y, classes[np.argmax(probabilities, axis=1)], average='macro', zero_division=0))
    for index in range(1, len(classes)):
        best = factors[index]
        for value in (.75, 1., 1.25, 1.5, 2.):
            trial = factors.copy()
            trial[index] = value
            candidate = float(f1_score(y, classes[np.argmax(probabilities * trial, axis=1)], average='macro', zero_division=0))
            if candidate > score + 1e-12:
                score, best = candidate, value
        factors[index] = best
    return {str(label): float(factor) for label, factor in zip(classes, factors)}, score


def _meta_predict(probabilities, state, classes):
    _require(state['component_order'] == ['raw', 'smote', 'frequency'] and state['clip'] == 1e-6,
             'Frozen stack protocol mismatch')
    _require(np.array_equal(state['classes'], classes), 'Frozen stack classes mismatch')
    features = np.concatenate([np.log(np.clip(p, 1e-6, 1.)) for p in probabilities], axis=1)
    coef, intercept = np.asarray(state['coef']), np.asarray(state['intercept'])
    _require(coef.shape == (len(classes), features.shape[1]) and intercept.shape == (len(classes),),
             'Frozen multiclass stack dimensions mismatch')
    _require(np.isfinite(coef).all() and np.isfinite(intercept).all(), 'Nonfinite frozen meta state')
    logits = features @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


class Verification:
    def __init__(self):
        self.checks = []

    def check(self, name, condition, detail=None):
        item = {'check': name, 'passed': bool(condition)}
        if detail is not None:
            item['detail'] = str(detail)
        self.checks.append(item)
        return item['passed']

    def run(self, name, callback):
        try:
            callback()
        except Exception as exc:
            return self.check(name, False, type(exc).__name__ + ': ' + str(exc))
        return self.check(name, True)


def _outer_component(root, suite, task, component, valid, classes=None):
    path = root / 'results' / suite.parent.name / OUTER_COMPONENTS[task][component] / 'validation_predictions.npz'
    with np.load(path, allow_pickle=True) as saved:
        _require(np.array_equal(saved['positions'], valid), 'Old outer component positions mismatch')
        return saved['prediction'].astype(float) if task == 'regression' else _align(saved['probabilities'], saved['classes'], classes)


def _check_fold_artifact(path, task, X, y, outer_fit, number, classes=None):
    fit, valid = _folds(task, X, y)[number]
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in [('fit', fit), ('valid', valid), ('fit_positions', outer_fit[fit]), ('valid_positions', outer_fit[valid])]:
            _require(np.array_equal(saved[key], expected), 'Cache fold identity mismatch: ' + key)
        _require(not set(saved['fit_positions']) & set(saved['valid_positions']), 'Inner fit/valid positions overlap')
        if task == 'multiclass':
            groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
            _require(not set(groups[fit]) & set(groups[valid]), 'Inner raw duplicate groups overlap')
            _require(np.array_equal(saved['classes'].astype(str), classes.astype(str)), 'OOF classes mismatch')
            result = _align(saved['probabilities'], saved['classes'], classes)
            _require(result.shape == (len(valid), len(classes)), 'OOF probability shape mismatch')
            _require(np.array_equal(saved['prediction'].astype(str), classes[np.argmax(result, axis=1)].astype(str)),
                     'OOF prediction/probability inconsistency')
            return fit, valid, result
        years = X['Year_Factor'].to_numpy()
        _require(years[fit].max() < years[valid].min(), 'Inner temporal direction mismatch')
        result = saved['prediction'].copy()
        _require(result.shape == (len(valid),) and np.isfinite(result).all(), 'OOF regression prediction invalid')
        return fit, valid, result


def _verify_synthetic(event, X, y, outer_fit, expected_fit):
    branch = event['branch']
    _require(np.array_equal(branch['original_fit_positions'], outer_fit[expected_fit]), 'Synthetic source positions mismatch')
    synthetic = pd.read_csv(branch['synthetic_artifact'])
    with np.load(branch['selection_artifact'], allow_pickle=False) as saved:
        _require(np.array_equal(saved['original_fit_positions'], outer_fit[expected_fit]), 'Synthetic selection positions mismatch')
        _require(int(saved['accepted'].sum()) == len(synthetic) == branch['generated_accepted'], 'Synthetic acceptance counts mismatch')
        if len(synthetic):
            _require(np.array_equal(saved['generated_target'][saved['accepted']].astype(str),
                                    synthetic['__synthetic_target__'].astype(str).to_numpy()), 'Synthetic targets mismatch')
    counts = pd.Series(y[expected_fit]).value_counts()
    generated = synthetic['__synthetic_target__'].value_counts() if '__synthetic_target__' in synthetic else {}
    _require(all(label in counts and count <= min(int(counts[label]), 500) for label, count in generated.items()),
             'Synthetic class cap violated')
    original = X.iloc[expected_fit]
    for column in branch.get('domains', {}).get('rounded_integer_fields', []):
        values = synthetic[column].dropna().to_numpy(dtype=float)
        _require(np.allclose(values, np.rint(values), atol=1e-8, rtol=0), 'Fractional synthetic integer field: ' + column)
    joint = [column for column in ('proto', 'service', 'state') if column in original]
    known = set(map(tuple, original[joint].astype('string').fillna('__MISSING__').astype(str).to_numpy()))
    produced = synthetic[joint].astype('string').fillna('__MISSING__').astype(str).to_numpy()
    _require(all(tuple(row) in known for row in produced), 'Unobserved synthetic joint categories')
    for protocol, column in branch.get('domains', {}).get('tcp_zero_support_constraints', []):
        values = pd.to_numeric(synthetic.loc[synthetic['proto'].astype(str).str.lower().eq(protocol), column], errors='coerce').to_numpy()
        _require(np.isfinite(values).all() and np.all(np.abs(values) <= 1e-8), 'Synthetic TCP zero support violated')


def _verify_class_composition(root, suite, directory, number, audit, state, X, y, fit, valid, check):
    classes = np.unique(y)
    components = ['smote'] if number == 46 else ['raw', 'frequency'] if number == 47 else ['raw', 'smote', 'frequency']
    oof = {}
    expected_folds = _folds('multiclass', X, y)
    for component in components:
        component_audit = audit['components'][component]
        _require(len(component_audit['folds']) == 3, 'Expected three grouped inner folds')
        values = np.full((len(X), len(classes)), np.nan)
        for event in component_audit['folds']:
            fold = int(event['fold'])
            inner_fit, inner_valid, probability = _check_fold_artifact(event['artifact'], 'multiclass', X, y, fit, fold, classes)
            values[inner_valid] = probability
            if component != 'raw':
                _require(Path(event['model_artifact']).is_file(), 'New OOF model artifact missing')
            else:
                _require(event['status'] == 'reused', 'Legacy raw OOF must be reused')
                _require('series-twenty-hypotheses-v1' in str(event['artifact']), 'Legacy raw OOF source unexpected')
            if component == 'smote':
                _verify_synthetic(event, X, y, fit, expected_folds[fold][0])
            check('HYP-%03d %s inner-fold-%d exact raw grouped split' % (number, component, fold), True)
        _require(np.isfinite(values).all(), 'Incomplete classification OOF coverage')
        oof[component] = values
    outer = {component: _outer_component(root, suite, 'multiclass', component, valid, classes) for component in components}
    if number == 48:
        meta = state['meta_state']
        counts = pd.Series(y).value_counts()
        expected = {str(label): float(1 / np.sqrt(counts[label])) for label in classes}
        _require(audit['meta']['class_weight'] == expected, 'Meta inverse-sqrt training class weights mismatch')
        parameters = audit['meta']['configuration']['parameters']
        _require(parameters['C'] == .1 and parameters['solver'] == 'lbfgs' and parameters['max_iter'] == 1000,
                 'Frozen meta parameters mismatch')
        reconstructed = _meta_predict([outer[c] for c in ['raw', 'smote', 'frequency']], meta, classes)
        _require(audit['meta']['meta_fit_on_oof'] and not audit['meta']['meta_training_is_independent_oof_evaluation'],
                 'Meta training provenance mismatch')
    else:
        expected, inner_score = _choose_multipliers(y, sum(oof.values()) / len(components), classes)
        _require(state['class_multipliers'] == expected, 'Learned multiplier rule mismatch')
        _require(abs(audit['inner_macro_f1'] - inner_score) < 1e-12, 'Training OOF score mismatch')
        reconstructed = _calibrate(sum(outer.values()) / len(components), classes, expected)
    with np.load(directory / 'validation_predictions.npz', allow_pickle=False) as saved:
        _require(np.allclose(saved['probabilities'], reconstructed, atol=1e-12, rtol=1e-10), 'Outer composed probabilities mismatch')
    _require(audit['outer_model_fits'] == 0, 'Unexpected classification outer model fit')


def _verify_convex(root, suite, directory, audit, state, X, y, fit, valid, check):
    weights = np.asarray(state['blend_weights'], dtype=float)
    _require(weights.shape == (4,) and np.isfinite(weights).all() and np.all(weights >= 0)
             and abs(weights.sum() - 1.) < 1e-10, 'Invalid convex weights')
    folds = _folds('regression', X, y)
    for event in audit['legacy_oof']['folds']:
        _require(event['status'] == 'reused_read_only', 'Regression legacy OOF refitted')
        _check_fold_artifact(event['source'], 'regression', X, y, fit, int(event['fold']))
    _require(len(audit['components']) == 6, 'Expected six new regression inner component evaluations')
    for event in audit['components']:
        inner_fit, inner_valid = folds[int(event['fold'])]
        _require(np.array_equal(event['fit_positions'], fit[inner_fit]) and np.array_equal(event['valid_positions'], fit[inner_valid]),
                 'Regression temporal component positions mismatch')
        _require(Path(event['artifact']).is_file(), 'Regression inner component model missing')
        check('HYP-044 component-%d fold-%d temporal positions' % (event['component'], event['fold']), True)
    with np.load(audit['oof_artifact'], allow_pickle=False) as saved:
        matrix, fold_id = saved['prediction'], saved['fold_id']
        _require(np.array_equal(saved['fit_positions'], fit) and np.array_equal(saved['target'], y), 'Convex OOF targets/positions mismatch')
        _require(np.array_equal(saved['weights'], weights), 'Convex OOF weights mismatch')
        _require(saved['component_order'].tolist() == ['raw', 'experts', 'season', 'relative'], 'Convex component order mismatch')
        expected_ids = np.full(len(X), -1)
        for fold, (_, inner_valid) in enumerate(folds):
            expected_ids[inner_valid] = fold
            with np.load(audit['legacy_oof']['folds'][fold]['source'], allow_pickle=False) as legacy:
                _require(np.array_equal(matrix[inner_valid, 0], legacy['prediction']), 'Legacy regression OOF prediction altered')
        _require(np.array_equal(fold_id, expected_ids), 'Convex temporal OOF coverage mismatch')
        _require(matrix.shape == (len(X), 4) and np.isfinite(matrix[fold_id >= 0]).all(), 'Convex OOF matrix invalid')
        objective = float(np.mean([np.mean((matrix[fold_id == fold] @ weights - y[fold_id == fold]) ** 2) for fold in range(2)]))
        _require(abs(objective - audit['selection']['inner_mse']) < 1e-8, 'Convex equal-fold MSE mismatch')
        # Convex quadratic KKT conditions independently check the reported optimum.
        gradient = np.mean([2 * matrix[fold_id == fold].T @ (matrix[fold_id == fold] @ weights - y[fold_id == fold])
                            / np.sum(fold_id == fold) for fold in range(2)], axis=0)
        active = weights > 1e-5
        lagrange = float(np.mean(gradient[active]))
        tolerance = 1e-3 * max(1., float(np.max(np.abs(gradient))))
        _require(np.all(np.abs(gradient[active] - lagrange) <= tolerance) and np.all(gradient[~active] >= lagrange - tolerance),
                 'Convex weights fail independent KKT optimum check')
    outer = np.column_stack([_outer_component(root, suite, 'regression', component, valid)
                             for component in ['raw', 'experts', 'season', 'relative']])
    with np.load(directory / 'validation_predictions.npz', allow_pickle=False) as saved:
        _require(np.allclose(saved['prediction'], outer @ weights, atol=1e-10, rtol=1e-12), 'Convex outer reconstruction mismatch')


def verify(root=ROOT, suite=SUITE, registry=None):
    root, suite = Path(root), Path(suite)
    registry = Path(registry) if registry is not None else root / 'results/spec008-attempts.json'
    verification = Verification()
    check = verification.check
    rows = _json(registry) if registry.exists() else []
    frozen = _json(suite / 'frozen-plan.json')
    frozen_jobs = {job['number']: job for job in frozen['jobs']}
    check('exactly seven frozen graphs H42..H48', set(frozen_jobs) == set(range(42, 49)) and len(frozen['jobs']) == 7)
    check('seven registry attempts', len(rows) == 7)
    check('seven terminal outcomes', len(rows) == 7 and all(row['status'] in TERMINAL for row in rows))
    check('seven completed outcomes', len(rows) == 7 and all(row['status'] == 'completed' for row in rows))
    check('all planned hypothesis IDs present', {row['job']['number'] for row in rows} == set(range(42, 49)))
    check('attempt fingerprints unique', len({row['fingerprint'] for row in rows}) == len(rows))
    check('frozen plan prohibits baseline refits', frozen['baseline_refits'] == 0)
    for name, expected in frozen['sources'].items():
        verification.run('frozen source snapshot ' + name, lambda name=name, expected=expected: _require(_sha(suite / 'source' / name) == expected, 'Snapshot SHA mismatch'))
    bundles = {task: load_task(root / 'data' / task / 'task.md') for task in BASELINE}
    for task, bundle in bundles.items():
        check(task + ' original train SHA', bundle.hashes['train.csv'] == TRAIN_SHA[task])
    table = []
    for number in range(42, 49):
        matching = [row for row in rows if row['job']['number'] == number]
        task = 'regression' if number < 46 else 'multiclass'
        label = 'HYP-%03d' % number
        table_row = {'hypothesis': label, 'task': task, 'metric': 'rmse' if task == 'regression' else 'macro_f1',
                     'status': 'missing', 'score': None, 'recomputed_score': None, 'previous_best': BASELINE[task],
                     'improvement': None, 'improved': False, 'verified': False}
        if len(matching) != 1:
            check(label + ' exactly one attempt', False)
            table.append(table_row)
            continue
        record = matching[0]
        table_row.update(status=record['status'], directory=record.get('directory'), score=record.get('score'),
                         seconds=record.get('seconds'), error=record.get('error'))
        start_check = len(verification.checks)
        def inspect_record():
            directory = root / record['directory']
            bundle = bundles[task]
            _require(record['job'] == frozen_jobs[number], 'Registry job differs from frozen plan')
            _require(record['job']['reference'] == BASELINE[task], 'Incumbent reference mismatch')
            _require(record['hashes'] == bundle.hashes, 'Data hashes mismatch')
            _require(record['split_key'] == SPLIT_KEYS[task], 'Historical frozen split key mismatch')
            _require(record['sources'] == frozen['sources'], 'Attempt source manifest mismatch')
            _require(len(record['ledger_sha256_before_fit']) == 64, 'No read-before-fit ledger evidence')
            identity = {'task': task, 'candidate': record['job']['candidate'], 'hashes': record['hashes'], 'split_key': record['split_key']}
            expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            _require(record['fingerprint'] == expected, 'Attempt identity mismatch')
            for filename in ('report.md', 'evidence.json', 'request.json', 'metrics.json', 'splits.npz'):
                _require((directory / filename).is_file(), 'Required artifact missing: ' + filename)
            _require(_json(directory / 'evidence.json') == record, 'Evidence/registry mismatch')
            request = _json(directory / 'request.json')
            _require(request['job'] == frozen_jobs[number], 'Request/frozen job mismatch')
            metrics = _json(directory / 'metrics.json')
            _require(metrics['status'] == record['status'], 'Metrics/registry status mismatch')
            with np.load(directory / 'splits.npz', allow_pickle=False) as saved:
                fit, valid = saved['fit'], saved['valid']
            reference = root / 'results' / suite.parent.name / ('series-' + task + '-preprocessing-v1') / 'splits.npz'
            with np.load(reference, allow_pickle=False) as saved:
                _require(np.array_equal(fit, saved['outer_fit']) and np.array_equal(valid, saved['outer_valid']), 'Frozen outer split changed')
            _require(split_key(fit, valid) == record['split_key'], 'Outer split digest mismatch')
            _require(not set(fit) & set(valid), 'Outer rows overlap')
            X, y = bundle.X.iloc[fit], np.asarray(bundle.y)[fit]
            if task == 'regression':
                _require(np.all(bundle.X.iloc[fit]['Year_Factor'] < 5) and np.all(bundle.X.iloc[valid]['Year_Factor'] == 5), 'Regression outer year protocol mismatch')
            else:
                groups = pd.util.hash_pandas_object(bundle.X, index=False).to_numpy()
                _require(not set(groups[fit]) & set(groups[valid]), 'Outer raw duplicate groups overlap')
            if record['status'] != 'completed':
                return
            audit = _json(directory / 'audit.json')
            state_artifact = _json(directory / 'learned-state.json')
            state = metrics.get('learned_state', {})
            _require(all(state_artifact.get(key) == value for key, value in state.items()), 'Saved learned state mismatch')
            _require(record.get('learned_state', {}) == state, 'Registry learned state mismatch')
            _require(metrics['outer_baseline_refits'] == 0 and record['outer_baseline_refits'] == 0, 'Old outer baseline refit reported')
            score = score_saved_predictions(task, directory / 'validation_predictions.npz', np.asarray(bundle.y)[valid], valid,
                                            np.unique(y) if task == 'multiclass' else None)
            _require(abs(score - record['score']) < 1e-10 and abs(score - metrics['score']) < 1e-10, 'Independent metric mismatch')
            _require(metrics['fit_rows'] == len(fit) and metrics['valid_rows'] == len(valid), 'Outer row counts mismatch')
            improvement = BASELINE[task] - score if task == 'regression' else score - BASELINE[task]
            improved = improvement > 1e-10
            _require(record['improved'] == improved, 'Strict incumbent decision mismatch')
            table_row.update(recomputed_score=score, improvement=improvement, improved=improved)
            if number == 44:
                _verify_convex(root, suite, directory, audit, state, X, y, fit, valid, check)
            elif number >= 46:
                _verify_class_composition(root, suite, directory, number, audit, state, X, y, fit, valid, check)
            elif number in (42, 43):
                _require(audit['feature_numbers'] == ([30] if number == 42 else [29, 30, 31]), 'Expert feature composition mismatch')
                _require(audit['fit_retained_rows'] == len(fit) and audit['global_fit_rows'] == len(fit), 'Expert row retention mismatch')
                _require(audit['minimum_expert_rows'] == 500 and audit['shrinkage'] == 500., 'Expert thresholds mismatch')
                for expert in audit['experts']:
                    count = int(X['building_class'].astype(str).eq(expert['group']).sum())
                    _require(expert['fit_rows'] == count and abs(expert['weight'] - count / (count + 500.)) < 1e-12,
                             'Expert training count/shrinkage mismatch')
                    _require(expert['fitted'] == (count >= 500), 'Expert minimum sample decision mismatch')
            else:
                threshold = float(np.quantile(y, .8))
                _require(audit['threshold'] == threshold and audit['tail_quantile'] == .8, 'Tail threshold is not training 80th quantile')
                _require(audit['low_rows'] == int(np.sum(y <= threshold)) and audit['high_rows'] == int(np.sum(y > threshold)), 'Tail partition rows mismatch')
                _require(audit['fit_retained_rows'] == len(fit), 'Tail mixture removed rows')
            candidate = dict(record['job']['candidate'])
            candidate.update(metrics.get('candidate', {}))
            candidate.update(state)
            table_row.update(candidate=candidate, learned_state=state)
        verification.run(label + ' artifacts/protocol/independent score', inspect_record)
        table_row['verified'] = record['status'] == 'completed' and all(item['passed'] for item in verification.checks[start_check:])
        if not table_row['verified']:
            table_row.pop('learned_state', None)
            table_row.pop('candidate', None)
        table.append(table_row)
    attempts_path, outcomes_path = suite / 'model-fit-attempts.jsonl', suite / 'model-fit-outcomes.jsonl'
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding='utf-8').splitlines()] if attempts_path.exists() else []
    outcomes = [json.loads(line) for line in outcomes_path.read_text(encoding='utf-8').splitlines()] if outcomes_path.exists() else []
    attempt_counts = Counter(item['fingerprint'] for item in attempts)
    outcome_counts = Counter(item['fingerprint'] for item in outcomes)
    check('nonempty model fit journal', bool(attempts))
    check('all actual model fit fingerprints unique', all(count == 1 for count in attempt_counts.values()))
    check('exactly one outcome per actual model attempt', attempt_counts == outcome_counts)
    check('all model fits completed', all(item['status'] == 'completed' for item in outcomes))
    check('all model fits read ledger first', all(len(item.get('ledger_sha256', '')) == 64 for item in attempts))
    check('only frozen hypotheses caused model fits', all(item.get('hypothesis') in {'HYP-%03d' % n for n in range(42, 49)} for item in attempts))
    check('old full outer raw baseline models not refitted', not any(
        (item['rows'] == 35000 and item['configuration'].get('component') == 'regression-extra-trees'
         and item['configuration'].get('features') == []) or
        (item['rows'] == 70529 and item['configuration'].get('family') == 'xgboost') for item in attempts))
    passed = all(item['passed'] for item in verification.checks)
    completed = sum(row['status'] == 'completed' for row in rows)
    result = {'status': 'passed' if passed else 'failed', 'checks_passed': sum(item['passed'] for item in verification.checks),
              'checks_failed': sum(not item['passed'] for item in verification.checks), 'checks': verification.checks,
              'planned_evaluations': 7, 'attempted': len(rows), 'completed_evaluations': completed,
              'terminal_outcomes': sum(row['status'] in TERMINAL for row in rows), 'model_fit_attempts': len(attempts),
              'baseline_refits': 0 if verification.checks[-1]['passed'] else 'not_verified',
              'verified_at': datetime.now(timezone(timedelta(hours=3))).isoformat(), 'verifier_sha256': _sha(__file__),
              'source_verification': 'frozen copies against pre-fit manifest; current production intentionally not compared',
              'no_model_fit_performed': True}
    best = promotable_best(table, passed)
    best_observed = {}
    for task in BASELINE:
        values = [row for row in table if row['task'] == task and row['recomputed_score'] is not None]
        if values:
            row = (min if task == 'regression' else max)(values, key=lambda item: item['recomputed_score'])
            best_observed[task] = {key: row[key] for key in ('hypothesis', 'recomputed_score', 'improvement', 'verified')}
    comparison = {'status': result['status'], 'baseline': BASELINE, 'rows': table, 'best': best,
                  'best_observed': best_observed, 'promotion_requires': 'global verification passed and strict incumbent improvement >1e-10'}
    _save(suite / 'verification.json', result)
    _save(suite / 'comparison.json', comparison)
    _report(suite, result, comparison)
    return result, comparison


def _report(suite, result, comparison):
    lines = ['# SPEC-008: семь новых комбинаций HYP-042–048', '',
             f'Независимая сверка: **{result["status"]}**; успешных проверок {result["checks_passed"]}, ошибок {result["checks_failed"]}.',
             f'Завершённых оценок {result["completed_evaluations"]}/7; terminal outcomes {result["terminal_outcomes"]}/7; новых модельных fit {result["model_fit_attempts"]}.',
             'Сверка не запускала обучение. Проверены сохранённые исходники против frozen SHA, исходные targets, split/order, OOF и журнал fit.', '',
             '| Гипотеза | Задача | Статус | Метрика | Пересчитано | Incumbent | Улучшение | Проверено |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for row in comparison['rows']:
        score = '—' if row['recomputed_score'] is None else '%.12f' % row['recomputed_score']
        delta = '—' if row['improvement'] is None else '%+.12f' % row['improvement']
        lines.append('| %s | %s | %s | %s | %s | %.12f | %s | %s |' %
                     (row['hypothesis'], row['task'], row['status'], row['metric'], score, row['previous_best'], delta, row['verified']))
    lines += ['', 'Положительное улучшение: incumbent−новый RMSE; новый F1−incumbent. Строгий порог допуска >1e-10.', '']
    for row in comparison['rows']:
        if row.get('error'):
            lines.append(row['hypothesis'] + ': техническая ошибка `' + row['error'].replace('`', "'") + '`.')
    if comparison['best']:
        for task, winner in comparison['best'].items():
            lines.append('К переносу после верификации: **%s / %s = %.12f**; frozen candidate и learned state сохранены в comparison.json.' %
                         (winner['hypothesis'], task, winner['score']))
    else:
        lines.append('Подтверждённых кандидатов к переносу нет; действующий baseline сохраняется.')
    if result['checks_failed']:
        lines += ['', 'Непройденные проверки:']
        lines.extend('- ' + item['check'] + (': ' + item['detail'] if item.get('detail') else '')
                     for item in result['checks'] if not item['passed'])
    lines += ['', 'Это исследовательские сравнения на ранее просмотренном outer holdout; статистическая значимость и hidden platform score не установлены.',
              'Отрицательные исходы и технические ошибки сохранены. Незавершённая проверка не считается опровержением гипотезы.', '',
              '[Все проверки](verification.json), [сравнение и frozen winners](comparison.json), [план до fit](frozen-plan.json).']
    (suite / 'COMBINATIONS_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--registry', type=Path)
    args = parser.parse_args()
    result, comparison = verify(args.root, args.suite, args.registry)
    print(json.dumps({key: result[key] for key in ('status', 'checks_passed', 'checks_failed', 'completed_evaluations', 'model_fit_attempts')}, ensure_ascii=False))
    print(json.dumps({'best': comparison['best']}, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
