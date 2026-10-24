"""Fit-free independent verification of all ten frozen SPEC-009 evaluations."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.model_selection import StratifiedKFold

from automl_data import load_task, split_key
from experiments.verify_combination_results import (
    Verification, _json, _save, _sha, _require, _check_fold_artifact, _align,
    _meta_predict, _folds, score_saved_predictions, promotable_best, TRAIN_SHA, SPLIT_KEYS)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'results/2026-10-24/series-alternatives-v1'
ORIGINAL = '2026-10-23'
BASELINE = {'regression': 40.972997661684175, 'multiclass': .6495311620074445}
CASES = [(69, 'regression'), (49, 'multiclass'), (51, 'multiclass'), (53, 'multiclass'),
         (54, 'regression'), (55, 'regression'), (56, 'regression'), (58, 'regression'),
         (60, 'regression'), (60, 'multiclass')]
TERMINAL = {'completed', 'failed', 'timeout'}
PROFILE_FIELDS = ['State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built', 'ELEVATION']
PROFILE_NUMERIC = {'floor_area', 'year_built', 'ELEVATION'}


def _profile_tuples(X):
    _require(set(PROFILE_FIELDS).issubset(X.columns), 'Missing fixed profile field')
    columns = []
    for column in PROFILE_FIELDS:
        if column in PROFILE_NUMERIC:
            numeric = pd.to_numeric(X[column], errors='coerce')
            columns.append([('N', format(float(value), '.17g')) if pd.notna(value) and np.isfinite(value) else ('M', '') for value in numeric])
        else:
            columns.append([('M', '') if pd.isna(value) else ('V', str(value)) for value in X[column].astype('string')])
    return list(zip(*columns))


def past_profile_components(X, y, Xvalid):
    """Independent row-level lookup; imports no production memory implementation."""
    years = pd.to_numeric(X['Year_Factor'], errors='raise').to_numpy(dtype=float)
    query_years = pd.to_numeric(Xvalid['Year_Factor'], errors='raise').to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    records = {}
    for key, year, target in zip(_profile_tuples(X), years, y):
        records.setdefault(key, []).append((year, target))
    means, alphas, counts = np.zeros(len(Xvalid)), np.zeros(len(Xvalid)), np.zeros(len(Xvalid), dtype=int)
    for index, (key, year) in enumerate(zip(_profile_tuples(Xvalid), query_years)):
        eligible = [(past_year, target) for past_year, target in records.get(key, []) if past_year < year]
        if not eligible:
            continue
        past = np.asarray(eligible, dtype=float)
        latest = float(past[:, 0].max())
        decay = np.exp2(past[:, 0] - latest)
        means[index] = float(np.sum(decay * past[:, 1]) / np.sum(decay))
        counts[index] = len(past)
        alphas[index] = min(.9, len(past) / (len(past) + 1)) * np.exp(-.25 * max(0., year - latest - 1))
    return means, alphas, counts


def density_ratio_weights(probability, domain):
    counts = np.bincount(np.asarray(domain, dtype=int), minlength=2)
    _require(counts.min() > 0, 'Two nonempty temporal domains required')
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    values = np.clip(p / (1 - p) * counts[0] / counts[1], .2, 5.)
    return values / values.mean()


def independent_gamma(X):
    """Recreate median/OHE/scaling geometry numerically, without fitting an estimator."""
    frame = X.replace([np.inf, -np.inf], np.nan).copy()
    metadata = {}
    for column in frame:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = ('V:' + frame[column].astype('string')).fillna('M:')
        counts = values.value_counts(dropna=False)
        ordered = sorted(counts.index, key=lambda key: (-int(counts[key]), str(key)))
        kept = ordered if len(ordered) <= 128 else ordered[:127]
        frame[column] = values.where(values.isin(kept), 'R:').astype(object)
        metadata[column] = {'fit_distinct': len(ordered), 'kept': len(kept), 'rare_bucket': len(ordered) > len(kept)}
    rows = np.arange(len(X))
    if len(rows) > 512:
        rows = np.random.RandomState(42).choice(rows, 512, replace=False)
    numeric = frame.select_dtypes(include='number').columns.tolist()
    if numeric:
        values = frame[numeric].to_numpy(dtype=float)
        medians = frame[numeric].median().fillna(0.).to_numpy(dtype=float)
        values = np.where(np.isnan(values), medians[None, :], values)
        scales = values.std(axis=0)
        scales[scales == 0] = 1.
        standardized = (values[rows] - values.mean(axis=0)) / scales
        distances = pdist(standardized, metric='sqeuclidean')
    else:
        distances = np.zeros(len(rows) * (len(rows) - 1) // 2)
    for column in [column for column in frame if column not in numeric]:
        labels = frame[column].astype('string').fillna('__MISSING__').to_numpy()[rows]
        codes = pd.factorize(labels)[0].reshape(-1, 1)
        distances += 2. * (pdist(codes, metric='hamming') > 0)
    positive = distances[distances > 0]
    return (float(1. / np.median(positive)) if len(positive) else 1.), rows, metadata


def _baseline(root, valid):
    path = root / 'results' / ORIGINAL / 'HYP-043-01-regression-combinations/validation_predictions.npz'
    with np.load(path, allow_pickle=False) as saved:
        _require(np.array_equal(saved['positions'], valid), 'H43 original outer positions mismatch')
        return saved['prediction'].copy()


def _read_prediction(directory):
    with np.load(directory / 'validation_predictions.npz', allow_pickle=False) as saved:
        return saved['prediction'].copy()


def _parameters(config, expected):
    parameters = config['parameters']
    _require(all(parameters.get(key) == value for key, value in expected.items()), 'Frozen model parameters mismatch: ' + str(expected))


def _teacher(root, audit, X, y, fit):
    classes = np.unique(y)
    events = audit['teacher']['base_oof_artifacts']
    _require(len(events) == 9, 'Expected nine reused teacher base OOF artifacts')
    components = {name: np.full((len(X), len(classes)), np.nan) for name in ('raw', 'smote', 'frequency')}
    identities = []
    for event in events:
        name, fold = event['component'], int(event['fold'])
        _require(name in components and event['status'] == 'reused_read_only', 'Teacher component must be read-only')
        expected_suite = 'series-twenty-hypotheses-v1' if name == 'raw' else 'series-combinations-v1'
        _require(expected_suite in str(event['artifact']), 'Teacher source suite mismatch')
        _, valid, probability = _check_fold_artifact(event['artifact'], 'multiclass', X, y, fit, fold, classes)
        components[name][valid] = probability
        identities.append((name, fold))
    _require(len(set(identities)) == 9, 'Teacher fold duplicated')
    _require(all(np.isfinite(p).all() for p in components.values()), 'Incomplete teacher probability coverage')
    state = _json(root / 'results' / ORIGINAL / 'HYP-048-01-multiclass-combinations/learned-state.json')['meta_state']
    return _meta_predict([components[name] for name in ('raw', 'smote', 'frequency')], state, classes)


def _special(root, directory, number, task, audit, X, y, Xvalid, fit, valid, check):
    label = 'HYP-%03d/%s' % (number, task)
    prediction = _read_prediction(directory)
    if number == 69:
        means, alpha, counts = past_profile_components(X, y, Xvalid)
        baseline = _baseline(root, valid)
        with np.load(directory / 'profile-memory-components.npz', allow_pickle=False) as saved:
            _require(np.array_equal(saved['positions'], valid), 'Memory component positions mismatch')
            _require(np.allclose(saved['memory_mean'], means, atol=1e-10, rtol=1e-12), 'Independent past-only target memory mismatch')
            _require(np.allclose(saved['memory_weight'], alpha, atol=1e-12, rtol=1e-12), 'Independent temporal memory weight mismatch')
            _require(np.array_equal(saved['baseline'], baseline), 'Memory fallback is not original H43')
        _require(np.allclose(prediction, alpha * means + (1 - alpha) * baseline, atol=1e-10, rtol=1e-12), 'Memory prediction reconstruction mismatch')
        _require(audit['profile_fields'] == PROFILE_FIELDS and audit['strict_past_only'] and not audit['id_or_weather_as_key'], 'Memory fixed keys/provenance mismatch')
        _require(audit['matched_rows'] == int((alpha > 0).sum()) and audit['past_rows_sum'] == int(counts.sum()), 'Memory support audit mismatch')
        keys = _profile_tuples(X)
        collisions = Counter(zip(keys, X['Year_Factor'].to_numpy()))
        _require(audit['profile_count'] == len(set(keys)) and audit['same_profile_year_multiple_rows'] == sum(count > 1 for count in collisions.values()), 'Memory collision audit mismatch')
        check(label + ' independent row-past memory reconstruction', True)
    elif number == 55:
        levels = [.1, .3, .5, .7, .9]
        _require(audit['quantiles'] == levels and audit['weights'] == [.2] * 5, 'Frozen quantile levels/weights mismatch')
        with np.load(audit['quantile_validation_artifact'], allow_pickle=False) as saved:
            _require(np.array_equal(saved['positions'], valid) and np.array_equal(saved['quantiles'], levels), 'Quantile component identities mismatch')
            _require(saved['prediction'].shape == (len(valid), 5), 'Five quantile predictions required')
            _require(np.allclose(prediction, saved['prediction'].mean(axis=1), atol=1e-10, rtol=1e-12), 'Quantile mean reconstruction mismatch')
        _require(len(audit['components']) == 5, 'Five quantile model audits required')
        for event, level in zip(audit['components'], levels):
            _require(event['fit_rows'] == len(fit), 'Quantile model did not use all training rows')
            _parameters(event['configuration'], {'objective': 'quantile', 'alpha': level, 'n_estimators': 200,
                                                  'num_leaves': 31, 'learning_rate': .05, 'min_child_samples': 20, 'reg_lambda': 1.})
        check(label + ' five original-scale quantiles averaged', True)
    elif number == 56:
        folds = _folds('regression', X, y)
        expected_coverage = np.zeros(len(X), dtype=bool)
        _require(len(audit['base_oof_folds']) == 2 and audit['base_outer_reused'], 'Residual requires two new temporal OOF and reused outer')
        for event, (inner_fit, inner_valid) in zip(audit['base_oof_folds'], folds):
            _require(np.array_equal(event['fit_positions'], inner_fit) and np.array_equal(event['valid_positions'], inner_valid), 'Residual temporal fold local positions mismatch')
            _require(np.array_equal(event['original_fit_positions'], fit[inner_fit]) and np.array_equal(event['original_valid_positions'], fit[inner_valid]), 'Residual temporal fold global positions mismatch')
            _require(not set(inner_fit) & set(inner_valid) and X.iloc[inner_fit]['Year_Factor'].max() < X.iloc[inner_valid]['Year_Factor'].min(), 'Residual temporal leakage')
            _require(Path(event['model_artifact']).is_file(), 'Residual base fold model cache missing')
            expected_coverage[inner_valid] = True
        with np.load(audit['oof_artifact'], allow_pickle=False) as saved:
            _require(np.array_equal(saved['positions'], fit) and np.array_equal(saved['target'], y), 'Residual original training targets/positions mismatch')
            _require(np.array_equal(saved['residual_positions'], fit[expected_coverage]), 'Residual used non-OOF training rows')
            _require(np.isfinite(saved['prediction'][expected_coverage]).all() and np.isnan(saved['prediction'][~expected_coverage]).all(), 'Residual OOF coverage invalid')
            _require(np.allclose(saved['residual_target'], y[expected_coverage] - saved['prediction'][expected_coverage], atol=1e-10), 'Residual target not original y minus OOF')
            _require(np.array_equal(saved['outer_positions'], valid) and np.array_equal(saved['outer_base_prediction'], _baseline(root, valid)), 'Residual reused H43 prediction mismatch')
            _require(np.allclose(prediction, saved['outer_base_prediction'] + saved['outer_correction'], atol=1e-10, rtol=1e-12), 'Residual outer composition mismatch')
        _require(audit['correction_factor'] == 1. and not audit['residual_fit_is_independent_validation'], 'Residual training provenance mismatch')
        _parameters(audit['residual'], {'n_estimators': 200, 'max_depth': 3, 'num_leaves': 7, 'learning_rate': .03, 'min_child_samples': 20, 'reg_lambda': 10.})
        check(label + ' honest temporal residual targets and outer composition', True)
    elif number == 58:
        domain = (X['Year_Factor'].to_numpy() == X['Year_Factor'].max()).astype(int)
        expected_columns = [column for column in X if column != 'Year_Factor']
        _require(audit['domain_columns'] == expected_columns, 'Year_Factor remains in domain classifier')
        folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, domain))
        _require(len(audit['domain_folds']) == 3, 'Expected three domain folds')
        coverage = np.zeros(len(X), dtype=int)
        for event, (inner_fit, inner_valid) in zip(audit['domain_folds'], folds):
            _require(event['year_excluded'] and np.array_equal(event['fit_positions'], inner_fit) and np.array_equal(event['valid_positions'], inner_valid), 'Domain OOF fold mismatch')
            _require(np.array_equal(event['original_fit_positions'], fit[inner_fit]) and np.array_equal(event['original_valid_positions'], fit[inner_valid]), 'Domain global OOF positions mismatch')
            _require(not set(inner_fit) & set(inner_valid), 'Domain classifier self-prediction')
            _parameters(event['configuration'], {'n_estimators': 100, 'num_leaves': 15, 'learning_rate': .05, 'min_child_samples': 20, 'reg_lambda': 1.})
            coverage[inner_valid] += 1
        _require(np.all(coverage == 1), 'Domain OOF coverage must be exactly once')
        with np.load(audit['domain_weights_artifact'], allow_pickle=False) as saved:
            _require(np.array_equal(saved['positions'], fit) and np.array_equal(saved['domain'], domain), 'Domain labels/source positions mismatch')
            _require(saved['domain_columns'].tolist() == expected_columns, 'Domain feature schema mismatch')
            p = saved['domain_probability']
            _require(p.shape == (len(fit),) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), 'Invalid cross-fit domain probabilities')
            weights = density_ratio_weights(p, domain)
            _require(np.allclose(saved['weights'], weights, atol=1e-12, rtol=1e-12), 'Density ratio/clipping/normalization mismatch')
            actual_weights = saved['weights'].copy()
        config = audit['weighted_configuration']
        _require(config['weights_sha256'] == hashlib.sha256(actual_weights.astype('<f8').tobytes()).hexdigest(), 'Weighted ET fingerprint does not include actual weights')
        _parameters(config, {'n_estimators': 200, 'max_depth': 24, 'min_samples_leaf': 2, 'max_features': .8})
        check(label + ' cross-fit X-density ratio and weighted fit identity', True)
    elif number == 54:
        config = audit['configuration']
        _parameters(config, {'alpha': 10., 'solver': 'sparse_cg'})
        _require(config['n_knots'] == 5 and config['degree'] == 3 and config['knots'] == 'quantile'
                 and config['extrapolation'] == 'constant' and config['target_transform'] == 'none', 'Spline protocol mismatch')
        _require(all(value <= 128 for value in audit['categorical_level_counts'].values()), 'Spline category cap exceeded')
        numeric = X.replace([np.inf, -np.inf], np.nan).select_dtypes(include='number')
        _require(audit['numeric_medians'] == numeric.median().fillna(0.).to_dict(), 'Spline imputer statistics not train-only')
        check(label + ' spline/Ridge frozen protocol and training statistics', True)
    elif number in (49, 51, 53) or (number == 60 and task == 'multiclass'):
        with np.load(audit['training_lineage_artifact'], allow_pickle=False) as lineage:
            expected = np.tile(fit, 2) if number == 60 else fit
            _require(np.array_equal(lineage['original_fit_positions'], fit) and np.array_equal(lineage['source_fit_positions'], expected), 'Class training lineage mismatch')
            _require(not set(lineage['source_fit_positions']) & set(valid), 'Class training includes outer validation')
            _require(int(lineage['is_augmented'].sum()) == (len(fit) if number == 60 else 0), 'Class augmentation indicator mismatch')
            if number == 53:
                expected_landmarks = fit[np.random.RandomState(42).permutation(len(X))[:min(256, len(X))]]
                _require(np.array_equal(lineage['landmark_positions'], expected_landmarks), 'Nyström landmarks are not frozen fit-only sample')
                gamma, gamma_rows, metadata = independent_gamma(X)
                _require(np.array_equal(lineage['gamma_fit_positions'], fit[gamma_rows]), 'Gamma sampling not fit-only')
                _require(np.isclose(audit['gamma'], gamma, rtol=1e-8, atol=1e-14), 'Independent RBF gamma mismatch')
                _require(audit['rare_categories']['columns'] == metadata, 'Rare-category training count/cap mismatch')
                _parameters(audit['configuration'], {'C': 1., 'solver': 'lbfgs', 'max_iter': 1000, 'class_weight': None})
            elif number == 60:
                fields = [column for column in X.select_dtypes(include='number') if column not in {'proto', 'service', 'state', 'Year_Factor'}]
                mask = np.random.RandomState(42).random_sample((len(X), len(fields))) < .05
                _require(lineage['masked_numeric_fields'].tolist() == fields and np.array_equal(lineage['numeric_mask'], mask), 'Classification training mask mismatch')
                _require(audit['augmentation']['masked_cells'] == int(mask.sum()) and audit['fit_model_rows'] == 2 * len(fit), 'Classification masked row/cell count mismatch')
        if number == 49:
            teacher = _teacher(root, audit, X, y, fit)
            classes = np.unique(y)
            q = .7 * np.eye(len(classes))[np.searchsorted(classes, y)] + .3 * teacher
            with np.load(audit['teacher']['target_artifact'], allow_pickle=False) as saved:
                _require(np.array_equal(saved['fit_positions'], fit) and np.array_equal(saved['classes'], classes), 'Distillation target classes/positions mismatch')
                _require(np.array_equal(saved['hard_labels'], y.astype(str)), 'Distillation hard target changed')
                _require(np.allclose(saved['teacher_probabilities'], teacher, atol=1e-12, rtol=1e-10), 'Teacher frozen meta reconstruction mismatch')
                _require(np.allclose(saved['soft_targets'], q, atol=1e-12, rtol=1e-10), 'Distillation .7/.3 soft target mismatch')
                actual_q = saved['soft_targets'].copy()
            _require(audit['teacher']['teacher_refits'] == 0 and not audit['teacher']['teacher_is_independent_oof']
                     and audit['teacher']['meta_fitted_on_same_base_oof_targets'], 'Distillation teacher provenance misrepresented')
            _require(audit['configuration']['soft_targets_sha256'] == hashlib.sha256(actual_q.astype('<f8').tobytes()).hexdigest(), 'Distillation soft-target fingerprint mismatch')
        if number == 51:
            classes, counts = np.unique(y, return_counts=True)
            _require(audit['class_counts'] == dict(zip(map(str, classes), map(int, counts))), 'Balanced Softmax counts not original training counts')
            _require(np.allclose(audit['configuration']['log_count_shift'], np.log(counts), atol=1e-12), 'Balanced Softmax training log-count shift mismatch')
            _require(audit['training_shift'] == 'log_fit_class_count' and audit['inference_shift'] == 'none', 'Balanced Softmax shift applied at wrong stage')
        if number != 53:
            _parameters(audit['configuration'], {'n_estimators': 600, 'max_depth': 6, 'learning_rate': .05, 'reg_lambda': 1., 'tree_method': 'hist', 'device': 'cpu'})
        check(label + ' class protocol/fit lineage and specialized formulas', True)
    elif number == 60:
        fields = [column for column in X.select_dtypes(include='number') if column != 'Year_Factor']
        mask = np.random.RandomState(42).random_sample((len(X), len(fields))) < .05
        with np.load(audit['mask_artifact'], allow_pickle=False) as saved:
            _require(np.array_equal(saved['positions'], fit), 'Regression mask source positions mismatch')
            _require(saved['columns'].tolist() == fields and np.array_equal(saved['mask'], mask), 'Regression numeric training mask mismatch')
        _require(audit['masked_values'] == int(mask.sum()) and audit['fit_model_rows'] == 2 * len(fit), 'Regression augmentation size mismatch')
        _require('Year_Factor' not in fields and audit['protected_columns'] == ['Year_Factor'], 'Temporal key was masked')
        check(label + ' regression masked-copy lineage', True)
    else:
        raise ValueError('No specialized check for planned candidate')


def verify(root=ROOT, suite=SUITE, registry=None):
    root, suite = Path(root), Path(suite)
    registry = Path(registry) if registry is not None else root / 'results/spec009-attempts.json'
    rows = _json(registry) if registry.exists() else []
    frozen = _json(suite / 'frozen-plan.json')
    jobs = {(job['number'], job['task']): job for job in frozen['jobs']}
    verification = Verification()
    check = verification.check
    check('ten frozen cases and nine hypothesis IDs', set(jobs) == set(CASES) and len(frozen['jobs']) == 10 and len({n for n, _ in jobs}) == 9)
    check('ten registered cases', len(rows) == 10 and {(r['job']['number'], r['job']['task']) for r in rows} == set(CASES))
    check('ten terminal outcomes', len(rows) == 10 and all(r['status'] in TERMINAL for r in rows))
    check('all completed evaluations contain primary metrics', all(
        r.get('metric') in {'rmse', 'macro_f1'} and isinstance(r.get('score'), (int, float)) and np.isfinite(r['score'])
        for r in rows if r['status'] == 'completed'))
    check('registered attempts unique', len({r['fingerprint'] for r in rows}) == len(rows))
    check('frozen baseline validation refits forbidden', frozen['baseline_validation_refits'] == 0)
    for name, expected in frozen['sources'].items():
        verification.run('frozen source copy ' + name, lambda name=name, expected=expected: _require(_sha(suite / 'source' / name) == expected, 'Frozen source copy SHA mismatch'))
    bundles = {task: load_task(root / 'data' / task / 'task.md') for task in BASELINE}
    for task, bundle in bundles.items():
        check(task + ' original train SHA', bundle.hashes['train.csv'] == TRAIN_SHA[task])
    table = []
    for number, task in CASES:
        label = 'HYP-%03d/%s' % (number, task)
        matches = [row for row in rows if row['job']['number'] == number and row['job']['task'] == task]
        item = {'hypothesis': 'HYP-%03d' % number, 'task': task, 'status': 'missing',
                'metric': 'rmse' if task == 'regression' else 'macro_f1', 'score': None, 'recomputed_score': None,
                'previous_best': BASELINE[task], 'improvement': None, 'improved': False, 'verified': False}
        if len(matches) != 1:
            check(label + ' exactly one registered attempt', False)
            table.append(item)
            continue
        record = matches[0]
        item.update(status=record['status'], score=record.get('score'), directory=record.get('directory'),
                    seconds=record.get('seconds'), error=record.get('error'))
        before = len(verification.checks)
        def inspect():
            directory = root / record['directory']
            bundle = bundles[task]
            _require(record['job'] == jobs[number, task] and record['job']['reference'] == BASELINE[task], 'Job or incumbent differs from frozen plan')
            _require(record['hashes'] == bundle.hashes and record['split_key'] == SPLIT_KEYS[task], 'Original data/split digest mismatch')
            _require(record['sources'] == frozen['sources'], 'Attempt/frozen source manifests differ')
            _require(len(record['ledger_sha256_before_fit']) == 64, 'Ledger not read before evaluation')
            identity = {'task': task, 'candidate': record['job']['candidate'], 'hashes': record['hashes'], 'split_key': record['split_key']}
            _require(record['fingerprint'] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), 'Evaluation fingerprint mismatch')
            for filename in ('report.md', 'request.json', 'evidence.json', 'metrics.json', 'splits.npz'):
                _require((directory / filename).is_file(), 'Required attempt artifact absent: ' + filename)
            _require(_json(directory / 'evidence.json') == record, 'Evidence/registry differ')
            _require(_json(directory / 'request.json')['job'] == jobs[number, task], 'Request/frozen job differ')
            metrics = _json(directory / 'metrics.json')
            _require(metrics['status'] == record['status'], 'Metrics/registry status mismatch')
            with np.load(directory / 'splits.npz', allow_pickle=False) as saved:
                fit, valid = saved['fit'], saved['valid']
            reference = root / 'results' / ORIGINAL / ('series-' + task + '-preprocessing-v1') / 'splits.npz'
            with np.load(reference, allow_pickle=False) as saved:
                _require(np.array_equal(fit, saved['outer_fit']) and np.array_equal(valid, saved['outer_valid']), 'Original outer positions changed')
            _require(split_key(fit, valid) == SPLIT_KEYS[task] and not set(fit) & set(valid), 'Outer split changed or rows overlap')
            X, y, Xvalid = bundle.X.iloc[fit], np.asarray(bundle.y)[fit], bundle.X.iloc[valid]
            if task == 'regression':
                _require(np.all(X['Year_Factor'] < 5) and np.all(Xvalid['Year_Factor'] == 5), 'Outer regression temporal protocol mismatch')
            else:
                groups = pd.util.hash_pandas_object(bundle.X, index=False).to_numpy()
                _require(not set(groups[fit]) & set(groups[valid]), 'Outer duplicate groups overlap')
            if record['status'] != 'completed':
                _require(record['status'] in TERMINAL and record.get('error'), 'Unsuccessful case needs a preserved technical error')
                _require(record['conclusion'] == 'Недостаточно данных' and not record.get('improved', False),
                         'Technical failure must not be a negative scientific result or winner')
                _require(record.get('score') is None, 'Technical failure must not invent a metric')
                return
            _require(metrics['outer_baseline_refits'] == 0 and record['outer_baseline_refits'] == 0, 'Old outer baseline refitted')
            _require(metrics['fit_rows'] == len(fit) and metrics['valid_rows'] == len(valid), 'Model evaluation row counts mismatch')
            score = score_saved_predictions(task, directory / 'validation_predictions.npz', np.asarray(bundle.y)[valid], valid,
                                            np.unique(y) if task == 'multiclass' else None)
            _require(abs(score - record['score']) < 1e-10 and abs(score - metrics['score']) < 1e-10, 'Independent primary metric mismatch')
            delta = BASELINE[task] - score if task == 'regression' else score - BASELINE[task]
            _require(record['improved'] == (delta > 1e-10), 'Strict incumbent promotion decision mismatch')
            item.update(recomputed_score=score, improvement=delta, improved=delta > 1e-10)
            audit = _json(directory / 'audit.json')
            _special(root, directory, number, task, audit, X, y, Xvalid, fit, valid, check)
            state = metrics.get('learned_state', {})
            _require(state == _json(directory / 'learned-state.json') == record.get('learned_state', {}), 'Learned state artifacts mismatch')
            candidate = {**record['job']['candidate'], **metrics.get('candidate', {}), **state}
            item.update(candidate=candidate, learned_state=state)
        verification.run(label + ' original data/protocol/saved metric', inspect)
        item['integrity_verified'] = all(c['passed'] for c in verification.checks[before:])
        item['verified'] = record['status'] == 'completed' and item['integrity_verified']
        item['scientific_conclusion'] = ('Подтверждена в проверенных условиях' if item['improved'] else 'Не подтверждена по заданному критерию') if item['verified'] else 'Недостаточно данных'
        if not item['verified']:
            item.pop('candidate', None)
            item.pop('learned_state', None)
        table.append(item)
    attempts_path, outcomes_path = suite / 'model-fit-attempts.jsonl', suite / 'model-fit-outcomes.jsonl'
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding='utf-8').splitlines()] if attempts_path.exists() else []
    outcomes = [json.loads(line) for line in outcomes_path.read_text(encoding='utf-8').splitlines()] if outcomes_path.exists() else []
    attempt_counts, outcome_counts = Counter(a['fingerprint'] for a in attempts), Counter(a['fingerprint'] for a in outcomes)
    check('model/stateful fit journal present', bool(attempts))
    check('model/stateful fit fingerprints unique', all(count == 1 for count in attempt_counts.values()))
    check('exactly one terminal outcome per actual fit', attempt_counts == outcome_counts)
    check('all actual fit outcomes completed', all(a['status'] == 'completed' for a in outcomes))
    check('all fits read ledger first', all(len(a.get('ledger_sha256', '')) == 64 for a in attempts))
    allowed = {'HYP-%03d/%s' % case for case in CASES}
    check('only frozen cases caused fits', all(a.get('hypothesis') in allowed for a in attempts))
    for number, task in CASES:
        label = 'HYP-%03d/%s' % (number, task)
        events = [a for a in attempts if a.get('hypothesis') == label]
        if any(row['hypothesis'] == 'HYP-%03d' % number and row['task'] == task and row['status'] == 'completed' for row in table):
            expected = {(69, 'regression'): 1, (49, 'multiclass'): 1, (51, 'multiclass'): 1, (53, 'multiclass'): 1,
                        (54, 'regression'): 1, (55, 'regression'): 5, (58, 'regression'): 4,
                        (60, 'regression'): 1, (60, 'multiclass'): 1}
            if (number, task) in expected:
                check(label + ' exact fit count; no extra teacher/control fits', len(events) == expected[number, task])
            if number == 56:
                check(label + ' residual uses new temporal subsets and no H43 outer fit', bool(events) and all(
                    event['rows'] < 35000 for event in events))
            if number == 60:
                check(label + ' estimator trains on exactly two copies', len(events) == 1 and events[0]['rows'] == (70000 if task == 'regression' else 141058))
    check('known raw/H43 outer baseline ET not refitted', not any(a['rows'] == 35000 and
        a['configuration'].get('component') == 'regression-extra-trees' for a in attempts))
    passed = all(c['passed'] for c in verification.checks)
    result = {'status': 'passed' if passed else 'failed', 'checks_passed': sum(c['passed'] for c in verification.checks),
              'checks_failed': sum(not c['passed'] for c in verification.checks), 'checks': verification.checks,
              'planned_evaluations': 10, 'planned_hypotheses': 9, 'attempted': len(rows),
              'completed_evaluations': sum(r['status'] == 'completed' for r in rows),
              'technical_failures': sum(r['status'] in {'failed', 'timeout'} for r in rows),
              'terminal_outcomes': sum(r['status'] in TERMINAL for r in rows), 'model_fit_attempts': len(attempts),
              'no_model_fit_performed': True, 'baseline_validation_refits': 0 if verification.checks[-1]['passed'] else 'not_verified',
              'verified_at': datetime.now(timezone(timedelta(hours=3))).isoformat(), 'verifier_sha256': _sha(__file__),
              'shared_verification_helpers_sha256': _sha(Path(__file__).with_name('verify_combination_results.py')),
              'source_verification': 'frozen source copies against pre-fit manifest, not mutable current production'}
    comparison = {'status': result['status'], 'baseline': BASELINE, 'rows': table, 'best': promotable_best(table, passed)}
    _save(suite / 'verification.json', result)
    _save(suite / 'comparison.json', comparison)
    _report(suite, result, comparison)
    return result, comparison


def _report(suite, result, comparison):
    lines = ['# SPEC-009: первая волна альтернативных подходов', '',
             'Независимая сверка: **%s**, успешных проверок %d, ошибок %d.' % (result['status'], result['checks_passed'], result['checks_failed']),
             'Завершено %d/10 оценок девяти гипотез; terminal outcomes %d/10; зарегистрированных новых fit %d.' %
             (result['completed_evaluations'], result['terminal_outcomes'], result['model_fit_attempts']),
             'Verifier не обучал модели и не загружал исполняемые model/joblib artifacts. Сравнение основано на исходных targets и сохранённых результатах.', '',
             '| Гипотеза | Задача | Статус | Метрика | Пересчитано | Incumbent | Улучшение | Проверено |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for row in comparison['rows']:
        score = '—' if row['recomputed_score'] is None else '%.12f' % row['recomputed_score']
        delta = '—' if row['improvement'] is None else '%+.12f' % row['improvement']
        lines.append('| %s | %s | %s | %s | %s | %.12f | %s | %s |' %
                     (row['hypothesis'], row['task'], row['status'], row['metric'], score, row['previous_best'], delta, row['verified']))
    lines += ['', 'Положительное улучшение: incumbent−новый RMSE; новый F1−incumbent. Строгий порог >1e-10.', '']
    lines.append('Успешная integrity-сверка допускает сохранённые технические ошибки: завершённых численных метрик %d, технических исходов %d; они не подменяются отрицательной научной оценкой.' %
                 (result['completed_evaluations'], result['technical_failures']))
    for row in comparison['rows']:
        if row.get('error'):
            lines.append('%s/%s: техническая ошибка `%s`.' % (row['hypothesis'], row['task'], row['error'].replace('`', "'")))
    for task, winner in comparison['best'].items():
        lines.append('Проверенный кандидат к переносу: **%s / %s = %.12f**; candidate/learned_state в comparison.json.' %
                     (winner['hypothesis'], task, winner['score']))
    if not comparison['best']:
        lines.append('Проверенных победителей к переносу нет; baseline сохраняется.')
    if result['checks_failed']:
        lines += ['', 'Непройденные проверки:']
        lines.extend('- ' + check['check'] + (': ' + check['detail'] if check.get('detail') else '')
                     for check in result['checks'] if not check['passed'])
    lines += ['', 'Public outer уже просматривался; это разведочные результаты, не независимое подтверждение private качества или статистической значимости.',
              'H49 использует teacher meta, обученную на тех же training base-OOF targets: её training score не является честной OOF оценкой teacher.',
              'Цель десятикратного улучшения платформенного результата не считается достигнутой по локальному RMSE/F1; формула платформенного score не установлена.',
              'Ошибки/таймауты сохраняются и не считаются научным опровержением. Остальные proposed H49–68 не объявляются проверенными.', '',
              '[Проверки](verification.json), [сравнение и verified winners](comparison.json), [план до fit](frozen-plan.json).']
    (suite / 'ALTERNATIVES_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--registry', type=Path)
    args = parser.parse_args()
    result, comparison = verify(args.root, args.suite, args.registry)
    print(json.dumps({key: result[key] for key in ('status', 'checks_passed', 'checks_failed', 'completed_evaluations', 'model_fit_attempts')}, ensure_ascii=False))
    print(json.dumps({'best': comparison['best']}, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
