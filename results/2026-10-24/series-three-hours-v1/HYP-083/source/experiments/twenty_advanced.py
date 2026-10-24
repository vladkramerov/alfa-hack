"""Nested, train-only advanced procedures for SPEC-006 (HYP-028, 033..041).

No data are loaded and no models are fitted at import. Old outer predictions
are supplied by the caller; cached inner fits are shared across subprocesses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold


def inner_folds(task, X, y):
    if task == 'regression':
        years = pd.to_numeric(X['Year_Factor'], errors='raise').to_numpy()
        values = sorted(np.unique(years))
        if len(values) < 3:
            raise ValueError('Temporal nested validation requires at least three years')
        return [(np.flatnonzero(years < year), np.flatnonzero(years == year)) for year in values[-2:]]
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(X, y, groups))
    for fit, valid in folds:
        if set(groups[fit]) & set(groups[valid]):
            raise AssertionError('Raw duplicate group crosses nested fold boundary')
    return folds


def inner_fingerprint(task, X, y, positions, config):
    digest = hashlib.sha256()
    digest.update(json.dumps({'version': 2, 'task': task, 'config': config,
                              'columns': list(X.columns), 'dtypes': list(map(str, X.dtypes)),
                              'split': 'latest-two-expanding-years' if task == 'regression' else 'SGKF3-seed42-rawgroups'},
                             sort_keys=True).encode())
    for values in (pd.util.hash_pandas_object(X, index=False).to_numpy(),
                   pd.util.hash_pandas_object(pd.Series(np.asarray(y)), index=False).to_numpy(),
                   np.asarray(positions, dtype=np.int64)):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _make(task, config):
    from experiments.twenty_common import make_model
    kwargs = dict(config)
    feature_number = kwargs.pop('feature_number', None)
    if feature_number is not None:
        from experiments.twenty_features import make_features
        kwargs['feature'] = make_features(feature_number)
    return make_model(task, **kwargs)


def _fit(task, config, X, y):
    resolved = dict(config)
    ratio = resolved.pop('huber_slope_ratio', None)
    if ratio is not None:
        parameters = dict(resolved.get('parameters', {}))
        parameters.update(huber_slope=max(1., float(np.subtract(*np.percentile(y, [75, 25]))) * ratio),
                          base_score=float(np.median(y)))
        resolved['parameters'] = parameters
    return _make(task, resolved).fit(X, y)


def _score(task, y, pred):
    return -float(np.sqrt(mean_squared_error(y, pred))) if task == 'regression' else float(f1_score(y, pred, average='macro', zero_division=0))


def _aligned(probabilities, source, target):
    result = np.zeros((len(probabilities), len(target)))
    for index, label in enumerate(source):
        result[:, list(target).index(label)] = probabilities[:, index]
    return result


def _write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def cached_oof(task, X, y, context, config=None):
    """Persist each completed fold, including original positional identities."""
    config = {'policy': 'legacy'} if config is None else config
    y = np.asarray(y)
    positions = np.asarray(context.get('fit_positions', np.arange(len(X))))
    key = inner_fingerprint(task, X, y, positions, config)
    directory = Path(context['cache_dir']) / key
    directory.mkdir(parents=True, exist_ok=True)
    classes = np.unique(y) if task != 'regression' else np.array([], dtype=str)
    prediction = np.full(len(y), np.nan) if task == 'regression' else np.full(len(y), '', dtype=classes.dtype)
    probabilities = None if task == 'regression' else np.full((len(y), len(classes)), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    events = []
    for number, (fit, valid) in enumerate(inner_folds(task, X, y)):
        file = directory / f'fold-{number}.npz'
        if file.exists():
            with np.load(file, allow_pickle=False) as saved:
                if not np.array_equal(saved['fit'], fit) or not np.array_equal(saved['valid'], valid):
                    raise ValueError('Cache fold identity mismatch')
                pred = saved['prediction']
                proba = saved['probabilities'] if task != 'regression' else None
            status = 'reused'
        else:
            model = _fit(task, config, X.iloc[fit], y[fit])
            pred = np.asarray(model.predict(X.iloc[valid]))
            proba = None if task == 'regression' else _aligned(model.predict_proba(X.iloc[valid]), model.classes_, classes)
            _write_npz(file, fit=fit, valid=valid, fit_positions=positions[fit], valid_positions=positions[valid],
                       prediction=pred.astype(str) if task != 'regression' else pred,
                       probabilities=np.array([]) if proba is None else proba, classes=classes.astype(str))
            joblib.dump(model, directory / f'fold-{number}-model.joblib', compress=3)
            status = 'fitted'
        prediction[valid] = pred
        if proba is not None:
            probabilities[valid] = proba
        fold_id[valid] = number
        events.append({'fold': number, 'status': status, 'fit_rows': len(fit), 'validation_rows': len(valid), 'artifact': str(file)})
    audit = {'key': key, 'config': config, 'folds': events, 'selection_source': 'training_temporal_oof' if task == 'regression' else 'training_grouped_oof'}
    (directory / 'metadata.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return {'prediction': prediction, 'probabilities': probabilities, 'classes': classes, 'fold_id': fold_id, 'audit': audit}


def smearing_factor(y, prediction_log, fold_id=None):
    ratio = np.exp(np.log1p(np.asarray(y, dtype=float)) - np.asarray(prediction_log))
    factors = [float(ratio.mean())] if fold_id is None else [float(ratio[np.asarray(fold_id) == k].mean()) for k in np.unique(fold_id)]
    factor = float(ratio.mean())
    accepted = bool(np.isfinite(ratio).all() and .5 <= factor <= 3. and min(factors) > 0 and max(factors) / min(factors) <= 2.)
    return (factor if accepted else 1.), {'accepted': accepted, 'raw_factor': factor, 'fold_factors': factors,
                                        'rule': 'finite; global factor in [0.5,3]; max/min fold factors <=2'}


def apply_smearing(original_prediction, factor):
    return np.maximum(0., (np.asarray(original_prediction) + 1.) * factor - 1.)


def choose_multipliers(y, probabilities, classes):
    classes = np.asarray(classes)
    multipliers = np.ones(len(classes))
    score = _score('multiclass', y, classes[np.argmax(probabilities, axis=1)])
    # One fixed pass, five options/class; lexical class 0 anchors identifiability.
    for index in range(1, len(classes)):
        best = multipliers[index]
        for value in (.75, 1., 1.25, 1.5, 2.):
            candidate = multipliers.copy()
            candidate[index] = value
            candidate_score = _score('multiclass', y, classes[np.argmax(probabilities * candidate, axis=1)])
            if candidate_score > score + 1e-12:
                score, best = candidate_score, value
        multipliers[index] = best
    return multipliers, {'selection_source': 'training_grouped_oof', 'inner_macro_f1': score,
                         'multipliers': dict(zip(map(str, classes), multipliers.tolist())), 'fixed_anchor': str(classes[0]),
                         'grid': [.75, 1., 1.25, 1.5, 2.], 'passes': 1}


def confused_pair(y, prediction, classes):
    candidates = sorted(str(label) for label in classes if str(label).lower() not in {'normal', 'benign'})
    y, prediction = np.asarray(y), np.asarray(prediction)
    scored = [(int(np.sum((y == a) & (prediction == b)) + np.sum((y == b) & (prediction == a))), a, b)
              for i, a in enumerate(candidates) for b in candidates[i + 1:]]
    if not scored:
        raise ValueError('A pair of attack classes is required')
    best = sorted(scored, key=lambda value: (-value[0], value[1], value[2]))[0]
    return best[1:]


def redistribute_pair(probabilities, pair_indices, expert_probability, margin=.15):
    result = np.asarray(probabilities, dtype=float).copy()
    pair = np.asarray(pair_indices)
    expert = np.asarray(expert_probability)
    if expert.shape != (len(result), 2) or not np.allclose(expert.sum(axis=1), 1) or np.any(expert < 0):
        raise ValueError('Invalid pair expert probabilities')
    top = np.argsort(result, axis=1, kind='stable')[:, -2:]
    gate = np.all(np.sort(top, axis=1) == np.sort(pair), axis=1) & (np.abs(result[:, pair[0]] - result[:, pair[1]]) < margin)
    mass = result[:, pair].sum(axis=1)
    result[np.ix_(gate, pair)] = expert[gate] * mass[gate, None]
    return result, gate


def stable_feature_selection(columns, importance):
    importance = np.asarray(importance)
    # Delete only columns non-positive in every temporal/group fold; cap at20%.
    eligible = [i for i in range(len(columns)) if np.all(importance[:, i] <= 0.)]
    eligible.sort(key=lambda i: (float(importance[:, i].mean()), str(columns[i])))
    dropped = set(eligible[:int(.2 * len(columns))])
    return [column for i, column in enumerate(columns) if i not in dropped]


def smote_categorical_columns(X):
    result = []
    for column in X:
        values = X[column].dropna()
        if (not pd.api.types.is_numeric_dtype(X[column]) or str(column).lower() in {'proto', 'protocol'}
                or set(values.unique()).issubset({0, 1})):
            result.append(column)
    return result


def smote_strategy(y):
    labels, counts = np.unique(y, return_counts=True)
    return {label: int(count + min(count, 500)) for label, count in zip(labels, counts) if 6 <= count < counts.max()}


def _result(model, Xvalid, task, audit):
    prediction = np.asarray(model.predict(Xvalid))
    audit['model'] = getattr(model, 'audit_', {})
    result = {'prediction': prediction, 'audit': audit}
    if task != 'regression':
        result.update(probabilities=np.asarray(model.predict_proba(Xvalid)), classes=np.asarray(model.classes_))
    return result


def _reuse(context, prefix, audit):
    result = {'prediction': np.asarray(context[prefix + '_prediction']), 'audit': audit}
    if context.get(prefix + '_probabilities') is not None:
        result.update(probabilities=np.asarray(context[prefix + '_probabilities']), classes=np.asarray(context[prefix + '_classes']))
    audit['outer_fit'] = 'reused_saved_' + prefix
    return result


def load_outer_artifact(artifact, task, Xvalid, context, audit):
    with np.load(artifact, allow_pickle=False) as saved:
        if 'positions' not in saved or not np.array_equal(saved['positions'], context['valid_positions']):
            raise ValueError('Selected route outer artifact positions mismatch')
        result = {'prediction': saved['prediction'].copy(), 'audit': audit}
        if result['prediction'].shape != (len(Xvalid),):
            raise ValueError('Selected route outer artifact prediction shape mismatch')
        if task == 'regression':
            if not np.isfinite(result['prediction']).all():
                raise ValueError('Selected route outer artifact nonfinite predictions')
        else:
            probabilities, classes = saved['probabilities'].copy(), saved['classes'].copy()
            if not np.array_equal(classes, context['baseline_classes']):
                raise ValueError('Selected route outer artifact classes mismatch')
            if (probabilities.shape != (len(Xvalid), len(classes)) or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5)):
                raise ValueError('Selected route outer artifact invalid probabilities')
            if not np.array_equal(result['prediction'], classes[np.argmax(probabilities, axis=1)]):
                raise ValueError('Selected route outer artifact class predictions mismatch')
            result.update(probabilities=probabilities, classes=classes)
    audit['outer_fit'] = 'reused_saved_new_route'
    audit['outer_artifact'] = str(artifact)
    return result


def constrain_synthetic(X, generated, numeric):
    """Freeze integer/support rules from fit rows, then reject invalid samples."""
    generated = generated.copy()
    valid = np.ones(len(generated), dtype=bool)
    audit = {'rounded_integer_fields': [], 'tcp_zero_support_constraints': []}
    for column in numeric:
        original = pd.to_numeric(X[column], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(original) and np.allclose(original, np.rint(original), atol=1e-8, rtol=0):
            generated[column] = np.rint(generated[column].astype(float))
            audit['rounded_integer_fields'].append(column)
        if len(original) and (original >= 0).all():
            valid &= generated[column].to_numpy(dtype=float) >= -1e-8
    joint = [column for column in ('proto', 'service', 'state') if column in X]
    if len(joint) >= 2:
        original_keys = X[joint].astype('string').fillna('__MISSING__').astype(str)
        generated_keys = generated[joint].astype('string').fillna('__MISSING__').astype(str)
        observed = set(map(tuple, original_keys.to_numpy()))
        valid &= np.array([tuple(row) in observed for row in generated_keys.to_numpy()])
    if 'proto' in X:
        original_protocol = X['proto'].astype('string').str.lower()
        generated_protocol = generated['proto'].astype('string').str.lower()
        tcp_fields = [column for column in ('stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat', 'swin', 'dwin') if column in X]
        for protocol in original_protocol.dropna().unique():
            if protocol in {'tcp', '6'}:
                continue
            for column in tcp_fields:
                observed = pd.to_numeric(X.loc[original_protocol.eq(protocol).fillna(False), column], errors='coerce').dropna()
                if len(observed) and np.allclose(observed, 0., atol=1e-12):
                    mask = generated_protocol.eq(protocol).fillna(False).to_numpy(bool)
                    values = pd.to_numeric(generated[column], errors='coerce').to_numpy()
                    valid &= ~mask | (np.isfinite(values) & (np.abs(values) <= 1e-8))
                    audit['tcp_zero_support_constraints'].append([str(protocol), column])
    return generated, valid, audit


def _permutation(task, X, y, Xvalid, context):
    path = Path(context['cache_dir']) / ('permutation-' + inner_fingerprint(task, X, y, context['fit_positions'], {'policy': 'legacy', 'repeats': 2}) + '.npz')
    if path.exists():
        with np.load(path) as data:
            importance = data['importance']
    else:
        oof = cached_oof(task, X, y, context)
        models = Path(context['cache_dir']) / oof['audit']['key']
        rows = []
        for fold, (fit, valid) in enumerate(inner_folds(task, X, y)):
            model_file = models / f'fold-{fold}-model.joblib'
            if not model_file.exists():
                raise ValueError('Cached OOF model absent; cannot refit an existing inner baseline for permutation')
            model = joblib.load(model_file)
            base = _score(task, np.asarray(y)[valid], model.predict(X.iloc[valid]))
            row = []
            for index, column in enumerate(X):
                scores = []
                for repeat in range(2):
                    shuffled = X.iloc[valid].copy()
                    rng = np.random.RandomState(4200 + fold * 1000 + index * 2 + repeat)
                    shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                    scores.append(base - _score(task, np.asarray(y)[valid], model.predict(shuffled)))
                row.append(float(np.mean(scores)))
            rows.append(row)
        importance = np.asarray(rows)
        _write_npz(path, importance=importance, columns=np.asarray(X.columns, dtype=str))
    keep = stable_feature_selection(list(X.columns), importance)
    audit = {'importance_artifact': str(path), 'dropped': [c for c in X if c not in keep], 'kept': keep,
             'rule': 'remove <=20%, mean two permutations non-positive in every inner fold', 'importance': importance.tolist()}
    if len(keep) == len(X.columns):
        audit['diagnosis'] = 'insufficient_stable_nonpositive_features'
        return _reuse(context, 'baseline', audit)
    model = _make(task, {'policy': 'legacy'})
    model.fit(X[keep], y)
    return _result(model, Xvalid[keep], task, audit)


def _smote(X, y, Xvalid, context):
    from imblearn.over_sampling import SMOTENC
    from sklearn.preprocessing import StandardScaler
    categorical = smote_categorical_columns(X)
    numeric = [column for column in X if column not in categorical]
    fit = X.copy()
    for column in categorical:
        fit[column] = fit[column].astype('string').fillna('__MISSING__').astype(object)
    medians = fit[numeric].replace([np.inf, -np.inf], np.nan).median().fillna(0)
    scaler = StandardScaler()
    fit[numeric] = scaler.fit_transform(fit[numeric].replace([np.inf, -np.inf], np.nan).fillna(medians))
    strategy = smote_strategy(y)
    if not categorical or not numeric or not strategy:
        return _reuse(context, 'baseline', {'diagnosis': 'no_eligible_mixed_type_sampling', 'strategy': strategy})
    sampler = SMOTENC(categorical_features=[X.columns.get_loc(column) for column in categorical],
                       sampling_strategy=strategy, k_neighbors=3, random_state=42)
    sampled, target = sampler.fit_resample(fit, np.asarray(y))
    sampled = pd.DataFrame(sampled, columns=X.columns)
    sampled[numeric] = scaler.inverse_transform(sampled[numeric])
    # Preserve observed protocol/service/state combinations. Reject invalid
    # synthetic combinations rather than changing them using validation data.
    # Restore exactly the original rows; generated floats do not overwrite data.
    generated = sampled.iloc[len(X):].copy()
    for column in categorical:
        if pd.api.types.is_numeric_dtype(X[column]):
            generated[column] = pd.to_numeric(generated[column].replace('__MISSING__', np.nan))
        else:
            generated[column] = generated[column].replace('__MISSING__', np.nan)
    generated, valid, domain_audit = constrain_synthetic(X, generated, numeric)
    generated = generated.loc[valid].copy()
    combined = pd.concat([X, generated], ignore_index=True)
    combined_y = np.concatenate([np.asarray(y), np.asarray(target)[len(X):][valid]])
    output = Path(context['output'])
    output.mkdir(parents=True, exist_ok=True)
    synthetic_artifact = output / 'accepted-synthetic.csv.gz'
    synthetic = generated.copy()
    synthetic['__synthetic_target__'] = np.asarray(target)[len(X):][valid]
    synthetic.to_csv(synthetic_artifact, index=False, compression='gzip')
    _write_npz(output / 'synthetic-selection.npz', accepted=valid, generated_target=np.asarray(target)[len(X):].astype(str),
               original_fit_positions=np.asarray(context['fit_positions']))
    model = _make('multiclass', {'policy': 'light'})
    model.fit(combined, combined_y)
    audit = {'categorical_columns': categorical, 'numeric_columns': numeric,
             'strategy': {str(k): v for k, v in strategy.items()}, 'k_neighbors': 3,
             'generated_before_constraints': len(sampled) - len(X), 'generated_accepted': len(generated),
             'constraint_rejections': int((~valid).sum()), 'sampling_after_outer_split': True, 'domains': domain_audit,
             'synthetic_artifact': str(synthetic_artifact), 'selection_artifact': str(output / 'synthetic-selection.npz'),
             'fractional_counts_preserved': [column for column in ('spkts', 'dpkts', 'ct_srv_src', 'ct_srv_dst')
                                             if column in generated and np.any(generated[column] % 1 != 0)]}
    return _result(model, Xvalid, 'multiclass', audit)


def evaluate_advanced(number, task, X, y, Xvalid, context):
    """Return one outer prediction from a configuration chosen without its y."""
    number = int(str(number).replace('HYP-', ''))
    y = np.asarray(y)
    Path(context['cache_dir']).mkdir(parents=True, exist_ok=True)
    if number == 28:
        return _permutation(task, X, y, Xvalid, context)
    if number == 33:
        oof = cached_oof(task, X, y, context)
        mask = oof['fold_id'] >= 0
        factor, diagnostic = smearing_factor(y[mask], np.log1p(oof['prediction'][mask]), oof['fold_id'][mask])
        return {'prediction': apply_smearing(context['baseline_prediction'], factor),
                'audit': {'oof': oof['audit'], 'smearing': diagnostic, 'applied_factor': factor, 'outer_fit': 'reused_saved_baseline'}}
    if number == 34:
        scores = []
        audits = []
        configs = []
        for ratio in (.25, 1.):
            config = {'policy': 'light', 'family': 'xgboost', 'target_transform': 'none',
                      'parameters': {'objective': 'reg:pseudohubererror'}, 'huber_slope_ratio': ratio}
            oof = cached_oof(task, X, y, context, config)
            fold_scores = [_score(task, y[oof['fold_id'] == k], oof['prediction'][oof['fold_id'] == k])
                           for k in sorted(set(oof['fold_id']) - {-1})]
            scores.append(float(np.mean(fold_scores)))
            configs.append(config)
            audits.append(oof['audit'])
        selected = int(np.argmax(scores))
        slope = max(1., float(np.subtract(*np.percentile(y, [75, 25]))) * (.25, 1.)[selected])
        model = _fit(task, configs[selected], X, y)
        return _result(model, Xvalid, task, {'slope_ratios': [.25, 1.], 'inner_scores': scores, 'oof': audits, 'huber_slope': slope, 'target_scale': 'original'})
    if number == 35:
        prediction = np.asarray(context['baseline_prediction'], dtype=float).copy()
        groups = sorted(X['building_class'].dropna().unique(), key=str)
        if len(groups) != 2:
            raise ValueError('Building experts require exactly two confirmed classes')
        audit = {'global_outer': 'reused_saved_baseline', 'experts': []}
        for group in groups:
            fit = np.flatnonzero(X['building_class'].eq(group).to_numpy())
            valid = np.flatnonzero(Xvalid['building_class'].eq(group).to_numpy())
            weight = len(fit) / (len(fit) + 500.)
            audit['experts'].append({'group': str(group), 'fit_rows': len(fit), 'weight': weight, 'fitted': len(fit) >= 500})
            if len(fit) < 500:
                continue
            model = _make(task, {'policy': 'legacy'})
            model.fit(X.iloc[fit], y[fit])
            if len(valid):
                prediction[valid] = weight * model.predict(Xvalid.iloc[valid]) + (1 - weight) * prediction[valid]
        return {'prediction': prediction, 'audit': audit}
    if number == 36:
        years = pd.to_numeric(X['Year_Factor'], errors='raise').to_numpy()
        weights = 2. ** (years - years.max())
        weights /= weights.mean()
        model = _make(task, {'policy': 'legacy'})
        model.fit(X, y, sample_weight=weights)
        weight_artifact = Path(context['output']) / 'training-weights.npz'
        _write_npz(weight_artifact, weights=weights, positions=np.asarray(context['fit_positions']))
        return _result(model, Xvalid, task, {'weight_min': float(weights.min()), 'weight_max': float(weights.max()),
                                           'weight_mean': float(weights.mean()), 'weights_artifact': str(weight_artifact),
                                           'formula': '2**(year-max_training_year), normalized mean=1'})
    if number in (37, 39, 40):
        oof = cached_oof(task, X, y, context)
        classes = oof['classes']
        if number == 37:
            factors, audit = choose_multipliers(y, oof['probabilities'], classes)
            probabilities = _aligned(context['baseline_probabilities'], context['baseline_classes'], classes) * factors
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return {'prediction': classes[np.argmax(probabilities, axis=1)], 'probabilities': probabilities, 'classes': classes,
                    'audit': {**audit, 'oof': oof['audit'], 'outer_fit': 'reused_saved_baseline'}}
        if number == 39:
            positions = np.array([list(classes).index(label) for label in y])
            ptrue = oof['probabilities'][np.arange(len(y)), positions]
            weights = np.where(ptrue >= .05, 1. + 2. * (1. - ptrue), 1.)
            weights /= weights.mean()
            model = _make(task, {'policy': 'legacy'})
            model.fit(X, y, sample_weight=weights)
            weight_artifact = Path(context['output']) / 'training-weights.npz'
            _write_npz(weight_artifact, weights=weights, positions=np.asarray(context['fit_positions']), p_true=ptrue)
            return _result(model, Xvalid, task, {'oof': oof['audit'], 'noise_guard_rows': int(np.sum(ptrue < .05)),
                                               'weight_min': float(weights.min()), 'weight_max': float(weights.max()), 'weight_mean': float(weights.mean()),
                                               'weights_artifact': str(weight_artifact)})
        pair = confused_pair(y, oof['prediction'], classes)
        fit = np.flatnonzero(np.isin(y, pair))
        expert = _make(task, {'policy': 'legacy'})
        expert.fit(X.iloc[fit], y[fit])
        expert_probabilities = _aligned(expert.predict_proba(Xvalid), expert.classes_, pair)
        probabilities = _aligned(context['baseline_probabilities'], context['baseline_classes'], classes)
        probabilities, gate = redistribute_pair(probabilities, [list(classes).index(label) for label in pair], expert_probabilities)
        return {'prediction': classes[np.argmax(probabilities, axis=1)], 'probabilities': probabilities, 'classes': classes,
                'audit': {'pair': list(pair), 'pair_selection': 'maximum symmetric training OOF attack confusions, lexical tie',
                          'oof': oof['audit'], 'gated_rows': int(gate.sum()), 'gate_margin': .15, 'mass_preserved': True,
                          'global_outer': 'reused_saved_baseline', 'expert': getattr(expert, 'audit_', {})}}
    if number == 38:
        return _smote(X, y, Xvalid, context)
    if number == 41:
        routes = [{'policy': 'legacy'}, {'policy': 'full'},
                  {'policy': 'light', 'family': 'lightgbm', 'feature_number': 24 if task == 'regression' else 26}]
        audits, scores = [], []
        for config in routes:
            oof = cached_oof(task, X, y, context, config)
            mask = oof['fold_id'] >= 0
            # Average fold primary scores so a large year does not dominate selection.
            fold_scores = [_score(task, y[oof['fold_id'] == k], oof['prediction'][oof['fold_id'] == k]) for k in np.unique(oof['fold_id'][mask])]
            scores.append(float(np.mean(fold_scores)))
            audits.append(oof['audit'])
        selected = int(np.argmax(scores))
        audit = {'routes': routes, 'inner_scores': scores, 'selected': routes[selected], 'oof': audits,
                 'selection': 'maximum mean inner primary score, ties prefer legacy then full then new LightGBM route', 'maximum_routes': 3}
        if selected < 2:
            return _reuse(context, 'baseline' if selected == 0 else 'full', audit)
        key = f'{routes[selected]["feature_number"]}:{task}'
        same_family = routes[selected].get('family', 'extra_trees' if task == 'regression' else 'xgboost') == ('extra_trees' if task == 'regression' else 'xgboost')
        artifact = context.get('candidate_artifacts', {}).get(key) if same_family else None
        if artifact:
            return load_outer_artifact(artifact, task, Xvalid, context, audit)
        model = _make(task, routes[selected])
        model.fit(X, y)
        return _result(model, Xvalid, task, audit)
    raise ValueError(f'Unsupported advanced hypothesis {number}')
