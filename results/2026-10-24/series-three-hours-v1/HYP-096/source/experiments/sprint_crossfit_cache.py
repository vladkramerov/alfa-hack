"""H88 honest raw teacher provider; all new fits require an injected registrar.

Original outer OOF/query artifacts are strictly read-only. Inner additionally
needs three new nested SGKF3 raw teachers; each gets immutable started evidence
and global positions before the hook receives the raw X/y. Interrupted claims
cannot be refitted. No fit is executed merely by importing this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from automl_class_combo import ClassBranch, aligned_probabilities
from experiments.sprint_cache import (_arrays, _indices, _json, _original_context,
    _read_component, _validate_source, assemble_oof, validate_probabilities)
from research_memory import atomic_json, data_digest, timestamp


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _position_hash(positions):
    return hashlib.sha256(np.asarray(positions, dtype='<i8').tobytes()).hexdigest()


def _sources():
    import automl_class_combo
    import automl_models
    import automl_combo_memory
    import sprint_contextual
    import sprint_honest
    modules = [automl_class_combo, automl_models, automl_combo_memory, sprint_contextual, sprint_honest]
    return {**{Path(module.__file__).name: _hash(module.__file__) for module in modules},
            Path(__file__).name: _hash(__file__)}


def _nested_folds(X, y):
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if len(np.unique(groups)) < 3:
        raise ValueError('At least three raw-X groups are needed for nested OOF')
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, y, groups))
    for fit, valid in folds:
        if (not len(fit) or not len(valid) or np.intersect1d(fit, valid).size
                or set(groups[fit]) & set(groups[valid])):
            raise ValueError('Nested raw groups overlap or fold is empty')
    return folds


def _build_inner(bundle, positions, classes, *, root, output, fit_hook):
    from experiments.twenty_advanced import inner_fingerprint
    X, y = bundle.X.iloc[positions], np.asarray(bundle.y)[positions]
    historical_key = inner_fingerprint('multiclass', X, y, positions, {'policy': 'legacy'})
    historical = root / 'results/2026-10-23/series-twenty-hypotheses-v1/cache' / historical_key
    if historical.exists():
        cached = _read_component(bundle, positions, 'raw', root)
        return cached['probabilities'], {'new_teacher_fits': 0, 'reused_historical_nested_cache': cached['audit'],
                                         'nested_folds': [], 'historical_nested_key': historical_key}
    folds = _nested_folds(X, y)
    directory = output / 'honest-raw-oof'
    saved_folds, events, new_fits = [], [], 0
    source_hashes = _sources()
    for number, (local_fit, local_valid) in enumerate(folds):
        global_fit, global_valid = positions[local_fit], positions[local_valid]
        config = {'hypothesis': 'HYP-088', 'role': 'nested_grouped_oof_raw_teacher',
            'family': 'xgboost', 'component': 'raw', 'parameters': {'n_estimators': 600,
                'max_depth': 6, 'learning_rate': .05, 'reg_lambda': 1, 'tree_method': 'hist',
                'device': 'cpu', 'objective': 'multi:softprob'},
            'encoder': {'native': False, 'scale': False}, 'parameters_override': {},
            'threads': 2, 'stage': 'inner', 'nested_fold': number,
            'fit_positions': global_fit.tolist(), 'valid_positions': global_valid.tolist(),
            'fit_positions_sha256': _position_hash(global_fit), 'valid_positions_sha256': _position_hash(global_valid),
            'data_hashes': dict(bundle.hashes), 'source_hashes': source_hashes,
            'X_sha256': data_digest(X.iloc[local_fit]), 'y_sha256': data_digest(y[local_fit].reshape(-1, 1)),
            'split': 'SGKF3_seed42_raw_X_groups', 'raw_groups_disjoint': True}
        artifact = directory / f'fold-{number}.npz'
        started = directory / f'fold-{number}.started.json'
        outcome = directory / f'fold-{number}.outcome.json'
        if artifact.exists() or started.exists() or outcome.exists():
            if not (artifact.exists() and started.exists() and outcome.exists()):
                raise ValueError('Nested teacher fit already attempted; unfinished/incomplete cache cannot be refitted')
            evidence, terminal = _json(started), _json(outcome)
            if evidence['configuration'] != config or terminal['status'] != 'completed' or terminal['prediction_sha256'] != _hash(artifact):
                raise ValueError('Nested teacher cache configuration/outcome/SHA mismatch')
            saved = _arrays(artifact)
            reused = True
        else:
            if not callable(fit_hook):
                raise ValueError('New nested raw teacher requires an injected pre-fit registration hook')
            directory.mkdir(parents=True, exist_ok=True)
            # Append-only started evidence and split exist before the registrar/fit.
            with started.open('x', encoding='utf8') as stream:
                json.dump({'status': 'started', 'started': timestamp(), 'configuration': config,
                           'classes': classes.tolist()}, stream, ensure_ascii=False, indent=2)
            np.savez_compressed(directory / f'fold-{number}.splits.npz', fit=local_fit, valid=local_valid,
                                fit_positions=global_fit, valid_positions=global_valid)
            try:
                estimator = ClassBranch('raw', parameters={}, threads=2)
                estimator = fit_hook(estimator, X.iloc[local_fit], y[local_fit], config)
                actual_classes = np.asarray(estimator.classes_).astype(str)
                if set(actual_classes) != set(np.unique(y[local_fit]).astype(str)):
                    raise ValueError('Fitted raw teacher classes differ from its own training labels')
                p = aligned_probabilities(estimator.predict_proba(X.iloc[local_valid]), actual_classes, classes)
                p = validate_probabilities(p, classes, len(local_valid))
                saved = {'fit': local_fit, 'valid': local_valid, 'fit_positions': global_fit,
                         'valid_positions': global_valid, 'classes': classes, 'probabilities': p}
                with artifact.open('xb') as stream:
                    np.savez_compressed(stream, **saved)
                atomic_json(outcome, {'status': 'completed', 'finished': timestamp(),
                    'prediction_sha256': _hash(artifact), 'fit_rows': len(local_fit), 'valid_rows': len(local_valid)})
                new_fits += 1
                reused = False
            except Exception as exc:
                atomic_json(outcome, {'status': 'failed', 'finished': timestamp(),
                                      'error': type(exc).__name__ + ': ' + str(exc)})
                raise
        saved_folds.append(saved)
        events.append({'fold': number, 'reused': reused, 'fit_rows': len(local_fit), 'valid_rows': len(local_valid),
                       'started_evidence': str(started), 'outcome': str(outcome),
                       'artifact': str(artifact), 'prediction_sha256': _hash(artifact), 'raw_groups_disjoint': True})
    assembled = assemble_oof(saved_folds, positions, classes, folds)
    # Save only after all three folds have complete, nonoverlapping coverage.
    return assembled['probabilities'], {'new_teacher_fits': new_fits, 'nested_folds': events,
        'historical_nested_key': historical_key, 'training_oof_rows': len(positions),
        'source_hashes': source_hashes, 'expected_nested_folds': 3}


def load_honest_teacher(bundle, stage, fit, valid, *, root, output, fit_hook):
    """Return honest training RAW OOF plus original held-out RAW query teacher."""
    if stage not in ('inner', 'outer'):
        raise ValueError('H88 teacher stage must be inner or outer')
    root, output = Path(root).resolve(), Path(output).resolve()
    fit, valid = _indices(fit, 'requested fit', len(bundle.y)), _indices(valid, 'requested valid', len(bundle.y))
    if np.intersect1d(fit, valid).size:
        raise ValueError('Requested H88 fit/validation overlap')
    _, outer_fit, outer_valid = _original_context(bundle, root)
    raw = _read_component(bundle, outer_fit, 'raw', root)
    classes = np.asarray(raw['classes']).astype(str)
    if not np.array_equal(classes, np.unique(np.asarray(bundle.y)[fit]).astype(str)):
        raise ValueError('H88 teacher class order differs from requested training labels')
    if stage == 'outer':
        if not np.array_equal(fit, outer_fit) or not np.array_equal(valid, outer_valid):
            raise ValueError('H88 outer teacher requires exact original outer positions')
        source = root / 'results/2026-10-23/HYP-011-02-multiclass-none'
        _validate_source(bundle, _json(source / 'evidence.json'))
        query = _arrays(source / 'validation_predictions.npz', ['positions', 'classes', 'probabilities'])
        source_split = _arrays(source / 'splits.npz')
        if (not np.array_equal(query['positions'], valid) or not np.array_equal(query['classes'].astype(str), classes)
                or not np.array_equal(source_split['fit'], fit) or not np.array_equal(source_split['valid'], valid)):
            raise ValueError('Original H11 RAW query class/position mismatch')
        train = raw['probabilities']
        query_p = validate_probabilities(query['probabilities'], classes, len(valid))
        query_artifact = source / 'validation_predictions.npz'
        extra = {'new_teacher_fits': 0, 'nested_folds': [], 'reused_original_outer_raw_oof': True}
    else:
        fold = raw['folds'][0]
        if not np.array_equal(fit, fold['fit_positions']) or not np.array_equal(valid, fold['valid_positions']):
            raise ValueError('H88 inner teacher requires exact original raw fold0 positions')
        groups = pd.util.hash_pandas_object(bundle.X, index=False).to_numpy()
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError('H88 inner query raw groups overlap training')
        query_p = validate_probabilities(fold['probabilities'], classes, len(valid))
        query_artifact = raw['directory'] / 'fold-0.npz'
        train, extra = _build_inner(bundle, fit, classes, root=root, output=output, fit_hook=fit_hook)
    train = validate_probabilities(train, classes, len(fit))
    audit = {'teacher': 'raw_xgboost_grouped_oof', 'protocol': 'H88_honest_raw_context', 'stage': stage,
        'training_predictions': 'grouped_oof_raw_teacher', 'independent_training_oof': True,
        'independent_valid_predictions': True, 'H48_meta_state_used': False, 'old_model_refits': 0,
        'comparison_reference_is_separate_from_raw_teacher': True,
        'raw_component': raw['audit'], 'query_artifact': str(query_artifact),
        'query_artifact_sha256': _hash(query_artifact), 'fit_positions_sha256': _position_hash(fit),
        'valid_positions_sha256': _position_hash(valid), **extra}
    audit['model_fits'] = audit['new_teacher_fits']
    audit['teacher_train_sha256'] = hashlib.sha256(train.tobytes()).hexdigest()
    return {'train_probabilities': train, 'valid_probabilities': query_p, 'classes': classes,
            'fit_positions': fit, 'valid_positions': valid, 'audit': audit}
