"""Read-only H88 teacher reuse for H92--94; never constructs or fits a learner."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.sprint_cache import (_arrays, _indices, _json, _original_context,
    _read_component, assemble_oof, validate_probabilities)
from experiments.sprint_crossfit_cache import _nested_folds, _position_hash, load_honest_teacher
from research_memory import data_digest


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen_sources(root):
    frozen = Path(root) / 'results/2026-10-24/series-three-hours-v1/HYP-088'
    sources = _json(frozen / 'frozen.json')['sources']
    directory = (frozen / 'source').resolve()
    for name, expected in sources.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or _hash(path) != expected:
            raise ValueError('H88 historical frozen source SHA mismatch: ' + name)
    required = ['automl_class_combo.py', 'automl_models.py', 'automl_combo_memory.py',
                'sprint_contextual.py', 'sprint_honest.py', 'experiments/sprint_crossfit_cache.py']
    if any(name not in sources for name in required):
        raise ValueError('H88 historical frozen teacher source manifest incomplete')
    return {Path(name).name: sources[name] for name in required}, frozen / 'frozen.json'


def load_reused_honest_teacher(bundle, stage, fit, valid, *, root, output):
    """Return compatible RAW grouped-OOF train/query arrays using saved fits only.

    output is deliberately unused: this provider never creates or mutates cache
    files. Missing/incomplete evidence fails closed instead of invoking a fit.
    """
    root = Path(root).resolve()
    if stage not in ('inner', 'outer'):
        raise ValueError('Honest reuse stage must be inner or outer')
    if stage == 'outer':
        result = load_honest_teacher(bundle, stage, fit, valid, root=root, output=output, fit_hook=None)
        result['audit'].update(protocol='H92_94_reused_honest_raw_teacher', reused_H88_teachers=True)
        if result['audit']['model_fits'] != 0:
            raise ValueError('Read-only reuse must never create teacher fits')
        return result
    fit = _indices(fit, 'requested fit', len(bundle.y))
    valid = _indices(valid, 'requested valid', len(bundle.y))
    if np.intersect1d(fit, valid).size:
        raise ValueError('Requested training/query positions overlap')
    _, outer_fit, outer_valid = _original_context(bundle, root)
    raw = _read_component(bundle, outer_fit, 'raw', root)
    classes = np.asarray(raw['classes']).astype(str)
    query_fold = raw['folds'][0]
    if (not np.array_equal(fit, query_fold['fit_positions'])
            or not np.array_equal(valid, query_fold['valid_positions'])
            or not np.array_equal(classes, np.unique(np.asarray(bundle.y)[fit]).astype(str))):
        raise ValueError('Reused inner teacher requires exact original RAW fold0 class/positions')
    groups = pd.util.hash_pandas_object(bundle.X, index=False).to_numpy()
    if set(groups[fit]) & set(groups[valid]):
        raise ValueError('Raw-X training/query groups overlap')
    source_hashes, manifest = _frozen_sources(root)
    run = root / 'results/2026-10-24/HYP-088-01-inner-default-sprint'
    directory = run / 'honest-raw-oof'
    X, y = bundle.X.iloc[fit], np.asarray(bundle.y)[fit]
    folds = _nested_folds(X, y)
    saved_folds, events = [], []
    for number, (local_fit, local_valid) in enumerate(folds):
        artifact = directory / f'fold-{number}.npz'
        started = directory / f'fold-{number}.started.json'
        outcome = directory / f'fold-{number}.outcome.json'
        splits_path = directory / f'fold-{number}.splits.npz'
        if not all(path.is_file() for path in [artifact, started, outcome, splits_path]):
            raise ValueError('H88 nested teacher missing or unfinished; no refit allowed')
        evidence, terminal = _json(started), _json(outcome)
        expected = {'hypothesis': 'HYP-088', 'role': 'nested_grouped_oof_raw_teacher',
            'family': 'xgboost', 'component': 'raw', 'parameters': {'n_estimators':600,
                'max_depth':6, 'learning_rate':.05, 'reg_lambda':1, 'tree_method':'hist',
                'device':'cpu', 'objective':'multi:softprob'},
            'encoder': {'native':False, 'scale':False}, 'parameters_override':{},
            'threads':2, 'stage':'inner', 'nested_fold':number,
            'fit_positions':fit[local_fit].tolist(), 'valid_positions':fit[local_valid].tolist(),
            'fit_positions_sha256':_position_hash(fit[local_fit]),
            'valid_positions_sha256':_position_hash(fit[local_valid]),
            'data_hashes':dict(bundle.hashes), 'source_hashes':source_hashes,
            'X_sha256':data_digest(X.iloc[local_fit]),
            'y_sha256':data_digest(y[local_fit].reshape(-1,1)),
            'split':'SGKF3_seed42_raw_X_groups', 'raw_groups_disjoint':True}
        if (evidence.get('status') != 'started' or evidence.get('configuration') != expected
                or evidence.get('classes') != classes.tolist() or terminal.get('status') != 'completed'
                or terminal.get('prediction_sha256') != _hash(artifact)
                or terminal.get('fit_rows') != len(local_fit) or terminal.get('valid_rows') != len(local_valid)):
            raise ValueError('H88 nested started/config/classes/outcome/artifact SHA mismatch')
        saved, partition = _arrays(artifact), _arrays(splits_path)
        for name, positions in [('fit',local_fit), ('valid',local_valid),
                                ('fit_positions',fit[local_fit]), ('valid_positions',fit[local_valid])]:
            if not np.array_equal(saved[name],positions) or not np.array_equal(partition[name],positions):
                raise ValueError('H88 nested local/global split mismatch')
        if not np.array_equal(np.asarray(saved['classes']).astype(str),classes):
            raise ValueError('H88 nested class order mismatch')
        saved_folds.append(saved)
        events.append({'fold':number, 'reused':True, 'artifact':str(artifact),
            'prediction_sha256':_hash(artifact), 'started_evidence':str(started),
            'outcome':str(outcome), 'fit_rows':len(local_fit), 'valid_rows':len(local_valid),
            'raw_groups_disjoint':True})
    assembled = assemble_oof(saved_folds,fit,classes,folds)
    train = validate_probabilities(assembled['probabilities'],classes,len(fit))
    query = validate_probabilities(query_fold['probabilities'],classes,len(valid))
    query_artifact = raw['directory'] / 'fold-0.npz'
    audit = {'teacher':'raw_xgboost_grouped_oof', 'protocol':'H92_94_reused_honest_raw_teacher',
        'stage':stage, 'training_predictions':'grouped_oof_raw_teacher',
        'independent_training_oof':True, 'independent_valid_predictions':True,
        'H48_meta_state_used':False, 'old_model_refits':0, 'new_teacher_fits':0, 'model_fits':0,
        'reused_H88_teachers':True, 'nested_folds':events, 'source_hashes':source_hashes,
        'historical_frozen_manifest':str(manifest), 'historical_frozen_manifest_sha256':_hash(manifest),
        'raw_component':raw['audit'], 'query_artifact':str(query_artifact),
        'query_artifact_sha256':_hash(query_artifact),
        'fit_positions_sha256':_position_hash(fit), 'valid_positions_sha256':_position_hash(valid),
        'teacher_train_sha256':hashlib.sha256(train.tobytes()).hexdigest(),
        'comparison_reference_is_separate_from_raw_teacher':True}
    return {'train_probabilities':train,'valid_probabilities':query,'classes':classes,
            'fit_positions':fit,'valid_positions':valid,'audit':audit}
