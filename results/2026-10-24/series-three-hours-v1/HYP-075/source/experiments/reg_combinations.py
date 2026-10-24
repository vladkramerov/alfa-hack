"""SPEC-008 evaluation adapters; old outer models/predictions are never refitted."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from automl_reg_combo import (FeatureGroupExperts, RegressionComponent, build_reg_combo,
                              choose_convex_weights, temporal_folds)
from experiments.twenty_advanced import inner_fingerprint


def read_legacy_oof(X, y, context):
    """Read existing SPEC-006 folds and models without invoking cached_oof()."""
    positions = np.asarray(context.get('fit_positions', np.arange(len(X))))
    key = inner_fingerprint('regression', X, y, positions, {'policy': 'legacy'})
    directory = Path(context['legacy_cache_dir']) / key
    prediction = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1)
    models, events = [], []
    for number, (fit, valid) in enumerate(temporal_folds(X)):
        file = directory / f'fold-{number}.npz'
        model_file = directory / f'fold-{number}-model.joblib'
        if not file.exists() or not model_file.exists():
            raise FileNotFoundError('Required read-only legacy OOF/model cache missing: ' + str(file))
        with np.load(file, allow_pickle=False) as saved:
            if (not np.array_equal(saved['fit'], fit) or not np.array_equal(saved['valid'], valid)
                    or not np.array_equal(saved['fit_positions'], positions[fit])
                    or not np.array_equal(saved['valid_positions'], positions[valid])):
                raise ValueError('Legacy OOF positions mismatch')
            if saved['prediction'].shape != (len(valid),) or not np.isfinite(saved['prediction']).all():
                raise ValueError('Invalid legacy OOF prediction')
            prediction[valid] = saved['prediction']
        models.append(joblib.load(model_file))
        fold_id[valid] = number
        events.append({'fold': number, 'source': str(file), 'model_source': str(model_file), 'status': 'reused_read_only'})
    return {'prediction': prediction, 'fold_id': fold_id, 'models': models, 'audit': {'key': key, 'folds': events}}


def _cache_key(X, y, positions, configuration):
    digest = hashlib.sha256(json.dumps({'columns': list(X.columns), 'dtypes': list(map(str, X.dtypes)),
                                      'configuration': configuration}, sort_keys=True).encode())
    for values in (pd.util.hash_pandas_object(X, index=False).to_numpy(),
                   pd.util.hash_pandas_object(pd.Series(np.asarray(y)), index=False).to_numpy(),
                   np.asarray(positions, dtype=np.int64)):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _cached_component(model, X, y, context, positions, configuration, global_model=None):
    key = _cache_key(X, y, positions, configuration)
    directory = Path(context['cache_dir']) / 'regression-models'
    path = directory / (key + '.joblib')
    if path.exists():
        return joblib.load(path), {'key': key, 'status': 'reused', 'artifact': str(path)}
    if global_model is None:
        model.fit(X, y)
    else:
        model.fit(X, y, global_model=global_model)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.joblib')
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)
    return model, {'key': key, 'status': 'fitted', 'artifact': str(path)}


def _convex_evaluation(X, y, Xvalid, context):
    legacy = read_legacy_oof(X, y, context)
    predictions = np.full((len(X), 4), np.nan)
    predictions[:, 0] = legacy['prediction']
    positions = np.asarray(context.get('fit_positions', np.arange(len(X))))
    component_audits = []
    for fold, (fit, valid) in enumerate(temporal_folds(X)):
        expert_candidate = {'strategy': 'reg_feature_experts', 'feature_numbers': [], 'group_column': 'building_class',
                            'minimum_expert_rows': 500, 'shrinkage': 500.}
        configurations = [expert_candidate, {'component': 'extra-trees', 'features': [30], 'scale': True, 'target_transform': 'log1p'},
                           {'component': 'extra-trees', 'features': [31], 'scale': True, 'target_transform': 'log1p'}]
        models = [FeatureGroupExperts(expert_candidate, threads=4), RegressionComponent([30], True, threads=4),
                  RegressionComponent([31], True, threads=4)]
        for column, (model, configuration) in enumerate(zip(models, configurations), start=1):
            model, audit = _cached_component(model, X.iloc[fit], y[fit], context, positions[fit], configuration,
                                              global_model=legacy['models'][fold] if column == 1 else None)
            predictions[valid, column] = model.predict(X.iloc[valid])
            component_audits.append({'fold': fold, 'component': column, **audit, 'fit_rows': len(fit), 'valid_rows': len(valid),
                                     'fit_positions': positions[fit].tolist(), 'valid_positions': positions[valid].tolist()})
    weights, selection = choose_convex_weights(y, predictions, legacy['fold_id'])
    order = ['raw', 'experts', 'season', 'relative']
    outer = np.column_stack([np.asarray(context['outer_components'][key], dtype=float) for key in order])
    if outer.shape != (len(Xvalid), 4) or not np.isfinite(outer).all():
        raise ValueError('Invalid reused outer component predictions')
    output = Path(context['output'])
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / 'convex-training-oof.npz'
    np.savez_compressed(artifact, prediction=predictions, target=y, fold_id=legacy['fold_id'],
                        fit_positions=positions, weights=weights, component_order=np.asarray(order))
    learned = {'blend_weights': weights.tolist()}
    audit = {'strategy': 'reg_convex_blend', 'fit_original_rows': len(X), 'fit_retained_rows': len(X),
             'legacy_oof': legacy['audit'], 'components': component_audits, 'selection': selection,
             'outer_fit': 'reused_four_saved_predictions', 'oof_artifact': str(artifact), 'component_order': order,
             'validation_target_used': False, 'blend_weights': weights.tolist()}
    return {'prediction': outer @ weights, 'audit': audit, 'learned_state': learned}


def evaluate_reg(number, X, y, Xvalid, context):
    number = int(str(number).replace('HYP-', ''))
    y = np.asarray(y)
    if number == 44:
        return _convex_evaluation(X, y, Xvalid, context)
    candidates = {
        42: {'strategy': 'reg_feature_experts', 'feature_numbers': [30], 'group_column': 'building_class',
             'minimum_expert_rows': 500, 'shrinkage': 500.},
        43: {'strategy': 'reg_feature_experts', 'feature_numbers': [29, 30, 31], 'group_column': 'building_class',
             'minimum_expert_rows': 500, 'shrinkage': 500.},
        45: {'strategy': 'target_tail_mixture', 'tail_quantile': .8}}
    if number not in candidates:
        raise ValueError('Unsupported regression combination: ' + str(number))
    candidate = candidates[number]
    model, event = _cached_component(build_reg_combo(candidate, threads=4), X, y, context,
                                    context.get('fit_positions', np.arange(len(X))), candidate)
    return {'prediction': np.asarray(model.predict(Xvalid)), 'audit': {**model.audit_, 'model_cache': event}, 'learned_state': {}}
