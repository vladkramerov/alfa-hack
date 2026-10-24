"""SPEC-009 evaluation adapters; never silently refit the measured H43 outer."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from automl_combo_memory import _data_hash
from automl_next_reg import build_next_reg


CANDIDATES = {54: {'strategy': 'spline_gam'}, 55: {'strategy': 'quantile_mean'},
              56: {'strategy': 'h43_residual'}, 58: {'strategy': 'covariate_shift'},
              60: {'strategy': 'masked_regression'}}


def read_h43_model(path, X, Xvalid, prediction):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required saved H43 outer model missing: ' + str(path))
    from automl_reg_combo import FeatureGroupExperts
    model = joblib.load(path)
    if not isinstance(model, FeatureGroupExperts):
        raise ValueError('Saved H43 model type mismatch')
    candidate = model.candidate
    if (candidate.get('feature_numbers') != [29, 30, 31] or candidate.get('group_column', 'building_class') != 'building_class'
            or candidate.get('minimum_expert_rows', 500) != 500 or candidate.get('shrinkage', 500.) != 500.
            or candidate.get('parameters', {}) != {}):
        raise ValueError('Saved H43 candidate metadata mismatch')
    if model.audit_['fit_retained_rows'] != len(X):
        raise ValueError('Saved H43 original training row count mismatch')
    prediction = np.asarray(prediction)
    if prediction.shape != (len(Xvalid),) or not np.isfinite(prediction).all():
        raise ValueError('Saved H43 outer prediction invalid')
    if not np.allclose(model.predict(Xvalid), prediction, atol=1e-10, rtol=1e-12):
        raise ValueError('Saved H43 model/prediction mismatch')
    return model


def _cache_fit(candidate, X, y, context, base_model=None):
    positions = np.asarray(context.get('fit_positions', np.arange(len(X))), dtype=np.int64)
    identity = {'candidate': candidate, 'X': _data_hash(X), 'y': _data_hash(np.asarray(y)),
                'positions_sha256': hashlib.sha256(positions.tobytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(context['cache_dir']) / 'next-regression-models' / (key + '.joblib')
    if path.exists():
        return joblib.load(path), {'status': 'reused', 'key': key, 'artifact': str(path)}
    model = build_next_reg(candidate, threads=context.get('threads', 4))
    if candidate['strategy'] == 'h43_residual':
        model.fit(X, y, base_model=base_model, cache_dir=context['cache_dir'])
    else:
        model.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.joblib')
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)
    return model, {'status': 'fitted', 'key': key, 'artifact': str(path)}


def evaluate_reg(number, X, y, Xvalid, context):
    number = int(str(number).replace('HYP-', ''))
    if number not in CANDIDATES:
        raise ValueError('Unsupported SPEC-009 regression candidate: ' + str(number))
    y = np.asarray(y, dtype=float)
    positions = np.asarray(context.get('fit_positions', np.arange(len(X))))
    valid_positions = np.asarray(context.get('valid_positions', np.arange(len(Xvalid))))
    base = None
    if number == 56:
        base = read_h43_model(context['h43_model_path'], X, Xvalid, context['baseline_prediction'])
    candidate = dict(CANDIDATES[number])
    model, event = _cache_fit(candidate, X, y, context, base_model=base)
    audit = dict(model.audit_)
    audit['model_cache'] = event
    output = Path(context['output'])
    output.mkdir(parents=True, exist_ok=True)
    if number == 55:
        values = model.encoder_.transform(Xvalid)
        quantiles = np.column_stack([branch.predict(values) for branch in model.models_])
        path = output / 'quantile-validation-components.npz'
        np.savez_compressed(path, positions=valid_positions, quantiles=np.asarray(model.QUANTILES), prediction=quantiles)
        audit['quantile_validation_artifact'] = str(path)
        prediction = quantiles.mean(axis=1)
    elif number == 56:
        corrections = model.residual_.predict(Xvalid)
        prediction = np.asarray(context['baseline_prediction']) + corrections
        path = output / 'residual-training-oof.npz'
        np.savez_compressed(path, positions=positions, prediction=model.oof_prediction_, target=y,
                            residual_positions=positions[model.residual_positions_], residual_target=model.residual_target_,
                            outer_positions=valid_positions, outer_base_prediction=np.asarray(context['baseline_prediction']),
                            outer_correction=corrections)
        audit.update(oof_artifact=str(path), outer_base_model_source=str(context['h43_model_path']), outer_baseline_refits=0)
        for event in audit['base_oof_folds']:
            event['original_fit_positions'] = positions[np.asarray(event['fit_positions'], dtype=int)].tolist()
            event['original_valid_positions'] = positions[np.asarray(event['valid_positions'], dtype=int)].tolist()
    else:
        prediction = model.predict(Xvalid)
    if number == 58:
        path = output / 'domain-cross-fit-weights.npz'
        np.savez_compressed(path, positions=positions, domain_probability=model.domain_probability_, weights=model.sample_weight_,
                            domain=(X['Year_Factor'].to_numpy() == X['Year_Factor'].max()).astype(int),
                            domain_columns=np.asarray(model.domain_columns_, dtype=str))
        audit['domain_weights_artifact'] = str(path)
        for event in audit['domain_folds']:
            event['original_fit_positions'] = positions[np.asarray(event['fit_positions'], dtype=int)].tolist()
            event['original_valid_positions'] = positions[np.asarray(event['valid_positions'], dtype=int)].tolist()
    if number == 60:
        path = output / 'training-mask.npz'
        np.savez_compressed(path, positions=positions, mask=model.mask_, columns=np.asarray(audit['masked_columns'], dtype=str))
        audit['mask_artifact'] = str(path)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(Xvalid),) or not np.isfinite(prediction).all():
        raise ValueError('Invalid SPEC-009 regression prediction')
    return {'prediction': prediction, 'audit': audit, 'learned_state': {}, 'candidate': candidate}
