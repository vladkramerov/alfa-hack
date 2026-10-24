"""Isolated experiment executor; imports only modules shipped in the flat ZIP."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from automl_data import load_task, score, make_submission, atomic_submission
from automl_models import build_pipeline, candidate_key, canonical_candidate
from automl_runtime import save_json


_PROBABILITY_METRICS = {'roc_auc', 'log_loss'}


def _class_key(value):
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__ + ':' + repr(value)


def aligned_probabilities(model, X, classes):
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    labels = list(model.classes_)
    if probabilities.shape != (len(X), len(labels)):
        raise ValueError('Estimator probability shape disagrees with classes_')
    result = np.zeros((len(X), len(classes)), dtype=float)
    for i, label in enumerate(labels):
        if label not in classes:
            raise ValueError('Estimator returned an unknown class')
        result[:, list(classes).index(label)] = probabilities[:, i]
    if not np.isfinite(result).all() or np.any(result < 0) or not np.allclose(result.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid class probabilities')
    return result


def train_priors(y_fit, classes):
    counts_by_key = {}
    for value in y_fit.tolist():
        key = _class_key(value)
        counts_by_key[key] = counts_by_key.get(key, 0) + 1
    counts = np.asarray([counts_by_key.get(_class_key(value), 0) for value in classes], dtype=float)
    total = counts.sum()
    if not len(classes) or total <= 0:
        raise ValueError('Cannot compute classification priors from empty fit labels')
    return counts / total


def class_decisions(probabilities, classes, priors, decision_power):
    probabilities = _validate_probabilities(probabilities, len(probabilities), len(classes), 'decision probabilities')
    priors = np.asarray(priors, dtype=float)
    if priors.shape != (len(classes),) or not np.isfinite(priors).all() or np.any(priors < 0):
        raise ValueError('Invalid train priors')
    adjusted = probabilities * np.power(np.maximum(priors, 1e-12), -float(decision_power))[None, :]
    return np.asarray(classes)[np.argmax(adjusted, axis=1)]


def _validate_probabilities(probabilities, rows, columns, name):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (rows, columns):
        raise ValueError(f'Invalid {name} shape')
    if (not np.isfinite(probabilities).all() or np.any(probabilities < -1e-10)
            or np.any(probabilities > 1 + 1e-10)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError(f'Invalid {name} values')
    return probabilities


def _training_key(candidate):
    return candidate_key({**candidate, 'decision_power': 0})


def _cache_classes(classes):
    return np.asarray([_class_key(value) for value in classes], dtype=str)


def _save_probability_cache(path, candidate, fit, valid, test_positions, classes,
                            priors, valid_probabilities, test_probabilities, model):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    payload = {
        'version': np.asarray(1, dtype=np.int64),
        'training_key': np.asarray(_training_key(candidate)),
        'fit_positions': np.asarray(fit, dtype=np.int64),
        'valid_positions': np.asarray(valid, dtype=np.int64),
        'test_positions': np.asarray(test_positions, dtype=np.int64),
        'class_order': _cache_classes(classes),
        'train_priors': np.asarray(priors, dtype=float),
        'valid_probabilities': _validate_probabilities(valid_probabilities, len(valid), len(classes), 'valid probabilities'),
        'test_probabilities': _validate_probabilities(test_probabilities, len(test_positions), len(classes), 'test probabilities'),
        'feature_code': np.asarray(str(getattr(model, 'feature_code_', ''))),
        'audit_metadata': np.asarray(json.dumps(getattr(model, 'audit_', {}), ensure_ascii=False, default=str)),
    }
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def _load_probability_cache(path, candidate, fit, valid, test_positions, classes, expected_priors, run_root):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('reuse_probabilities must be an absolute path')
    resolved = path.resolve()
    run_root = Path(run_root).resolve()
    if resolved.name != 'probability_cache.npz' or resolved.parent.parent != run_root:
        raise ValueError('Probability cache must belong to an experiment in this run')
    with np.load(resolved, allow_pickle=False) as cache:
        required = {'version', 'training_key', 'fit_positions', 'valid_positions', 'test_positions',
                    'class_order', 'train_priors', 'valid_probabilities', 'test_probabilities',
                    'feature_code', 'audit_metadata'}
        if not required.issubset(cache.files):
            raise ValueError('Probability cache is missing required fields')
        if int(cache['version'].item()) != 1:
            raise ValueError('Unsupported probability cache version')
        if str(cache['training_key'].item()) != _training_key(candidate):
            raise ValueError('Probability cache training_key mismatch')
        for field, expected in (('fit_positions', fit), ('valid_positions', valid), ('test_positions', test_positions)):
            actual = np.asarray(cache[field])
            if actual.dtype.kind not in 'iu' or not np.array_equal(actual, np.asarray(expected, dtype=np.int64)):
                raise ValueError(f'Probability cache {field} mismatch')
        if not np.array_equal(np.asarray(cache['class_order']), _cache_classes(classes)):
            raise ValueError('Probability cache class order mismatch')
        priors = np.asarray(cache['train_priors'], dtype=float)
        if (priors.shape != (len(classes),) or not np.isfinite(priors).all()
                or np.any(priors < 0) or np.any(priors > 1)
                or not np.isclose(priors.sum(), 1, atol=1e-10, rtol=0)
                or not np.allclose(priors, expected_priors, atol=1e-12, rtol=0)):
            raise ValueError('Probability cache train priors mismatch')
        valid_probabilities = _validate_probabilities(
            cache['valid_probabilities'], len(valid), len(classes), 'cached valid probabilities')
        test_probabilities = _validate_probabilities(
            cache['test_probabilities'], len(test_positions), len(classes), 'cached test probabilities')
        feature_code = str(cache['feature_code'].item())
        audit_metadata = json.loads(str(cache['audit_metadata'].item()))
        if not isinstance(audit_metadata, dict):
            raise ValueError('Probability cache audit metadata must be an object')
    return valid_probabilities, test_probabilities, priors, feature_code, audit_metadata


def predict(model, bundle, X, fit_labels=None):
    if bundle.task_type == 'classification':
        probabilities = aligned_probabilities(model, X, bundle.classes)
        if bundle.metric in _PROBABILITY_METRICS:
            return probabilities
        priors = train_priors(fit_labels, bundle.classes) if fit_labels is not None else _model_priors(model, bundle.classes)
        return class_decisions(probabilities, bundle.classes, priors,
                               model.candidate.get('decision_power', 0.0))
    result = np.asarray(model.predict(X)).reshape(-1)
    if bundle.task_type == 'regression' and not np.isfinite(result).all():
        raise ValueError('Nonfinite regression prediction')
    return result


def _model_priors(model, classes):
    source_classes = list(model.classes_)
    source_priors = np.asarray(model.priors_, dtype=float)
    result = np.zeros(len(classes), dtype=float)
    targets = {_class_key(label): index for index, label in enumerate(classes)}
    for label, prior in zip(source_classes, source_priors):
        if _class_key(label) not in targets:
            raise ValueError('Estimator returned an unknown prior class')
        result[targets[_class_key(label)]] = prior
    return result


def diagnostics(bundle, valid, prediction, model=None, fit_labels=None):
    y = bundle.y.iloc[valid].to_numpy()
    if bundle.task_type == 'classification':
        labels = list(bundle.classes)
        predicted_labels = (np.asarray(labels)[np.argmax(prediction, axis=1)]
                            if np.ndim(prediction) == 2 else prediction)
        report = classification_report(y, predicted_labels, labels=labels,
            target_names=[str(c) for c in labels], output_dict=True, zero_division=0)
        matrix = confusion_matrix(y, predicted_labels, labels=labels)
        pairs = [(int(matrix[a, b]), str(labels[a]), str(labels[b]))
                 for a in range(len(labels)) for b in range(len(labels)) if a != b and matrix[a, b]]
        fit_keys = {_class_key(value) for value in (fit_labels.tolist() if fit_labels is not None else model.classes_)}
        return {'per_class': report, 'largest_confusions': sorted(pairs, reverse=True)[:10],
                'missing_fit_classes': [str(c) for c in labels if _class_key(c) not in fit_keys]}
    residual = np.asarray(prediction) - y
    absolute = np.abs(residual)
    bins = []
    boundaries = np.unique(np.quantile(y, [0, .25, .5, .75, .9, 1]))
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (y >= low) & (y <= high)
        if mask.any():
            bins.append({'target_low': float(low), 'target_high': float(high),
                'rows': int(mask.sum()), 'bias': float(residual[mask].mean()),
                'rmse': float(np.sqrt(np.mean(residual[mask] ** 2)))})
    correlations = {}
    for column in bundle.X.select_dtypes(include='number').columns:
        values = bundle.X.iloc[valid][column].replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() > 10 and values.nunique() > 1 and np.std(absolute) > 0:
            value = values.corr(pd.Series(absolute, index=values.index))
            if np.isfinite(value):
                correlations[str(column)] = float(value)
    return {'residual_bias': float(residual.mean()),
            'absolute_error_quantiles': dict(zip(('p50', 'p90', 'p99'), map(float, np.quantile(absolute, [.5, .9, .99])))),
            'target_bins': bins,
            'error_feature_correlations': dict(sorted(correlations.items(), key=lambda v: abs(v[1]), reverse=True)[:8])}


def run(manifest):
    request = json.loads(Path(manifest).read_text(encoding='utf-8'))
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        bundle = load_task(request['task_path'])
        raw_candidate = request['candidate']
        if isinstance(raw_candidate, dict) and 'reuse_probabilities' in raw_candidate:
            raise ValueError('reuse_probabilities is an engine request field, not a candidate field')
        candidate = canonical_candidate(request['candidate'])
        model = None
        fit_labels = None
        test_prediction = None
        generated = ''
        model_audit = {}
        training_reused = False
        fit_seconds = 0.0
        cache_payload = None
        if request['mode'] in ('evaluate', 'audit'):
            with np.load(request['split_path']) as split:
                fit = np.asarray(split['fit'], dtype=np.int64)
                valid = np.asarray(split['valid'], dtype=np.int64)
            fit_labels = bundle.y.iloc[fit]
            test_positions = np.arange(len(bundle.X_test), dtype=np.int64)
            cache_eligible = (request['mode'] == 'evaluate' and bundle.task_type == 'classification'
                              and bundle.metric not in _PROBABILITY_METRICS and candidate['family'] != 'blend')
            reuse_path = request.get('reuse_probabilities') if cache_eligible else None
            if reuse_path:
                (valid_probabilities, test_probabilities, priors,
                 generated, cached_audit) = _load_probability_cache(
                    reuse_path, candidate, fit, valid, test_positions, bundle.classes,
                    train_priors(fit_labels, bundle.classes), output.resolve().parent)
                fit_seconds = 0.0
                training_reused = True
                model_audit = cached_audit
                prediction = class_decisions(valid_probabilities, bundle.classes, priors,
                                             candidate.get('decision_power', 0.0))
                test_prediction = class_decisions(test_probabilities, bundle.classes, priors,
                                                  candidate.get('decision_power', 0.0))
            else:
                model = build_pipeline(bundle.task_type, candidate, request.get('threads', 4))
                fit_started = time.monotonic()
                model.fit(bundle.X.iloc[fit], fit_labels)
                fit_seconds = time.monotonic() - fit_started
                model_audit = getattr(model, 'audit_', {})
                generated = getattr(model, 'feature_code_', '')
                if cache_eligible:
                    priors = train_priors(fit_labels, bundle.classes)
                    valid_probabilities = aligned_probabilities(model, bundle.X.iloc[valid], bundle.classes)
                    test_probabilities = aligned_probabilities(model, bundle.X_test, bundle.classes)
                    prediction = class_decisions(valid_probabilities, bundle.classes, priors,
                                                 candidate.get('decision_power', 0.0))
                    test_prediction = class_decisions(test_probabilities, bundle.classes, priors,
                                                      candidate.get('decision_power', 0.0))
                    cache_payload = (fit, valid, test_positions, priors, valid_probabilities, test_probabilities)
                else:
                    prediction = predict(model, bundle, bundle.X.iloc[valid], fit_labels)
            value = score(bundle.metric, bundle.y.iloc[valid], prediction,
                          classes=bundle.classes, positive_class=bundle.positive_class)
            if not np.isfinite(value):
                raise ValueError('Validation metric is not finite')
            result = {'status': 'completed', 'metric': bundle.metric, 'score': value,
                'fit_rows': len(fit), 'valid_rows': len(valid),
                'diagnostics': diagnostics(bundle, valid, prediction, model, fit_labels)}
        elif request['mode'] == 'refit':
            model = build_pipeline(bundle.task_type, candidate, request.get('threads', 4))
            fit_labels = bundle.y
            fit_started = time.monotonic()
            model.fit(bundle.X, fit_labels)
            fit_seconds = time.monotonic() - fit_started
            model_audit = getattr(model, 'audit_', {})
            generated = getattr(model, 'feature_code_', '')
            result = {'status': 'completed', 'mode': 'refit', 'train_rows': len(bundle.train)}
        else:
            raise ValueError('Unknown worker mode')
        # Every completed model, including the first small baseline, has usable test output.
        if test_prediction is None:
            test_prediction = predict(model, bundle, bundle.X_test, fit_labels)
        if request['mode'] in ('evaluate', 'audit'):
            np.savez_compressed(output / 'validation_predictions.npz', positions=valid,
                prediction=prediction.astype(str) if prediction.dtype.kind == 'O' else prediction)
        atomic_submission(bundle, make_submission(bundle, test_prediction), output / 'candidate_submission.csv')
        if generated:
            (output / 'generated_features.py').write_text(generated, encoding='utf-8')
        if cache_payload is not None:
            fit, valid, test_positions, priors, valid_probabilities, test_probabilities = cache_payload
            _save_probability_cache(output / 'probability_cache.npz', candidate, fit, valid,
                                   test_positions, bundle.classes, priors,
                                   valid_probabilities, test_probabilities, model)
        result.update(seconds=time.monotonic() - started, fit_seconds=fit_seconds,
                      test_rows=len(bundle.test),
                      model_audit=model_audit, generated_code=bool(generated),
                      training_reused=training_reused)
    except Exception as exc:
        (output / 'candidate_submission.csv').unlink(missing_ok=True)
        (output / 'probability_cache.npz').unlink(missing_ok=True)
        (output / 'validation_predictions.npz').unlink(missing_ok=True)
        result = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc),
                  'seconds': time.monotonic() - started}
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    save_json(output / 'metrics.json', result)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
