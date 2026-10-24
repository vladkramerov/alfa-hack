"""Isolated experiment executor; imports only modules shipped in the flat ZIP."""
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from automl_data import load_task, score, make_submission, atomic_submission
from automl_models import build_pipeline, canonical_candidate
from automl_runtime import save_json


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


def predict(model, bundle, X):
    if bundle.task_type == 'classification' and bundle.metric in ('roc_auc', 'log_loss'):
        return aligned_probabilities(model, X, bundle.classes)
    result = np.asarray(model.predict(X)).reshape(-1)
    if bundle.task_type == 'regression' and not np.isfinite(result).all():
        raise ValueError('Nonfinite regression prediction')
    return result


def diagnostics(bundle, valid, prediction, model):
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
        return {'per_class': report, 'largest_confusions': sorted(pairs, reverse=True)[:10],
                'missing_fit_classes': [str(c) for c in labels if c not in model.classes_]}
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
        candidate = canonical_candidate(request['candidate'])
        model = build_pipeline(bundle.task_type, candidate, request.get('threads', 4))
        if request['mode'] in ('evaluate', 'audit'):
            with np.load(request['split_path']) as split:
                fit, valid = split['fit'], split['valid']
            fit_started = time.monotonic()
            model.fit(bundle.X.iloc[fit], bundle.y.iloc[fit])
            fit_seconds = time.monotonic() - fit_started
            prediction = predict(model, bundle, bundle.X.iloc[valid])
            value = score(bundle.metric, bundle.y.iloc[valid], prediction,
                          classes=bundle.classes, positive_class=bundle.positive_class)
            if not np.isfinite(value):
                raise ValueError('Validation metric is not finite')
            result = {'status': 'completed', 'metric': bundle.metric, 'score': value,
                'fit_rows': len(fit), 'valid_rows': len(valid),
                'diagnostics': diagnostics(bundle, valid, prediction, model)}
            np.savez_compressed(output / 'validation_predictions.npz', positions=valid,
                prediction=prediction.astype(str) if prediction.dtype.kind == 'O' else prediction)
        elif request['mode'] == 'refit':
            fit_started = time.monotonic()
            model.fit(bundle.X, bundle.y)
            fit_seconds = time.monotonic() - fit_started
            result = {'status': 'completed', 'mode': 'refit', 'train_rows': len(bundle.train)}
        else:
            raise ValueError('Unknown worker mode')
        # Every completed model, including the first small baseline, has usable test output.
        test_prediction = predict(model, bundle, bundle.X_test)
        atomic_submission(bundle, make_submission(bundle, test_prediction), output / 'candidate_submission.csv')
        generated = getattr(model, 'feature_code_', '')
        if generated:
            (output / 'generated_features.py').write_text(generated, encoding='utf-8')
        result.update(seconds=time.monotonic() - started, fit_seconds=fit_seconds,
                      test_rows=len(bundle.test),
                      model_audit=getattr(model, 'audit_', {}), generated_code=bool(generated))
    except Exception as exc:
        result = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc),
                  'seconds': time.monotonic() - started}
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    save_json(output / 'metrics.json', result)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
