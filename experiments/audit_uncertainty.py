"""Paired cluster-bootstrap intervals over already saved audit predictions.

This script never fits models and never reads test targets. It resamples exact
duplicate validation feature rows as groups, preserving paired predictions.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_data import load_task
from automl_models import candidate_key


REPLICATES = 200
SEED = 20261023


def _label_indices(values, classes):
    exact = {}
    by_text = {}
    ambiguous_text = set()
    for index, label in enumerate(classes):
        try:
            exact[label] = index
        except TypeError:
            pass
        text = str(label)
        if text in by_text:
            ambiguous_text.add(text)
        else:
            by_text[text] = index
    result = np.empty(len(values), dtype=np.int64)
    for i, raw in enumerate(values):
        value = raw.item() if isinstance(raw, np.generic) else raw
        try:
            if value in exact:
                result[i] = exact[value]
                continue
        except TypeError:
            pass
        text = str(value)
        if text in ambiguous_text or text not in by_text:
            raise ValueError(f'Prediction label is not uniquely aligned to task classes: {text!r}')
        result[i] = by_text[text]
    return result


def _macro_f1_from_counts(actual, predicted, weights, n_classes):
    support = np.bincount(actual, weights=weights, minlength=n_classes)
    predicted_count = np.bincount(predicted, weights=weights, minlength=n_classes)
    correct = actual == predicted
    true_positive = np.bincount(actual[correct], weights=weights[correct], minlength=n_classes)
    denominator = support + predicted_count
    per_class = np.divide(2 * true_positive, denominator, out=np.zeros(n_classes, dtype=float),
                          where=denominator > 0)
    return float(per_class.mean())


def paired_cluster_bootstrap(y, baseline_prediction, selected_prediction, group_hashes,
                             metric, classes=None, replicates=REPLICATES, seed=SEED):
    """Return selected-minus-baseline delta and percentile CI without fitting."""
    y = np.asarray(y)
    baseline_prediction = np.asarray(baseline_prediction)
    selected_prediction = np.asarray(selected_prediction)
    group_hashes = np.asarray(group_hashes)
    n = len(y)
    if n == 0 or any(len(value) != n for value in
                     (baseline_prediction, selected_prediction, group_hashes)):
        raise ValueError('Paired targets, predictions, and groups must have equal nonzero length')
    if replicates < 1:
        raise ValueError('replicates must be positive')
    unique_groups, group_index = np.unique(group_hashes, return_inverse=True)
    n_groups = len(unique_groups)
    rng = np.random.default_rng(seed)

    if metric == 'rmse':
        actual = y.astype(float)
        baseline = baseline_prediction.astype(float)
        selected = selected_prediction.astype(float)
        if not (np.isfinite(actual).all() and np.isfinite(baseline).all()
                and np.isfinite(selected).all()):
            raise ValueError('RMSE inputs must be finite')
        base_error = np.square(actual - baseline)
        selected_error = np.square(actual - selected)
        baseline_value = float(np.sqrt(base_error.mean()))
        selected_value = float(np.sqrt(selected_error.mean()))

        def delta_for(weights):
            mass = weights.sum()
            return float(np.sqrt(np.dot(selected_error, weights) / mass)
                         - np.sqrt(np.dot(base_error, weights) / mass))

    elif metric == 'macro_f1':
        labels = tuple(classes or ())
        if not labels:
            labels = tuple(pd.unique(y))
        actual = _label_indices(y, labels)
        baseline = _label_indices(baseline_prediction, labels)
        selected = _label_indices(selected_prediction, labels)
        unit_weights = np.ones(n, dtype=float)
        baseline_value = _macro_f1_from_counts(actual, baseline, unit_weights, len(labels))
        selected_value = _macro_f1_from_counts(actual, selected, unit_weights, len(labels))

        def delta_for(weights):
            return (_macro_f1_from_counts(actual, selected, weights, len(labels))
                    - _macro_f1_from_counts(actual, baseline, weights, len(labels)))

    else:
        raise ValueError('Only rmse and macro_f1 are supported')

    samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, n_groups, size=n_groups)
        multiplicity = np.bincount(sampled, minlength=n_groups)
        samples[replicate] = delta_for(multiplicity[group_index])
    delta = selected_value - baseline_value
    low, high = np.quantile(samples, [.025, .975])
    return {
        'metric': metric,
        'baseline': baseline_value,
        'selected': selected_value,
        'delta_selected_minus_baseline': delta,
        'ci95_delta': [float(low), float(high)],
        'bootstrap_mean_delta': float(samples.mean()),
        'bootstrap_standard_error': float(samples.std(ddof=1)) if replicates > 1 else 0.,
        'replicates': replicates,
        'seed': seed,
        'rows': n,
        'groups': n_groups,
    }


def _read_predictions(run_dir, audit_item):
    relative = audit_item.get('directory')
    if not relative:
        return None
    path = (run_dir / relative / 'validation_predictions.npz').resolve()
    if run_dir.resolve() not in path.parents or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if not {'positions', 'prediction'}.issubset(archive.files):
            return None
        return archive['positions'].copy(), archive['prediction'].copy()


def _analyze_run(bundle, run_dir, summary, validation):
    metric = bundle.metric
    if metric not in {'rmse', 'macro_f1'}:
        return {'status': 'unsupported_metric', 'metric': metric, 'run': str(run_dir)}
    selected_candidate = summary.get('selected')
    audits = summary.get('audit') or []
    if not selected_candidate or not audits:
        return {'status': 'pending', 'metric': metric, 'run': str(run_dir),
                'reason': 'No frozen selected-candidate audit is recorded.'}
    try:
        selected_key = candidate_key(selected_candidate)
        selected_item = next((item for item in audits
                              if item.get('status') == 'completed'
                              and candidate_key(item.get('candidate', {})) == selected_key), None)
    except (TypeError, ValueError):
        selected_item = None
    baseline_item = audits[0] if audits and audits[0].get('status') == 'completed' else None
    if selected_item is None or baseline_item is None:
        return {'status': 'pending', 'metric': metric, 'run': str(run_dir),
                'reason': 'Saved baseline or frozen selected audit prediction is missing.'}
    if candidate_key(baseline_item['candidate']) == selected_key:
        return {'status': 'not_applicable', 'metric': metric, 'run': str(run_dir),
                'reason': 'Frozen search winner is the baseline; there is no distinct paired candidate.'}

    baseline_data = _read_predictions(run_dir, baseline_item)
    selected_data = _read_predictions(run_dir, selected_item)
    if baseline_data is None or selected_data is None:
        return {'status': 'pending', 'metric': metric, 'run': str(run_dir),
                'reason': 'Paired audit prediction files are incomplete.'}
    baseline_positions, baseline_prediction = baseline_data
    selected_positions, selected_prediction = selected_data
    if not np.array_equal(baseline_positions, selected_positions):
        return {'status': 'invalid_pair', 'metric': metric, 'run': str(run_dir),
                'reason': 'Baseline and selected predictions use different validation positions.'}
    if (len(baseline_positions) == 0 or baseline_positions.min() < 0
            or baseline_positions.max() >= len(bundle.y)
            or len(np.unique(baseline_positions)) != len(baseline_positions)):
        return {'status': 'invalid_positions', 'metric': metric, 'run': str(run_dir)}

    y = bundle.y.iloc[baseline_positions].to_numpy()
    raw_features = bundle.X.iloc[baseline_positions].reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(raw_features, index=False).to_numpy(dtype=np.uint64)
    result = paired_cluster_bootstrap(y, baseline_prediction, selected_prediction,
        row_hashes, metric, classes=bundle.classes)
    result.update(status='completed', run=str(run_dir),
        search_selected=selected_candidate,
        baseline_candidate=baseline_item['candidate'],
        validation=summary.get('audit_validation'),
        audit_validation_key=validation.get('audit_key'),
        audit_used_for_selection=summary.get('audit_used_for_selection'),
        uncertainty_scope=('conditional within the audited year-5 period; not uncertainty for future years'
                           if str(summary.get('audit_validation', '')).startswith('temporal:')
                           else 'conditional on this outer validation split'),
        grouping='bootstrap exact duplicate raw validation feature rows as clusters',
        limitations=['Unknown temporal, site, household, or other flow dependencies are not represented unless they produce identical raw feature rows.',
                     'The interval describes this validation population; it does not estimate task-to-task or future-regime uncertainty.'])
    return result


def analyze(task_dir, runs_root, output):
    task_path = Path(task_dir) / 'task.md'
    bundle = load_task(task_path)
    expected_hashes = bundle.hashes
    analyses = []
    for summary_path in sorted(Path(runs_root).rglob('summary.json')):
        run_dir = summary_path.parent
        validation_path = run_dir / 'validation.json'
        if not validation_path.is_file():
            continue
        try:
            validation = json.loads(validation_path.read_text(encoding='utf-8'))
            if validation.get('data_hashes') != expected_hashes:
                continue
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        analyses.append(_analyze_run(bundle, run_dir, summary, validation))
    usable = [item for item in analyses if item['status'] == 'completed']
    output_value = {
        'status': 'completed' if usable else ('pending' if not analyses or
            all(item['status'] == 'pending' for item in analyses) else 'no_comparable_pair'),
        'task_hashes': expected_hashes,
        'replicates': REPLICATES,
        'seed': SEED,
        'analyses': analyses,
        'method': 'Paired percentile cluster bootstrap over saved audit predictions; no model fitting.',
        'test_targets_read': False,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return output_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True, help='Directory containing task.md and its linked data files')
    parser.add_argument('--runs-root', default='/private/tmp/alfa-public-v2', help='Root to search for saved run summaries')
    parser.add_argument('--output', required=True, help='JSON path for the interval report')
    args = parser.parse_args()
    result = analyze(args.task_dir, args.runs_root, args.output)
    print(json.dumps({'status': result['status'], 'analyses': len(result['analyses']),
                      'completed': sum(x['status'] == 'completed' for x in result['analyses'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
