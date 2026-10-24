"""Compare four frozen regression candidates on inner temporal periods 3 and 4.

Only the established task train data is used. The newest period is peeled off
to locate the inner windows; its target is never fitted or scored.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from automl_data import load_task, score, split_key, validation_split
from automl_models import build_pipeline, canonical_candidate


THREADS = 4
EXPECTED_INNER_PERIODS = {'3', '4'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_for(bundle):
    candidates = []
    for validation_path in (bundle.task_path.parent / 'agent_runs').glob('*/validation.json'):
        try:
            validation = json.loads(validation_path.read_text(encoding='utf-8'))
            if validation.get('data_hashes') == bundle.hashes:
                profile_path = validation_path.with_name('profile.json')
                if profile_path.is_file():
                    candidates.append((validation_path.parent.name, profile_path, validation))
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError('No profile.json with matching task data hashes found under task-dir/agent_runs')
    stamp, profile_path, validation = sorted(candidates)[-1]
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    return profile, profile_path, validation, stamp


def _candidate_set(profile):
    frequency_columns = sorted(
        str(column) for column, count in profile.get('categories', {}).items()
        if 3 <= int(count) <= 10000
    )[:8]
    drift_columns = sorted(set(map(str, profile.get('drift_columns', []))))
    frequency_columns = [column for column in frequency_columns if column not in drift_columns]
    raw = {
        'extra_trees_log1p_control': {
            'family': 'extra_trees', 'target_transform': 'log1p',
            'parameters': {'n_estimators': 200, 'max_depth': 24,
                           'min_samples_leaf': 2, 'max_features': .8},
            'hypothesis': 'Frozen historical control: ExtraTrees with log1p target.',
        },
        'catboost_onehot': {
            'family': 'catboost', 'catboost_encoding': 'onehot',
            'parameters': {'iterations': 500, 'depth': 6},
            'hypothesis': 'Compare categorical one-hot encoding on the same temporal windows.',
        },
        'catboost_native': {
            'family': 'catboost', 'catboost_encoding': 'native',
            'parameters': {'iterations': 500, 'depth': 6},
            'hypothesis': 'Compare native categorical encoding on the same temporal windows.',
        },
        'catboost_native_frequency_drop_drift': {
            'family': 'catboost', 'catboost_encoding': 'native',
            'parameters': {'iterations': 500, 'depth': 6},
            'frequency_columns': frequency_columns,
            'drop_columns': drift_columns,
            'hypothesis': 'Apply fit-only profile category frequencies and profile drift-column removal.',
        },
    }
    return {name: canonical_candidate(candidate) for name, candidate in raw.items()}, {
        'frequency_columns': frequency_columns,
        'profile_drift_columns': drift_columns,
    }


def _inner_folds(bundle):
    """Peel newest period, then obtain and retain only period 4 and 3 splits."""
    current = bundle
    original_positions = np.arange(len(bundle.train), dtype=np.int64)
    observed_periods = []
    outer_discard = None
    selected_folds = []
    # Three calls are deliberate: newest period is excluded, then two inner windows.
    for depth in range(3):
        fit, valid, description = validation_split(current)
        if not description.startswith('temporal:') or ':train_last=' not in description:
            raise ValueError(f'Expected declared temporal split, got {description!r}')
        period = description.split(':train_last=', 1)[1].split(':', 1)[0]
        observed_periods.append(period)
        fit_global = original_positions[fit]
        valid_global = original_positions[valid]
        if depth == 0:
            # This split only removes the latest time slice. Do not index its y,
            # fit a model on it, calculate a metric, or put its target in output.
            outer_discard = {'period': period, 'rows': int(len(valid)),
                             'split_key': split_key(fit_global, valid_global)}
        else:
            selected_folds.append({
                'period': period,
                'description': description,
                'fit': fit_global,
                'valid': valid_global,
                'split_key': split_key(fit_global, valid_global),
            })
        if depth < 2:
            current = replace(current, train=current.train.iloc[fit].reset_index(drop=True))
            original_positions = fit_global
    if set(observed_periods[1:]) != EXPECTED_INNER_PERIODS:
        raise ValueError(f'Inner periods 3 and 4 could not both be derived; observed {observed_periods!r}')
    return sorted(selected_folds, key=lambda fold: float(fold['period'])), outer_discard, observed_periods


def run(task_dir, output):
    started = time.monotonic()
    bundle = load_task(Path(task_dir) / 'task.md')
    if bundle.task_type != 'regression' or bundle.metric != 'rmse':
        raise ValueError(f'Expected the public regression RMSE task, got {bundle.task_type}/{bundle.metric}')
    profile, profile_path, profile_validation, profile_stamp = _profile_for(bundle)
    candidates, profile_features = _candidate_set(profile)
    folds, outer_discard, all_periods = _inner_folds(bundle)

    fold_results = []
    by_candidate = {name: [] for name in candidates}
    for fold in folds:
        fit, valid = fold['fit'], fold['valid']
        X_fit, y_fit = bundle.X.iloc[fit], bundle.y.iloc[fit]
        X_valid, y_valid = bundle.X.iloc[valid], bundle.y.iloc[valid]
        per_candidate = {}
        for name, candidate in candidates.items():
            model = build_pipeline('regression', candidate, THREADS)
            fit_started = time.monotonic()
            model.fit(X_fit, y_fit)
            fit_seconds = time.monotonic() - fit_started
            train_prediction = np.asarray(model.predict(X_fit), dtype=float)
            valid_prediction = np.asarray(model.predict(X_valid), dtype=float)
            result = {
                'candidate': candidate,
                'train_rmse': score('rmse', y_fit.to_numpy(), train_prediction),
                'validation_rmse': score('rmse', y_valid.to_numpy(), valid_prediction),
                'fit_seconds': fit_seconds,
                'fit_rows': int(len(fit)),
                'validation_rows': int(len(valid)),
            }
            per_candidate[name] = result
            by_candidate[name].append(result['validation_rmse'])
        fold_results.append({
            'period': fold['period'],
            'split_description': fold['description'],
            'split_key': fold['split_key'],
            'fit_indices': fold['fit'].tolist(),
            'validation_indices': fold['valid'].tolist(),
            'candidates': per_candidate,
        })

    averages = {name: float(np.mean(values)) for name, values in by_candidate.items()}
    ranking_two_periods = sorted(averages, key=lambda name: averages[name])
    first_fold = next(fold for fold in fold_results if fold['period'] == '3')
    last_fold = next(fold for fold in fold_results if fold['period'] == '4')
    ranking_period3 = sorted(first_fold['candidates'],
                             key=lambda name: first_fold['candidates'][name]['validation_rmse'])
    ranking_period4 = sorted(last_fold['candidates'],
                             key=lambda name: last_fold['candidates'][name]['validation_rmse'])
    single_period_winner = ranking_period4[0]
    two_period_winner = ranking_two_periods[0]
    profile_hash = _sha256(profile_path)
    source_paths = [Path(__file__), Path(__file__).parents[1] / 'automl_data.py',
                    Path(__file__).parents[1] / 'automl_models.py']
    result = {
        'status': 'completed',
        'study': 'frozen four-candidate ranking on inner periods 3 and 4',
        'selection_rule': 'Equal-weight mean of period-3 and period-4 validation RMSE; period 5 is not scored or used for fitting.',
        'period_recursion': all_periods,
        'excluded_latest_period': outer_discard,
        'folds': fold_results,
        'candidates': candidates,
        'candidate_profile_features': profile_features,
        'average_validation_rmse': averages,
        'ranking_by_period3_only': ranking_period3,
        'ranking_by_period4_only': ranking_period4,
        'ranking_by_two_period_mean': ranking_two_periods,
        'winner_by_period3_only': ranking_period3[0],
        'winner_by_period4_only': ranking_period4[0],
        'winner_by_two_period_mean': ranking_two_periods[0],
        'ranking_changed': ranking_period4[0] != ranking_two_periods[0],
        'architecture_signal': {
            'single_period4_winner_mean_rmse_regret': float(averages[single_period_winner] - averages[two_period_winner]),
            'interpretation': ('The best candidate changes when period 3 is included; this supports testing a generic multi-window stability criterion, '
                               'but one task and two periods are insufficient to switch production selection policy.'
                               if ranking_period4[0] != ranking_two_periods[0]
                               else 'The same candidate wins period 4 and the two-period mean; this study does not show a ranking benefit from adding period 3.'),
        },
        'profile_source': {'run': profile_stamp,
                           'validation': profile_validation.get('search'),
                           'sha256': profile_hash},
        'data_hashes': bundle.hashes,
        'source_hashes': {str(path.relative_to(Path(__file__).parents[1])): _sha256(path)
                          for path in source_paths},
        'threads_per_candidate': THREADS,
        'elapsed_seconds': time.monotonic() - started,
        'interpretation': {
            'motivation': 'Test whether selection on one recent inner period is sensitive to the immediately preceding period; the two-period mean is a stability check, not tuning against the outer period.',
            'limitations': ['Only two historical validation periods are compared.',
                            'The profile-driven candidate uses the already saved fit-only profile; it is a fixed recipe for this study.',
                            'Neither result estimates future-period or hidden-test performance.'],
        },
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True,
                        help='Regression task directory containing task.md, data CSVs, and matching agent_runs profile')
    parser.add_argument('--output', default='results/rolling-study.json')
    args = parser.parse_args()
    result = run(args.task_dir, args.output)
    print(json.dumps({'status': result['status'],
                      'winner_period3': result['winner_by_period3_only'],
                      'winner_period4': result['winner_by_period4_only'],
                      'winner_two_period_mean': result['winner_by_two_period_mean'],
                      'ranking_changed': result['ranking_changed'],
                      'elapsed_seconds': round(result['elapsed_seconds'], 2)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
