"""Result-driven AutoResearch with rolling search windows, nested audit and bounded workers."""
from dataclasses import replace
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from automl_data import (load_task, validation_split, temporal_windows, split_key, better,
                         fallback_submission, bootstrap_submission, atomic_submission,
                         validate_submission, LOWER_IS_BETTER)
from automl_models import canonical_candidate, candidate_key
from automl_runtime import Budget, run_child, save_json


def profile_data(bundle, fit, validation):
    X, y = bundle.X.iloc[fit], bundle.y.iloc[fit]
    numeric = X.select_dtypes(include='number').replace([np.inf, -np.inf], np.nan)
    correlations = {}
    if bundle.task_type == 'regression':
        for c in numeric:
            if numeric[c].nunique() > 1 and y.nunique() > 1:
                value = numeric[c].corr(y)
                if np.isfinite(value):
                    correlations[str(c)] = float(value)
    summary = (json.loads(numeric.describe(percentiles=[.01, .5, .99])
                          .replace([np.inf, -np.inf], np.nan).to_json())
               if len(numeric.columns) else {})
    shift, drift_columns = {}, []
    test = bundle.X_test
    for column in numeric:
        train_values = numeric[column].dropna()
        test_values = pd.to_numeric(test[column], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if train_values.nunique() < 10 or not len(test_values):
            continue
        q01, q25, median, q75, q99 = train_values.quantile([.01, .25, .5, .75, .99])
        scale = max(float(q75 - q25), float(train_values.std()) * .5, 1e-9)
        outside = float(((test_values < q01) | (test_values > q99)).mean())
        location = abs(float(test_values.median()) - float(median)) / scale
        if outside > .2 or location > 1:
            shift[str(column)] = {'outside_fit_1_99': outside, 'median_shift_iqr': location}
        if outside > .5 and location > 1:
            drift_columns.append(str(column))
    # Preserve enough predictors; this is a measured hypothesis, never compulsory cleaning.
    drift_columns.sort(key=lambda c: shift[c]['median_shift_iqr'], reverse=True)
    drift_columns = drift_columns[:min(40, max(1, int(X.shape[1] * .6)))]
    return {'task': bundle.text, 'target': bundle.target, 'id': bundle.id_column,
        'type': bundle.task_type, 'metric': bundle.metric, 'validation': validation,
        'shape': list(X.shape), 'dtypes': X.dtypes.astype(str).to_dict(),
        'missing': X.isna().mean().to_dict(), 'nunique': X.nunique().to_dict(),
        'categories': {str(c): int(X[c].nunique()) for c in X.select_dtypes(exclude='number')},
        'numeric_summary': summary,
        'distribution_shift': dict(sorted(shift.items(), key=lambda item: item[1]['median_shift_iqr'], reverse=True)[:24]),
        'drift_columns': drift_columns,
        'numeric_target_correlations': dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:8]),
        'target_fit_summary': (y.value_counts().to_dict() if bundle.task_type == 'classification'
                               else json.loads(y.describe(percentiles=[.01, .5, .99]).to_json())),
        'target_min': float(y.min()) if bundle.task_type == 'regression' else None,
        'target_skew': float(y.skew()) if bundle.task_type == 'regression' and len(y) > 2 and np.isfinite(y.skew()) else 0.,
        'classes': list(bundle.classes) if bundle.classes else None,
        'positive_class': bundle.positive_class,
        'time_column': validation.split(':')[1] if validation.startswith('temporal:') else None,
        'validation_warning': 'Search split is reused adaptively; audit targets are withheld until proposals stop.'}


def _expert_options(profile):
    """Group columns with enough fit rows and a bounded number of values."""
    rows = profile['shape'][0]
    options = []
    for column, cardinality in profile['categories'].items():
        if not 2 <= cardinality <= 40 or rows / max(cardinality, 1) < 300:
            continue
        missing = profile.get('missing', {}).get(column, 0.)
        if missing > .4:
            continue
        # Prefer compact, nearly balanced categorical structures.
        options.append((abs(cardinality - 8), -rows / cardinality, column, cardinality))
    options.sort()
    return [(column, cardinality) for _, _, column, cardinality in options]


def offline_proposal(profile, observations, seen):
    """Fallback policies branch on measured errors and fit-only data, not dataset names."""
    completed = [r for r in observations if r.get('status') == 'completed' and 'score' in r]
    ordered = sorted(completed, key=lambda r: r['score'], reverse=profile['metric'] not in LOWER_IS_BETTER)
    best = ordered[0]['candidate'] if ordered else {'family': 'lightgbm'}
    proposals = []
    families = {r['candidate']['family'] for r in completed}
    classification = profile['type'] == 'classification'
    classifier_research = classification and profile.get('classification_research', False)
    if not completed:
        proposals.append({'family': 'lightgbm', 'parameters': {'n_estimators': 120},
                          'hypothesis': 'Quick native-category boosting baseline before research.'})
    strong_linear_signal = (not classification and
        any(abs(value) > .8 for value in profile['numeric_target_correlations'].values()))
    if completed and 'linear' not in families and (strong_linear_signal or
            profile['shape'][1] > max(20, profile['shape'][0] * .03)):
        proposals.append({'family': 'linear',
            'hypothesis': 'Fit-only linear association or feature-to-row ratio justifies a cheap regularized linear comparison.'})
    if not classification and profile.get('time_column'):
        proposals.extend([
            {'family': 'extra_trees', 'target_transform': 'log1p' if profile['target_min'] > -1 else 'none',
             'panel_time_column': profile['time_column'],
             'hypothesis': 'Discover recurrent entity keys using fit features and test strictly-past history with randomized trees.'},
            {'family': 'lightgbm', 'parameters': {'n_estimators': 500, 'reg_lambda': 2},
             'panel_time_column': profile['time_column'],
             'hypothesis': 'Compare direct-loss boosting on automatically discovered causal history; cold starts retain raw features.'},
        ])
    if classification:
        if classifier_research and profile['metric'] == 'macro_f1':
            frequency = [c for c, n in profile['categories'].items() if 3 <= n <= 10000][:8]
            proposals.append({'family': 'xgboost', 'parameters': {'n_estimators': 600},
                'frequency_columns': frequency, 'decision_power': .3,
                'hypothesis': 'Establish a strong frequency-aware histogram control before spending budget on exploratory hypotheses.'})
        if 'xgboost' not in families:
            proposals.append({'family': 'xgboost', 'parameters': {'n_estimators': 600},
                              'hypothesis': 'Test histogram boosting against the measured baseline.'})
        if classifier_research and profile['metric'] == 'macro_f1':
            proposals.append({'family': 'lightgbm', 'early_stopping': True,
                'parameters': {'n_estimators': 1200, 'num_leaves': 63, 'min_child_samples': 10,
                               'reg_lambda': 3, 'subsample': .8, 'colsample_bytree': .85},
                'class_weight_power': .2,
                'hypothesis': 'Compare stronger native boosting with grouped train-only Macro-F1 stopping and actual row subsampling.'})
        diagnostic = ordered[0].get('diagnostics', {}).get('per_class', {}) if ordered else {}
        weak = [k for k, v in diagnostic.items() if isinstance(v, dict) and v.get('support', 0) > 0 and v.get('f1-score', 1) < .45]
        if weak and profile['metric'] in ('macro_f1', 'binary_f1', 'balanced_accuracy'):
            for power in (.15, .3, .5):
                proposals.append({**best, 'decision_power': power,
                    'hypothesis': f'Low per-class F1 for {weak[:4]}; test soft training-prior decision correction {power}.'})
            proposals.append({**best, 'class_weight_power': .35, 'decision_power': 0.,
                              'hypothesis': 'Low minority recall: use soft inverse-frequency training weights instead of full balancing.'})
        if 'catboost' not in families:
            proposals.append({'family': 'catboost', 'parameters': {'iterations': 500},
                              'hypothesis': 'Compare native categorical boosting after observed class confusions.'})
    else:
        if profile['target_min'] > -1 and profile['target_skew'] > 1:
            proposals.append({'family': 'extra_trees', 'target_transform': 'log1p',
                'hypothesis': 'Fit-only target is right-skewed; compare log-target randomized trees with boosting.'})
        if (profile['metric'] == 'rmse' and best['family'] == 'extra_trees'
                and best.get('target_transform') == 'log1p'
                and best.get('numeric_view', 'none') == 'none'):
            proposals.append({**best, 'numeric_view': 'signed_log',
                'hypothesis': 'Append signed-log numeric views to raw features so randomized thresholds can represent relative and absolute scales; retain only on measured inner gain.'})
        experts = _expert_options(profile)
        if best['family'] not in ('blend', 'linear'):
            for column, cardinality in experts[:2]:
                proposals.append({**best, 'expert_column': column,
                    'expert_min_rows': max(300, int(profile['shape'][0] / cardinality * .5)),
                    'max_expert_groups': min(32, cardinality),
                    'hypothesis': f'Test per-{column} experts with a global fallback; interactions may be hidden from one global tree.'})
        if 'catboost' not in families:
            proposals.append({'family': 'catboost', 'parameters': {'iterations': 500},
                              'hypothesis': 'Compare symmetric native-category trees for measured residual structure.'})
        if 'xgboost' not in families:
            proposals.append({'family': 'xgboost', 'parameters': {'n_estimators': 600},
                              'hypothesis': 'Compare histogram boosting for residual bias and tail errors.'})
        if profile['target_min'] > -1 and profile['target_skew'] > 1:
            proposals.append({**best, 'target_transform': 'none',
                              'hypothesis': 'Check whether inverse log transformation introduces bias for the primary loss.'})
    if profile.get('drift_columns') and best['family'] != 'blend':
        proposals.append({**best, 'drop_columns': sorted(set(best.get('drop_columns', [])) | set(profile['drift_columns'])),
            'hypothesis': 'Measured train/test covariate shift: test removal of severely shifted numeric predictors on the untouched search fold.'})
    if any(r['candidate']['family'] == 'catboost' for r in completed):
        proposals.append({'family': 'catboost', 'catboost_encoding': 'native',
            'hypothesis': 'Compare native categorical statistics with measured one-hot boosting, without assuming either encoding always wins.'})
    frequent = [c for c, cardinality in profile['categories'].items() if 3 <= cardinality <= 10000]
    if frequent:
        proposals.append({**best, 'frequency_columns': frequent[:8],
            'hypothesis': 'Add train-only category prevalence for categorical structure missed by the incumbent.'})
    if not classification and len(profile['numeric_target_correlations']) >= 2:
        left, right = list(profile['numeric_target_correlations'])[:2]
        proposals.append({**best, 'feature_operations': [
            {'op': 'mul', 'name': 'research_product', 'left': left, 'right': right},
            {'op': 'div', 'name': 'research_ratio', 'left': left, 'right': right}],
            'hypothesis': 'Test interactions between the two strongest fit-only numeric associations.'})
    if len(ordered) >= 2:
        diverse = next((r for r in ordered[1:] if r['candidate']['family'] != best['family'] and r['candidate']['family'] != 'blend'), None)
        if diverse and best['family'] != 'blend':
            proposals.append({'family': 'blend', 'members': [best, diverse['candidate']], 'weights': [.8, .2],
                'decision_power': best.get('decision_power', 0) if classification else 0,
                'hypothesis': 'Test a conservative blend of the two measured distinct model families.'})
            if (classifier_research and profile['metric'] == 'macro_f1'
                    and (best.get('early_stopping') or diverse['candidate'].get('early_stopping'))
                    and ordered[0]['score'] - diverse['score'] < .05):
                proposals.insert(0, {'family': 'blend', 'blend_mode': 'geometric',
                    'members': [best, diverse['candidate']], 'weights': [.5, .5], 'decision_power': .3,
                    'hypothesis': 'Test log-probability pooling of diverse measured classifiers; no OOF meta-model or fitted class thresholds.'})
    constants = [c for c, n in profile['nunique'].items() if n <= 1]
    if constants and len(constants) < profile['shape'][1]:
        proposals.append({**best, 'drop_columns': constants,
                          'hypothesis': 'Remove fit-constant features and measure robustness to uninformative columns.'})
    if 'extra_trees' not in families:
        proposals.append({'family': 'extra_trees', 'hypothesis': 'Compare random thresholds to observed boosting errors.'})
    for proposal in proposals:
        try:
            proposal = canonical_candidate(proposal)
            if candidate_key(proposal) not in seen:
                return proposal
        except (ValueError, TypeError):
            continue
    return None


def _search_windows(bundle, outer_fit, budget_seconds):
    """Return honest search windows strictly inside the outer fit rows.

    Temporal tasks use rolling origins (newest first). Non-temporal tasks keep
    the existing single nested split. Each entry carries its own weight and a
    stable key; no validation window is ever scored by the audit.
    """
    inner_bundle = replace(bundle, train=bundle.train.iloc[outer_fit].reset_index(drop=True))
    windows = temporal_windows(inner_bundle)
    if windows:
        if budget_seconds < 120:
            # A short run cannot fit every candidate twice without losing the
            # final refit; keep the newest origin only.
            windows = windows[:1]
        for window in windows:
            window['fit'] = outer_fit[window['fit']]
            window['valid'] = outer_fit[window['valid']]
        return windows, 'rolling-origin'
    inner_fit, inner_valid, description = validation_split(inner_bundle)
    if not len(inner_fit) or not len(inner_valid):
        raise ValueError('Empty inner split')
    return [{'fit': outer_fit[inner_fit], 'valid': outer_fit[inner_valid],
             'description': description, 'weight': 1.0}], 'single'


def main(task_path, started=None, threads=4):
    budget = Budget(os.getenv('AUTOML_TIME_LIMIT', '10200'), started)
    # Contract-critical: put a structurally valid file next to task.md before
    # parsing schemas, metrics, or models. It is replaced atomically later.
    bootstrap_submission(task_path)
    try:
        bundle = load_task(task_path)
    except Exception as exc:
        summary = {'status': 'bootstrap_only',
                   'error': type(exc).__name__ + ': ' + str(exc),
                   'metric': None, 'seconds': time.monotonic() - budget.started,
                   'submission_rows': None}
        print(json.dumps(summary, ensure_ascii=False))
        return summary
    atomic_submission(bundle, fallback_submission(bundle))
    first_submission = time.monotonic() - budget.started
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = bundle.task_path.parent / 'agent_runs' / stamp
    root.mkdir(parents=True)
    records, api_records, proposal_feedback = [], [], []
    best, best_score, baseline = None, None, None
    best_seconds, fit_seconds, duplicate_count, rejected_count = 0., 0., 0, 0
    first_model_seconds = None
    summary = {'status': 'fallback', 'metric': bundle.metric, 'first_submission_seconds': first_submission,
               'task_type': bundle.task_type, 'submission_rows': len(bundle.sample)}
    modules = Path(__file__).parent

    def execute(candidate, mode, split_path, label, timeout, reuse_probabilities=None, reuse_members=None):
        directory = root / label
        directory.mkdir()
        request = {'task_path': str(bundle.task_path), 'candidate': candidate, 'mode': mode,
                   'split_path': str(split_path), 'output': str(directory), 'threads': threads}
        if reuse_probabilities is not None:
            request['reuse_probabilities'] = str(reuse_probabilities)
        if reuse_members is not None:
            request['reuse_members'] = list(map(str, reuse_members))
        save_json(directory / 'request.json', request)
        process = run_child(modules / 'automl_worker.py', [directory / 'request.json'], directory,
                            min(timeout, budget.remaining()), threads)
        path = directory / 'metrics.json'
        result = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'status': process['status']}
        result.update(process=process, directory=str(directory.relative_to(root)))
        if process['status'] != 'completed':
            result['status'] = process['status']
        return result, directory

    def publish(directory):
        path = directory / 'candidate_submission.csv'
        if path.is_file():
            atomic_submission(bundle, pd.read_csv(path))

    try:
        outer_fit, outer_valid, outer_description = validation_split(bundle)
        if not len(outer_fit) or not len(outer_valid):
            raise ValueError('No honest validation split is possible; retained the valid fallback submission')
        default_policy = 'nested' if bundle.task_type == 'classification' else 'deployment'
        deployment = os.getenv('AUTOML_VALIDATION_POLICY', default_policy) != 'nested'
        nested = not deployment
        if deployment:
            windows = temporal_windows(bundle)
            if windows:
                if budget.seconds < 120:
                    windows = windows[:1]
                search_policy = 'deployment-rolling-origin'
            else:
                windows = [{'fit': outer_fit, 'valid': outer_valid,
                            'description': outer_description, 'weight': 1.0}]
                search_policy = 'deployment-group-or-random'
        else:
            try:
                windows, search_policy = _search_windows(bundle, outer_fit, budget.seconds)
            except ValueError:
                nested = False
                windows = [{'fit': outer_fit, 'valid': outer_valid,
                            'description': outer_description + ':no-independent-audit', 'weight': 1.0}]
                search_policy = 'single-shared'
        for index, window in enumerate(windows, 1):
            if nested and np.intersect1d(np.r_[window['fit'], window['valid']], outer_valid).size:
                raise ValueError('Search positions overlap audit; refuse contaminated validation')
            window['key'] = split_key(window['fit'], window['valid'])
            window['path'] = root / f'search_split-{index:02d}.npz'
            np.savez_compressed(window['path'], fit=window['fit'], valid=window['valid'])
        total_weight = sum(window['weight'] for window in windows)
        for window in windows:
            window['weight'] = window['weight'] / total_weight
        search_key = (windows[0]['key'] if len(windows) == 1 else
                      hashlib.sha256('|'.join(window['key'] for window in windows).encode()).hexdigest())
        # Profile only the oldest fit window: its rows are disjoint from every
        # search validation target, so proposal policies cannot read fold labels.
        profile_fit = windows[-1]['fit']
        inner_description = ' + '.join(window['description'] for window in windows)
        profile = profile_data(bundle, profile_fit, inner_description)
        if deployment:
            profile['validation_warning'] = 'Selection uses known training labels, including the latest available period. Independent evaluation requires an external masked test, never these search scores.'
        profile['validation_windows'] = [
            {'description': window['description'], 'fit_rows': len(window['fit']),
             'valid_rows': len(window['valid']), 'key': window['key'], 'weight': window['weight']}
            for window in windows]
        save_json(root / 'profile.json', profile)
        np.savez_compressed(root / 'audit_split.npz', fit=outer_fit, valid=outer_valid)
        save_json(root / 'validation.json', {'search': inner_description, 'audit': outer_description if nested else 'external evaluation required',
            'nested': nested, 'search_policy': search_policy, 'search_key': search_key,
            'audit_key': split_key(outer_fit, outer_valid) if nested else None,
            'search_fit_rows': int(sum(len(window['fit']) for window in windows)),
            'search_valid_rows': int(sum(len(window['valid']) for window in windows)),
            'audit_rows': len(outer_valid), 'search_folds': profile['validation_windows'],
            'search_aggregation': 'weighted mean across rolling windows' if len(windows) > 1 else 'single window',
            'data_hashes': bundle.hashes})
        seen = set()
        probability_cache = {}
        default_trials = '16' if bundle.task_type == 'classification' else '36'
        max_experiments = min(64, max(0, int(os.getenv('AUTOML_MAX_EXPERIMENTS', default_trials))))
        api_configured = all(os.getenv(k) for k in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL'))
        llm_failures, llm_tokens = 0, 0

        def trial_timeout():
            available = budget.remaining() - budget.reserve(fit_seconds)
            return max(.1, min(1200., max(30., budget.seconds * .12), available))

        def evaluate_windows(candidate, label, timeout):
            results, directories = [], []
            total_rows = sum(max(len(window['fit']), 1) for window in windows)
            training_key = candidate_key({**candidate, 'decision_power': 0})
            cache_allowed = (bundle.task_type == 'classification' and bundle.metric not in ('roc_auc', 'log_loss')
                             and candidate['family'] != 'blend')
            for index, window in enumerate(windows, 1):
                share = max(15., timeout * max(len(window['fit']), 1) / total_rows)
                reuse = probability_cache.get((training_key, index)) if cache_allowed else None
                member_paths = None
                if bundle.task_type == 'classification' and candidate['family'] == 'blend':
                    possible = [probability_cache.get((candidate_key({**m, 'decision_power': 0}), index))
                                for m in candidate['members']]
                    if all(possible):
                        member_paths = possible
                result, directory = execute(candidate, 'evaluate', window['path'],
                    f'{label}-fold{index:02d}', share, reuse, member_paths)
                results.append(result)
                directories.append(directory)
                cache_path = directory / 'probability_cache.npz'
                if cache_allowed and result['status'] == 'completed' and cache_path.is_file():
                    probability_cache[(training_key, index)] = cache_path
                if result['status'] != 'completed':
                    failed = dict(result)
                    failed['fold_results'] = results
                    failed['fold_descriptions'] = [w['description'] for w in windows]
                    failed['error'] = f"fold {index}: " + str(result.get('error', result['status']))
                    failed['fit_seconds'] = sum(float(r.get('fit_seconds', 0.)) for r in results)
                    failed['seconds'] = sum(float(r.get('seconds', 0.)) for r in results)
                    failed['process'] = {'status': result['status'], 'seconds': failed['seconds'],
                                         'peak_rss_mb': max(float(r.get('process', {}).get('peak_rss_mb', 0.)) for r in results)}
                    return failed, directory
            completed = all(result['status'] == 'completed' for result in results)
            if not completed:
                return results[-1], directories[-1]
            aggregated = dict(results[0])
            aggregated['status'] = 'completed'
            aggregated['score'] = float(sum(window['weight'] * result['score']
                                            for window, result in zip(windows, results)))
            aggregated['fold_results'] = results
            aggregated['fold_descriptions'] = [window['description'] for window in windows]
            aggregated['diagnostics'] = {
                **results[0].get('diagnostics', {}),
                'temporal_fold_scores': [result['score'] for result in results],
                'temporal_fold_weights': [window['weight'] for window in windows]}
            aggregated['validation_aggregation'] = (
                'weighted mean of rolling temporal windows' if len(windows) > 1 else 'single search window')
            aggregated['fit_seconds'] = sum(float(result.get('fit_seconds', 0.)) for result in results)
            aggregated['seconds'] = sum(float(result.get('seconds', 0.)) for result in results)
            aggregated['model_fit_count'] = sum(int(result.get('fit_seconds', 0.) > 0) for result in results)
            aggregated['process'] = {'status': 'completed', 'seconds': aggregated['seconds'],
                'peak_rss_mb': max(float(result.get('process', {}).get('peak_rss_mb', 0.)) for result in results)}
            save_json(directories[-1] / 'aggregate.json', aggregated)
            checkpoint = max(range(len(windows)), key=lambda i: len(windows[i]['fit']))
            aggregated['checkpoint_window'] = checkpoint + 1
            return aggregated, directories[checkpoint]

        for attempt in range(max_experiments * 3 + 1):
            if len(records) >= max_experiments or budget.remaining() <= budget.reserve(fit_seconds):
                break
            candidate, source = None, 'adaptive_fallback'
            portfolio_ready = max_experiments < 8 or len(records) >= 4
            llm_round = max_experiments < 8 or (len(records) - 4) % 3 == 0
            if records and portfolio_ready and llm_round and api_configured and len(api_records) < 4 and llm_failures < 3 and llm_tokens < 48000:
                call_dir = root / f'llm-{len(api_records) + 1:02d}'
                call_dir.mkdir()
                timeout = min(25., max(.1, (budget.remaining() - budget.reserve(fit_seconds)) / 2))
                live_profile = {**profile, 'remaining_seconds': budget.remaining(), 'best_search_score': best_score}
                save_json(call_dir / 'request.json', {'profile': live_profile,
                    'observations': records + proposal_feedback[-4:], 'timeout': timeout})
                process = run_child(modules / 'automl_llm.py', [call_dir / 'request.json', call_dir / 'response.json'],
                                    call_dir, timeout + min(2., budget.remaining()), threads=1, include_api=True)
                response = call_dir / 'response.json'
                payload = json.loads(response.read_text(encoding='utf-8')) if response.is_file() else {}
                api_status = payload.get('result', {'status': process['status']})
                api_status['process'] = process
                api_records.append(api_status)
                usage = api_status.get('usage') or {}
                llm_tokens += max(int(usage.get('total_tokens') or 0), int(api_status.get('estimated_tokens') or 0))
                candidate = payload.get('action')
                if candidate and candidate.get('stop') is True:
                    api_records[-1]['action'] = 'stop'
                    save_json(root / 'llm.json', api_records)
                    break
                if candidate:
                    source = 'llm'
                    llm_failures = 0
                else:
                    llm_failures += 1
                save_json(root / 'llm.json', api_records)
            if candidate is None:
                candidate = offline_proposal(profile, records, seen)
            if candidate is None:
                break
            try:
                candidate = canonical_candidate(candidate)
                if bundle.metric in ('roc_auc', 'log_loss'):
                    # Decision-only correction cannot change a probability-based objective.
                    candidate['decision_power'] = 0
                identity = candidate_key(candidate)
            except (ValueError, TypeError) as exc:
                rejected_count += 1
                feedback = {'status': 'candidate_rejected', 'error': str(exc)[:300]}
                proposal_feedback.append(feedback)
                save_json(root / 'proposal_feedback.json', proposal_feedback)
                continue
            if identity in seen:
                duplicate_count += 1
                proposal_feedback.append({'status': 'skipped', 'candidate_key': identity,
                                          'reason': 'Equivalent configuration already attempted; propose a different mechanism.'})
                continue
            seen.add(identity)
            record = {'id': f'RUN-{len(records) + 1:03d}', 'candidate': candidate,
                      'candidate_key': identity, 'source': source, 'status': 'started'}
            records.append(record)
            save_json(root / 'experiments.json', records)
            result, directory = evaluate_windows(candidate, f'experiment-{len(records):02d}', trial_timeout())
            record.update(result)
            fit_seconds = max(fit_seconds, float(result.get('seconds', 0.)))
            if result['status'] == 'completed' and better(bundle.metric, result['score'], best_score):
                publish(directory)
                best, best_score = candidate, result['score']
                best_seconds = time.monotonic() - budget.started
                if baseline is None:
                    baseline = candidate
                    first_model_seconds = best_seconds
                record['promoted'] = True
            else:
                record['promoted'] = False
            save_json(root / 'experiments.json', records)
        # Proposals stop before audit targets are ever scored. No return to search afterwards.
        audit = []
        selected = best
        if best is not None and nested and budget.remaining() > max(3., fit_seconds * 2.5):
            frozen = [best] if candidate_key(best) == candidate_key(baseline) else [baseline, best]
            save_json(root / 'frozen_selection.json', {'candidates': frozen, 'search_score': best_score})
            for index, candidate in enumerate(frozen):
                if budget.remaining() <= max(2., fit_seconds * 1.2):
                    break
                result, directory = execute(candidate, 'audit', root / 'audit_split.npz',
                    f'audit-{index + 1:02d}', max(.1, budget.remaining() - max(2., fit_seconds * 1.2)))
                audit.append({'candidate': candidate, **result})
                if result['status'] == 'completed' and candidate_key(candidate) == candidate_key(best):
                    publish(directory)
            save_json(root / 'audit.json', audit)
        refit = None
        if selected is not None and budget.remaining() > 1.:
            refit, directory = execute(selected, 'refit', root / 'search_split-01.npz',
                                      'final-refit', max(.1, budget.remaining() - .5))
            if refit['status'] == 'completed':
                publish(directory)
        audit_completed = any(r['status'] == 'completed' and
            candidate_key(r['candidate']) == candidate_key(best) for r in audit)
        summary.update(status='completed' if selected else 'fallback', selected=selected,
            selected_search_score=best_score, search_selected=best,
            has_disjoint_search_and_audit=nested, independent_search_and_audit=nested and audit_completed,
            audit=audit, final_refit=refit, validation=inner_description, audit_validation=outer_description,
            audit_status='external_evaluation_required' if deployment else 'completed' if audit_completed else 'skipped_or_failed',
            audit_used_for_selection=False,
            best_submission_seconds=best_seconds, experiments=len(records),
            first_model_seconds=first_model_seconds, search_window_count=len(windows),
            search_policy=search_policy,
            distinct_experiments=len(seen), llm_calls=len(api_records),
            llm_http_calls_reported=sum(s.get('attempts', 0) for s in api_records),
            llm_token_upper_estimate=llm_tokens, duplicate_proposals=duplicate_count,
            rejected_proposals=rejected_count,
            api_configured=api_configured,
            llm_status=('called' if api_records else 'configured_no_call' if api_configured else 'unconfigured'),
            score_limitation='Search scores are adaptive, not Boosters scores. Deployment selection uses all known periods; its independent assessment is the external masked benchmark. Optional nested mode audits frozen candidates and has historically exposed public folds.')
    except Exception as exc:
        summary.update(status='recovered_error', error=type(exc).__name__ + ': ' + str(exc))
    validate_submission(bundle, pd.read_csv(bundle.task_path.parent / 'submission.csv'))
    elapsed = time.monotonic() - budget.started
    executed = records + summary.get('audit', []) + ([summary['final_refit']] if summary.get('final_refit') else [])
    actual_fit_seconds = sum(float(r.get('fit_seconds', 0.)) for r in executed)
    summary.update(seconds=elapsed, budget_seconds=budget.seconds,
                   seconds_remaining=budget.remaining(), training_seconds=actual_fit_seconds,
                   training_fraction=actual_fit_seconds / max(elapsed, .001),
                   completed_model_fits=sum(int(r.get('model_fit_count', int(r.get('fit_seconds', 0.) > 0))) for r in executed),
                   cached_decision_experiments=sum(bool(r.get('training_reused')) for r in records))
    save_json(root / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'metric', 'seconds', 'submission_rows')}, ensure_ascii=False))
    return summary
