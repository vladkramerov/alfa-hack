"""Result-driven AutoResearch with nested search, early output and bounded workers."""
from dataclasses import replace
import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from automl_data import (load_task, validation_split, split_key, better, fallback_submission,
                         atomic_submission, validate_submission, LOWER_IS_BETTER)
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
    return {'task': bundle.text, 'target': bundle.target, 'id': bundle.id_column,
        'type': bundle.task_type, 'metric': bundle.metric, 'validation': validation,
        'shape': list(X.shape), 'dtypes': X.dtypes.astype(str).to_dict(),
        'missing': X.isna().mean().to_dict(), 'nunique': X.nunique().to_dict(),
        'categories': {str(c): int(X[c].nunique()) for c in X.select_dtypes(exclude='number')},
        'numeric_summary': summary,
        'numeric_target_correlations': dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:8]),
        'target_fit_summary': (y.value_counts().to_dict() if bundle.task_type == 'classification'
                               else json.loads(y.describe(percentiles=[.01, .5, .99]).to_json())),
        'target_min': float(y.min()) if bundle.task_type == 'regression' else None,
        'target_skew': float(y.skew()) if bundle.task_type == 'regression' and len(y) > 2 and np.isfinite(y.skew()) else 0.,
        'classes': list(bundle.classes) if bundle.classes else None,
        'positive_class': bundle.positive_class,
        'validation_warning': 'Search split is reused adaptively; audit targets are withheld until proposals stop.'}


def offline_proposal(profile, observations, seen):
    """Fallback policies branch on measured errors and fit-only data, not dataset names."""
    completed = [r for r in observations if r.get('status') == 'completed' and 'score' in r]
    ordered = sorted(completed, key=lambda r: r['score'], reverse=profile['metric'] not in LOWER_IS_BETTER)
    best = ordered[0]['candidate'] if ordered else {'family': 'lightgbm'}
    proposals = []
    families = {r['candidate']['family'] for r in completed}
    classification = profile['type'] == 'classification'
    if not completed:
        proposals.append({'family': 'lightgbm', 'parameters': {'n_estimators': 120},
                          'hypothesis': 'Quick native-category boosting baseline before research.'})
    if classification:
        if 'xgboost' not in families:
            proposals.append({'family': 'xgboost', 'parameters': {'n_estimators': 600},
                              'hypothesis': 'Test histogram boosting against the measured baseline.'})
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
        if 'catboost' not in families:
            proposals.append({'family': 'catboost', 'parameters': {'iterations': 500},
                              'hypothesis': 'Compare symmetric native-category trees for measured residual structure.'})
        if 'xgboost' not in families:
            proposals.append({'family': 'xgboost', 'parameters': {'n_estimators': 600},
                              'hypothesis': 'Compare histogram boosting for residual bias and tail errors.'})
        if profile['target_min'] > -1 and profile['target_skew'] > 1:
            proposals.append({**best, 'target_transform': 'none',
                              'hypothesis': 'Check whether inverse log transformation introduces bias for the primary loss.'})
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
                'hypothesis': 'Test a conservative blend of the two measured distinct model families.'})
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


def main(task_path, started=None, threads=4):
    budget = Budget(os.getenv('AUTOML_TIME_LIMIT', '10200'), started)
    bundle = load_task(task_path)
    atomic_submission(bundle, fallback_submission(bundle))
    first_submission = time.monotonic() - budget.started
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = bundle.task_path.parent / 'agent_runs' / stamp
    root.mkdir(parents=True)
    records, api_records, proposal_feedback = [], [], []
    best, best_score, baseline = None, None, None
    best_seconds, fit_seconds, duplicate_count, rejected_count = 0., 0., 0, 0
    summary = {'status': 'fallback', 'metric': bundle.metric, 'first_submission_seconds': first_submission,
               'task_type': bundle.task_type, 'submission_rows': len(bundle.sample)}
    modules = Path(__file__).parent

    def execute(candidate, mode, split_path, label, timeout):
        directory = root / label
        directory.mkdir()
        request = {'task_path': str(bundle.task_path), 'candidate': candidate, 'mode': mode,
                   'split_path': str(split_path), 'output': str(directory), 'threads': threads}
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
        nested = True
        try:
            inner_bundle = replace(bundle, train=bundle.train.iloc[outer_fit].reset_index(drop=True))
            inner_fit, inner_valid, inner_description = validation_split(inner_bundle)
            fit, valid = outer_fit[inner_fit], outer_fit[inner_valid]
            if not len(fit) or not len(valid):
                raise ValueError('Empty inner split')
        except ValueError:
            nested = False
            fit, valid = outer_fit, outer_valid
            inner_description = outer_description + ':no-independent-audit'
        np.savez_compressed(root / 'search_split.npz', fit=fit, valid=valid)
        np.savez_compressed(root / 'audit_split.npz', fit=outer_fit, valid=outer_valid)
        profile = profile_data(bundle, fit, inner_description)
        save_json(root / 'profile.json', profile)
        save_json(root / 'validation.json', {'search': inner_description, 'audit': outer_description,
            'nested': nested, 'search_key': split_key(fit, valid), 'audit_key': split_key(outer_fit, outer_valid),
            'search_fit_rows': len(fit), 'search_valid_rows': len(valid), 'audit_rows': len(outer_valid),
            'data_hashes': bundle.hashes})
        seen = set()
        max_experiments = min(24, max(0, int(os.getenv('AUTOML_MAX_EXPERIMENTS', '12'))))
        api_configured = all(os.getenv(k) for k in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL'))
        llm_failures, llm_tokens = 0, 0
        for attempt in range(max_experiments * 3 + 1):
            if len(records) >= max_experiments or budget.remaining() <= budget.reserve(fit_seconds):
                break
            candidate, source = None, 'adaptive_fallback'
            if records and api_configured and len(api_records) < 8 and llm_failures < 3 and llm_tokens < 48000:
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
            available = budget.remaining() - budget.reserve(fit_seconds)
            result, directory = execute(candidate, 'evaluate', root / 'search_split.npz',
                f'experiment-{len(records):02d}', max(.1, min(1200., max(30., budget.seconds * .12), available)))
            record.update(result)
            fit_seconds = max(fit_seconds, result.get('seconds', 0.))
            if result['status'] == 'completed' and better(bundle.metric, result['score'], best_score):
                publish(directory)
                best, best_score = candidate, result['score']
                best_seconds = time.monotonic() - budget.started
                if baseline is None:
                    baseline = candidate
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
            refit, directory = execute(selected, 'refit', root / 'search_split.npz',
                                      'final-refit', max(.1, budget.remaining() - .5))
            if refit['status'] == 'completed':
                publish(directory)
        audit_completed = any(r['status'] == 'completed' and
            candidate_key(r['candidate']) == candidate_key(best) for r in audit)
        summary.update(status='completed' if selected else 'fallback', selected=selected,
            selected_search_score=best_score, search_selected=best,
            has_disjoint_search_and_audit=nested, independent_search_and_audit=nested and audit_completed,
            audit=audit, final_refit=refit, validation=inner_description, audit_validation=outer_description,
            audit_status='completed' if audit_completed else 'skipped_or_failed',
            audit_used_for_selection=False,
            best_submission_seconds=best_seconds, experiments=len(records),
            distinct_experiments=len(seen), llm_calls=sum(s.get('attempts', 0) for s in api_records),
            llm_token_upper_estimate=llm_tokens, duplicate_proposals=duplicate_count,
            rejected_proposals=rejected_count,
            api_configured=api_configured,
            llm_status=('called' if api_records else 'configured_no_call' if api_configured else 'unconfigured'),
            score_limitation='Search scores are adaptive. Audit evaluates frozen baseline and search-selected candidates without choosing between them. Public folds have historical exposure; neither score is a Boosters score.')
    except Exception as exc:
        summary.update(status='recovered_error', error=type(exc).__name__ + ': ' + str(exc))
    validate_submission(bundle, pd.read_csv(bundle.task_path.parent / 'submission.csv'))
    summary.update(seconds=time.monotonic() - budget.started, budget_seconds=budget.seconds,
                   seconds_remaining=budget.remaining(), training_seconds=sum(r.get('seconds', 0.) for r in records))
    save_json(root / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'metric', 'seconds', 'submission_rows')}, ensure_ascii=False))
    return summary
