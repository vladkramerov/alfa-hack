"""Independent bounded tabular research loop with measured diagnostic feedback."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

import numpy as np
import pandas as pd

from research_memory import Memory, atomic_json, semantic_recipe, data_digest

BOUNDS = {
    'mixed_neighbors': {'n_neighbors': (1, 101)},
    'factorization_machine': {'rank': (1, 32), 'epochs': (1, 40), 'batch_size': (32, 2048),
                              'learning_rate': (1e-5, .1), 'l2': (0., 10.)},
    'lightgbm': {'n_estimators': (10, 2000), 'num_leaves': (4, 128), 'learning_rate': (.005, .3),
                 'reg_lambda': (0., 100.), 'min_child_samples': (5, 200)},
    'catboost': {'iterations': (10, 2000), 'depth': (2, 10), 'learning_rate': (.005, .3),
                'l2_leaf_reg': (.01, 100.)},
}
INTEGER_PARAMS = {'n_neighbors', 'rank', 'epochs', 'batch_size', 'n_estimators', 'num_leaves',
                  'min_child_samples', 'iterations', 'depth'}
DEFAULTS = {
    'mixed_neighbors': {'n_neighbors': 25},
    'factorization_machine': {'rank': 8, 'epochs': 20, 'batch_size': 512, 'learning_rate': .005, 'l2': .001},
    'lightgbm': {'n_estimators': 300, 'num_leaves': 31, 'learning_rate': .05, 'reg_lambda': 1., 'min_child_samples': 20},
    'catboost': {'iterations': 300, 'depth': 6, 'learning_rate': .05, 'l2_leaf_reg': 3.},
}


def validate_recipe(recipe, kind):
    family = recipe.get('family')
    if family not in BOUNDS:
        raise ValueError('Unsupported learner family')
    if (family == 'mixed_neighbors' and kind != 'classification') or (family == 'factorization_machine' and kind != 'regression'):
        raise ValueError('Learner family does not match task type')
    params = recipe.get('parameters', {})
    if not isinstance(params, dict):
        raise ValueError('parameters must be an object')
    for name, value in params.items():
        if name not in BOUNDS[family] or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Unsupported parameter: ' + name)
        if name in INTEGER_PARAMS and not isinstance(value, int):
            raise ValueError('Expected integer parameter: ' + name)
        low, high = BOUNDS[family][name]
        if not low <= value <= high:
            raise ValueError('Parameter exceeds resource bounds: ' + name)
    return {'family': family, 'parameters': {**DEFAULTS[family], **params}, 'hypothesis': str(recipe.get('hypothesis', ''))[:2000]}


def parse_action(payload):
    try:
        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        first = choices[0]
        if first.get('finish_reason') in {'length', 'content_filter'}:
            return None
        message = first.get('message') or {}
        calls = message.get('tool_calls') or []
        if calls:
            function = calls[0]['function']
            if function['name'] != 'run_experiment':
                return None
            action = json.loads(function['arguments'])
        else:
            action = json.loads(message.get('content', ''))
        if isinstance(action, dict) and (action.get('family') or action.get('stop') is True):
            return action
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        pass
    return None


def propose(profile, observations, timeout=30):
    deadline = time.monotonic() + max(0., timeout)
    base, key, model = (os.getenv(k) for k in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL'))
    if not all((base, key, model)):
        return None, {'status': 'unconfigured'}
    context = {'task': profile, 'experiments': observations[-10:]}
    body = {'model': model, 'temperature': .2, 'max_tokens': 1600, 'messages': [
        {'role': 'system', 'content': 'You direct a tabular ML researcher. Use INNER errors and prior failed hypotheses to propose ONE genuinely new mechanism, not a seed or small parameter rename. Explain a falsifiable cause before its intervention. Preprocessing is learner-appropriate; do not force Gaussian data or delete rare/extreme observations. Never tune on a final confirmation set. Allowed learners: mixed_neighbors (classification), factorization_machine (regression), lightgbm, catboost. Return JSON with family, parameters, hypothesis, or {"stop":true}. Resource bounds: ' + json.dumps(BOUNDS)},
        {'role': 'user', 'content': json.dumps(context, ensure_ascii=False, default=str)}]}
    # Small numeric/categorical profiles and bounded confusion/residual summaries only.
    encoded = json.dumps(body).encode()
    if len(encoded) > 70000:
        return None, {'status': 'context_limit'}
    url = base.rstrip('/')
    if not url.endswith('/chat/completions'):
        url += '/chat/completions'
    errors = []
    for _ in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            request = urllib.request.Request(url, data=encoded, headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=remaining) as response:
                payload = json.loads(response.read(2_000_000))
            action = parse_action(payload)
            if action is not None:
                return action, {'status': 'ok', 'usage': payload.get('usage', {})}
            errors.append('empty_truncated_or_invalid_response')
        except Exception as exc:
            errors.append(type(exc).__name__)
    return None, {'status': 'failed', 'errors': errors}


class TreeModel:
    """Learner-specific missing/category handling; no Gaussian transform or row removal."""
    def __init__(self, family, kind, parameters):
        self.family, self.kind, self.parameters = family, kind, parameters

    def _transform(self, X):
        result = X[self.columns_].copy()
        for column in self.categories_:
            text = result[column].astype('string').fillna('__MISSING__')
            result[column] = text.astype(object) if self.family == 'catboost' else pd.Categorical(text, categories=self.categories_[column])
        for column in result.columns.difference(list(self.categories_)):
            result[column] = pd.to_numeric(result[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
            if self.family == 'catboost':
                result[column] = result[column].fillna(self.medians_[column])
        return result

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        self.categories_ = {c: list(pd.unique(X[c].astype('string').fillna('__MISSING__'))) for c in X if not pd.api.types.is_numeric_dtype(X[c])}
        self.medians_ = {c: float(v) if pd.notna(v) else 0. for c, v in X.select_dtypes(include='number').replace([np.inf, -np.inf], np.nan).median().items()}
        target = y
        if self.kind == 'classification':
            self.classes_ = np.asarray(sorted(pd.unique(pd.Series(y).astype(str))), dtype=str)
            mapping = {c: i for i, c in enumerate(self.classes_)}
            target = pd.Series(y).astype(str).map(mapping).to_numpy()
        if self.family == 'lightgbm':
            from lightgbm import LGBMClassifier, LGBMRegressor
            constructor = LGBMClassifier if self.kind == 'classification' else LGBMRegressor
            self.model_ = constructor(n_jobs=int(os.getenv('RESEARCH_THREADS', '4')), verbosity=-1, random_state=42, **self.parameters)
            self.model_.fit(self._transform(X), target)
        else:
            from catboost import CatBoostClassifier, CatBoostRegressor
            constructor = CatBoostClassifier if self.kind == 'classification' else CatBoostRegressor
            self.model_ = constructor(thread_count=int(os.getenv('RESEARCH_THREADS', '4')), verbose=False, random_seed=42,
                                      allow_writing_files=False, **self.parameters)
            self.model_.fit(self._transform(X), target, cat_features=list(self.categories_))
        self.audit_ = {'fit_rows': len(X), 'removed_rows': 0, 'categorical_columns': list(self.categories_), 'gaussian_transform': False}
        return self

    def predict(self, X):
        result = self.model_.predict(self._transform(X)).reshape(-1)
        return self.classes_[result.astype(int)] if self.kind == 'classification' else result

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))


def make_model(kind, recipe):
    recipe = validate_recipe(recipe, kind)
    family, params = recipe['family'], recipe['parameters']
    if family == 'mixed_neighbors':
        from research_neighbors import MixedNeighborsClassifier
        return MixedNeighborsClassifier(**params)
    if family == 'factorization_machine':
        from research_fm import FactorizationRegressor
        return FactorizationRegressor(**params)
    return TreeModel(family, kind, params)


def prediction_for(task, model, X):
    if task.metric in {'roc_auc', 'log_loss'}:
        order = np.argsort(np.asarray(model.classes_, dtype=str))
        probabilities = model.predict_proba(X)[:, order]
        return probabilities[:, 1] if task.metric == 'roc_auc' and probabilities.shape[1] == 2 else probabilities
    prediction = model.predict(X)
    return np.maximum(prediction, 0) if task.metric == 'rmsle' else prediction


def execute_worker(request):
    from research_data import load_task, score, diagnostics, save_submission
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    history, fingerprint = None, None
    try:
        task = load_task(request['task_path'])
        recipe = validate_recipe(request['recipe'], task.task_type)
        model = make_model(task.task_type, recipe)
        if request['mode'] != 'refit':
            with np.load(request['split_path'], allow_pickle=False) as split:
                fit, valid = split['fit'], split['valid']
            if len(set(fit) & set(valid)):
                raise ValueError('Fit/validation positions overlap')
        else:
            fit = np.arange(len(task.y))
        history = Memory(request['history'], request.get('global_ledger'))
        fingerprint = history.claim(recipe, task.X.iloc[fit], task.y.iloc[fit], request['mode'])
        model.fit(task.X.iloc[fit], task.y.iloc[fit])
        audit = getattr(model, 'audit_', {})
        atomic_json(output / 'audit.json', audit)
        if request['mode'] == 'refit':
            prediction = prediction_for(task, model, task.X_test)
            if task.metric == 'roc_auc' and len(task.probability_columns) > 1 and prediction.ndim == 1:
                prediction = np.column_stack((1. - prediction, prediction))
            if task.metric in {'roc_auc', 'log_loss'} and prediction.ndim == 2:
                outputs = [c for c in task.sample if c != task.id_column]
                order = {str(c): i for i, c in enumerate(sorted(np.asarray(model.classes_, dtype=str)))}
                if len(outputs) > 1:
                    prediction = prediction[:, [order[c] for c in outputs]]
                elif prediction.shape[1] == 2:
                    prediction = prediction[:, 1]
            save_submission(task, prediction, output / 'candidate_submission.csv')
            result = {'status': 'completed', 'mode': 'refit', 'train_rows': len(fit),
                      'submission_path': str((output / 'candidate_submission.csv').resolve()),
                      'test_X_sha256': data_digest(task.X_test), 'sample_sha256': data_digest(task.sample)}
        else:
            probabilities, classes = None, None
            if task.task_type == 'classification':
                raw_probabilities = model.predict_proba(task.X.iloc[valid])
                raw_classes = np.asarray(model.classes_, dtype=str)
                hard_prediction = raw_classes[raw_probabilities.argmax(axis=1)]
                order = np.argsort(raw_classes)
                classes, probabilities = raw_classes[order], raw_probabilities[:, order]
                if task.metric in {'roc_auc', 'log_loss'}:
                    prediction = probabilities[:, 1] if task.metric == 'roc_auc' and len(classes) == 2 else probabilities
                else:
                    prediction = hard_prediction
            else:
                prediction = prediction_for(task, model, task.X.iloc[valid])
            saved = {'positions': valid, 'y': task.y.iloc[valid].to_numpy(), 'prediction': prediction}
            if task.task_type == 'classification':
                saved['y'] = saved['y'].astype(str)
                if prediction.ndim == 1 and task.metric not in {'roc_auc'}:
                    saved['prediction'] = prediction.astype(str)
                saved['classes'] = classes
                saved['probabilities'] = probabilities
            np.savez_compressed(output / 'validation_predictions.npz', **saved)
            result = {'status': 'completed', 'mode': request['mode'], 'metric': task.metric,
                      'score': score(task.metric, task.y.iloc[valid], prediction), 'fit_rows': len(fit), 'valid_rows': len(valid),
                      'validation_X_sha256': data_digest(task.X.iloc[valid]),
                      'validation_y_sha256': data_digest(task.y.iloc[valid].to_numpy().reshape(-1, 1))}
            # Probability objectives get hard labels only for descriptive confusion.
            diagnostic_prediction = hard_prediction if task.task_type == 'classification' else prediction
            result['diagnostics'] = diagnostics(task.task_type, task.y.iloc[valid], diagnostic_prediction,
                                                 classes=getattr(model, 'classes_', None))
        result.update(seconds=time.monotonic() - started, fingerprint=fingerprint)
        history.finish(fingerprint, result)
    except Exception as exc:
        result = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc), 'seconds': time.monotonic() - started}
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
        if fingerprint is not None:
            history.finish(fingerprint, result)
    atomic_json(output / 'metrics.json', result)
    return 0 if result['status'] == 'completed' else 1


def launch_worker(request, timeout):
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'request.json', request)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', RESEARCH_THREADS='4')
    # Training workers never require or inherit LLM credentials.
    for key in list(environment):
        if 'API_KEY' in key:
            environment.pop(key, None)
    started = time.monotonic()
    try:
        with (output / 'worker.log').open('w', encoding='utf8') as stream:
            if timeout <= 0:
                raise subprocess.TimeoutExpired('research worker', timeout)
            source = Path(request.get('worker_source', __file__)).resolve()
            child = subprocess.run([sys.executable, '-X', 'utf8', str(source), '--worker', str(output / 'request.json')],
                                   env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        result = json.loads((output / 'metrics.json').read_text(encoding='utf8'))
        if child.returncode and result['status'] == 'completed':
            raise RuntimeError('Worker status and exit disagree')
    except Exception as exc:
        result = {'status': 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'failed',
                  'error': type(exc).__name__, 'seconds': time.monotonic() - started}
        atomic_json(output / 'metrics.json', result)
        memory = Memory(request['history'], request.get('global_ledger'))
        for row in memory.records():
            if row['status'] == 'started':
                memory.finish(row['fingerprint'], result)
    return result


def _restore_submission(task, source, destination=None):
    """Read lexical identifiers, validate all values, then atomically promote."""
    from research_data import save_submission
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    numeric = task.probability_columns or ([task.target_column] if task.task_type == 'regression' else [])
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    return save_submission(task, frame, destination)


def _matching_history(rows, recipe, X, y, mode, **extra):
    identity = semantic_recipe(recipe)
    x_digest = data_digest(X)
    y_digest = data_digest(np.asarray(y).reshape(-1, 1))
    return [row for row in rows if row.get('status') == 'completed'
            and row.get('mode', row.get('stage')) == mode
            and row.get('recipe') == identity and row.get('X_sha256') == x_digest
            and row.get('y_sha256') == y_digest
            and all(row.get(key) == value for key, value in extra.items())]


def main(argv=None):
    from research_data import load_task, split_data, save_submission, better
    parser = argparse.ArgumentParser()
    parser.add_argument('task_path')
    parser.add_argument('--time-budget', type=float, default=10200)
    parser.add_argument('--max-experiments', type=int, default=6)
    parser.add_argument('--hypothesis-budget', type=float, default=900)
    args = parser.parse_args(argv)
    deadline = time.monotonic() + max(0., args.time_budget)
    task = load_task(args.task_path)
    root = task.task_path.parent / 'research_runs'
    root.mkdir(exist_ok=True)
    submission = task.task_path.parent / 'submission.csv'
    try:
        _restore_submission(task, submission)
    except (OSError, ValueError, KeyError):
        if task.task_type == 'classification':
            modes = task.y.astype(str).value_counts()
            fallback = np.repeat(str(modes.index[0]), len(task.X_test))
            if task.metric in {'roc_auc', 'log_loss'}:
                labels = sorted(task.y.astype(str).unique())
                probs = task.y.astype(str).value_counts(normalize=True).reindex(labels).to_numpy()
                outputs = task.probability_columns or [c for c in task.sample if c != task.id_column]
                if len(outputs) == 1 and len(labels) == 2:
                    fallback = np.repeat(probs[1], len(task.X_test))
                else:
                    mapping = dict(zip(labels, probs))
                    fallback = np.tile([mapping[column] for column in outputs], (len(task.X_test), 1))
        else:
            fallback = np.repeat(float(task.y.median()), len(task.X_test))
        save_submission(task, fallback)
    fit, valid, description = split_data(task)
    np.savez_compressed(root / 'inner-split.npz', fit=fit, valid=valid)
    profile = {'task': task.task_path.read_text(encoding='utf8')[:8000], 'task_type': task.task_type,
               'metric': task.metric, 'rows': len(fit), 'columns': list(task.X.columns),
               'missing': task.X.iloc[fit].isna().mean().round(4).to_dict(), 'validation': description}
    candidates = [{'family': 'mixed_neighbors' if task.task_type == 'classification' else 'factorization_machine'},
                  {'family': 'lightgbm'}, {'family': 'catboost'}]
    records, seen = [], set()
    ledger = os.getenv('RESEARCH_LEDGER_PATH')
    local_ledger = Path(__file__).resolve().parent / 'HYPOTHESES.md'
    if not ledger and local_ledger.exists():
        ledger = str(local_ledger)
    memory = Memory(root / 'hypotheses.json', ledger)
    historical = memory.records()
    observations = [{k: r[k] for k in ('recipe', 'status', 'score', 'diagnostics', 'error') if k in r} for r in historical]
    best = None
    reserve = min(600., max(10., args.time_budget * .25))
    validation_identity = {'metric': task.metric, 'validation_X_sha256': data_digest(task.X.iloc[valid]),
                           'validation_y_sha256': data_digest(task.y.iloc[valid].to_numpy().reshape(-1, 1))}
    for index in range(max(0, args.max_experiments)):
        remaining = deadline - time.monotonic() - reserve
        if remaining <= 0:
            break
        if index == 0:
            recipe = candidates[0]
        else:
            recipe, api = propose(profile, observations, timeout=min(30., remaining))
            atomic_json(root / f'api-{index:02d}.json', api)
            if recipe and recipe.get('stop') is True:
                break
            if not recipe:
                recipe = next((candidate for candidate in candidates if json.dumps(semantic_recipe(validate_recipe(candidate, task.task_type)), sort_keys=True) not in seen), None)
                if recipe is None:
                    break
        # API/retry overhead consumes the same absolute budget as workers.
        remaining = deadline - time.monotonic() - reserve
        if remaining <= 0:
            break
        try:
            recipe = validate_recipe(recipe, task.task_type)
            identity = json.dumps(semantic_recipe(recipe), sort_keys=True)
            if identity in seen:
                observations.append({'recipe': recipe, 'status': 'rejected', 'error': 'repeated mechanism'})
                continue
            seen.add(identity)
            reused = _matching_history(historical, recipe, task.X.iloc[fit], task.y.iloc[fit], 'evaluate', **validation_identity)
            if reused:
                result = {**reused[-1], 'reused': True}
            else:
                directory = root / f'run-{len(list(root.glob("run-*"))) + 1:03d}'
                request = {'task_path': str(task.task_path), 'recipe': recipe, 'mode': 'evaluate',
                           'output': str(directory), 'split_path': str(root / 'inner-split.npz'),
                           'history': str(memory.path), 'global_ledger': ledger}
                remaining = deadline - time.monotonic() - reserve
                if remaining <= 0:
                    break
                result = launch_worker(request, min(args.hypothesis_budget, remaining))
            record = {**result, 'recipe': recipe}
            records.append(record)
            observations.append({k: record[k] for k in ('recipe', 'status', 'score', 'diagnostics', 'error') if k in record})
            if result['status'] == 'completed' and (best is None or better(task.metric, result['score'], best['score'])):
                best = record
        except Exception as exc:
            records.append({'recipe': recipe, 'status': 'rejected', 'error': type(exc).__name__ + ': ' + str(exc)})
        atomic_json(root / 'research.json', records)
    if best is not None and deadline - time.monotonic() > 0:
        completed_refits = _matching_history(historical, best['recipe'], task.X, task.y, 'refit',
                            test_X_sha256=data_digest(task.X_test), sample_sha256=data_digest(task.sample))
        if completed_refits:
            result = {'status': 'failed', 'error': 'Completed refit has no compatible readable submission artifact; repeat fit refused'}
            for previous in reversed(completed_refits):
                try:
                    _restore_submission(task, previous['submission_path'])
                    result = {**previous, 'reused': True}
                    break
                except (OSError, ValueError, KeyError):
                    continue
        else:
            request = {'task_path': str(task.task_path), 'recipe': best['recipe'], 'mode': 'refit',
                       'output': str(root / ('final-' + str(time.time_ns()))), 'history': str(memory.path), 'global_ledger': ledger}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 0
            result = launch_worker(request, remaining)
            if result['status'] == 'completed':
                try:
                    _restore_submission(task, Path(request['output']) / 'candidate_submission.csv')
                except (OSError, ValueError, KeyError) as exc:
                    result = {'status': 'failed', 'error': 'Invalid refit artifact: ' + str(exc)}
        atomic_json(root / 'final.json', {'selected': best['recipe'], **result})
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        sys.exit(execute_worker(json.loads(Path(sys.argv[2]).read_text(encoding='utf8'))))
    sys.exit(main())
