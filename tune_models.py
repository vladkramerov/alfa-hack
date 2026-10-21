"""Bounded Optuna search on an inner split, compared on the original outer split."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent


def split_indices(X, y, regression):
    if regression:
        outer_year = X['Year_Factor'].max()
        fit = X.index[X['Year_Factor'] < outer_year]
        outer = X.index[X['Year_Factor'] == outer_year]
        inner_year = X.loc[fit, 'Year_Factor'].max()
        inner_fit = X.index[X['Year_Factor'] < inner_year]
        inner_valid = X.index[X['Year_Factor'] == inner_year]
        outer_description = f'Year_Factor < {outer_year} train; == {outer_year} validation'
        inner_description = f'Year_Factor < {inner_year} train; == {inner_year} tuning'
    else:
        fit, outer = train_test_split(X.index, test_size=0.2, random_state=42, stratify=y)
        inner_fit, inner_valid = train_test_split(fit, test_size=0.25, random_state=42, stratify=y.loc[fit])
        outer_description = 'stratified 80/20; random_state=42'
        inner_description = 'stratified 75/25 within outer training; random_state=42'
    if not all(len(v) for v in [fit, outer, inner_fit, inner_valid]):
        raise ValueError('Empty partition')
    if set(inner_fit) & set(inner_valid) or (set(inner_fit) | set(inner_valid)) != set(fit):
        raise ValueError('Invalid inner partitions')
    if set(fit) & set(outer):
        raise ValueError('Outer validation leaked into training')
    return fit, outer, inner_fit, inner_valid, outer_description, inner_description


def preprocessing(X):
    categories = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numeric = [c for c in X if c not in categories]
    return ColumnTransformer([
        ('numeric', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore'))]), categories),
    ])


def base_parameters(regression):
    if regression:
        return {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0}
    return {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 2, 'max_features': 0.8}


def suggest(trial, regression):
    if regression:
        return {'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 12.0, log=True)}
    return {'n_estimators': trial.suggest_categorical('n_estimators', [100, 150, 250]),
            'max_depth': trial.suggest_categorical('max_depth', [12, 20, 28, None]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4, 8]),
            'max_features': trial.suggest_categorical('max_features', [0.5, 0.8, 1.0])}


def estimator(parameters, regression):
    if regression:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**parameters, loss_function='RMSE', thread_count=4,
                                 random_seed=42, verbose=False, allow_writing_files=False)
    return RandomForestClassifier(**parameters, random_state=42, n_jobs=4)


def score(y, pred, regression):
    return float(root_mean_squared_error(y, pred) if regression else f1_score(y, pred, average='macro'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['regression', 'multiclass'])
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=420, help='Search budget; running trial is allowed to finish')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data_for_participants')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout < 1:
        parser.error('trials and timeout must be positive')
    import optuna
    started = time.monotonic()
    regression = args.task == 'regression'
    folder = 'Регрессия' if regression else 'Многоклассовая классификация'
    target = 'site_eui' if regression else 'attack_cat'
    metric = 'rmse' if regression else 'macro_f1'
    data = args.data_dir.resolve() / folder
    out = (args.out or ROOT / 'runs' / f'{args.task}-optuna-{time.strftime("%Y%m%d-%H%M%S")}').resolve()
    out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    X, y = train.drop(columns=['id', target]), train[target]
    assert list(test.drop(columns='id')) == list(X)
    fit, outer, inner_fit, inner_valid, split, inner_split = split_indices(X, y, regression)
    for name, ids in [('fit', fit), ('validation', outer), ('inner_fit', inner_fit), ('inner_validation', inner_valid)]:
        train.loc[ids, ['id']].to_csv(out / f'{name}_ids.csv', index=False)
    prep = preprocessing(X)
    inner_X = prep.fit_transform(X.loc[inner_fit])
    inner_V = prep.transform(X.loc[inner_valid])
    incumbent_params = base_parameters(regression)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=4)
    study = optuna.create_study(direction='minimize' if regression else 'maximize', sampler=sampler,
                                study_name=args.task, storage=f'sqlite:///{out / "study.sqlite3"}')
    study.enqueue_trial(incumbent_params)
    def objective(trial):
        params = suggest(trial, regression)
        model = estimator(params, regression)
        before = time.monotonic()
        model.fit(inner_X, y.loc[inner_fit])
        pred = np.asarray(model.predict(inner_V)).reshape(-1)
        trial.set_user_attr('seconds', round(time.monotonic() - before, 2))
        return score(y.loc[inner_valid], pred, regression)
    def checkpoint(study, trial):
        study.trials_dataframe().to_csv(out / 'trials.csv', index=False)
    print(f'{args.task}: {args.trials} trials, inner split: {inner_split}. Outer split is not used by Optuna.', flush=True)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, n_jobs=1,
                   callbacks=[checkpoint], gc_after_trial=True)
    candidate_params = study.best_params
    # Freeze parameters before inspecting outer validation.
    (out / 'best_inner_parameters.json').write_text(json.dumps(candidate_params, indent=2) + '\n')
    print(f'Inner best: trial {study.best_trial.number}, score={study.best_value:.6f}. Checking outer validation...', flush=True)
    incumbent = Pipeline([('prepare', preprocessing(X)), ('model', estimator(incumbent_params, regression))])
    incumbent.fit(X.loc[fit], y.loc[fit])
    incumbent_pred = np.asarray(incumbent.predict(X.loc[outer])).reshape(-1)
    incumbent_score = score(y.loc[outer], incumbent_pred, regression)
    if candidate_params == incumbent_params:
        candidate, candidate_pred, candidate_score = incumbent, incumbent_pred, incumbent_score
    else:
        candidate = Pipeline([('prepare', preprocessing(X)), ('model', estimator(candidate_params, regression))])
        candidate.fit(X.loc[fit], y.loc[fit])
        candidate_pred = np.asarray(candidate.predict(X.loc[outer])).reshape(-1)
        candidate_score = score(y.loc[outer], candidate_pred, regression)
    improved = candidate_score < incumbent_score if regression else candidate_score > incumbent_score
    selected, selected_pred = (candidate, candidate_pred) if improved else (incumbent, incumbent_pred)
    selected_params = candidate_params if improved else incumbent_params
    selected_score = candidate_score if improved else incumbent_score
    pd.DataFrame({'id': train.loc[outer, 'id'], 'true': y.loc[outer],
                  'incumbent_prediction': incumbent_pred, 'candidate_prediction': candidate_pred}).to_csv(out / 'outer_predictions.csv', index=False)
    dummy = DummyRegressor(strategy='mean') if regression else DummyClassifier(strategy='most_frequent')
    dummy.fit(X.loc[fit], y.loc[fit])
    scores = {metric: selected_score, f'dummy_{metric}': score(y.loc[outer], dummy.predict(X.loc[outer]), regression)}
    if not regression:
        scores.update(accuracy=float(accuracy_score(y.loc[outer], selected_pred)),
                      weighted_f1=float(f1_score(y.loc[outer], selected_pred, average='weighted')))
        for label, predictions in [('classification_report', selected_pred), ('candidate_classification_report', candidate_pred)]:
            (out / f'{label}.txt').write_text(classification_report(y.loc[outer], predictions, zero_division=0), encoding='utf-8')
    print(f'Outer {metric}: incumbent={incumbent_score:.6f}, candidate={candidate_score:.6f}; selected={"candidate" if improved else "incumbent"}', flush=True)
    print('Refitting selected configuration on all train rows...', flush=True)
    final_model = clone(selected).fit(X, y)
    predictions = np.asarray(final_model.predict(test[X.columns])).reshape(-1)
    submission = pd.DataFrame({'id': test.id, target: predictions})
    assert submission.shape == sample.shape and submission.columns.tolist() == sample.columns.tolist()
    assert submission.id.equals(test.id) and submission.id.is_unique and submission.notna().all().all()
    if regression:
        assert np.isfinite(submission[target]).all()
    else:
        assert set(submission[target]) <= set(y)
    submission.to_csv(out / 'submission.csv', index=False)
    assert pd.read_csv(out / 'submission.csv').id.equals(test.id)
    metadata = {'task': args.task, 'model': type(selected.named_steps['model']).__name__,
                'model_family': 'optuna', 'model_parameters': selected.named_steps['model'].get_params(),
                'hypothesis': 'Inner-split hyperparameter search improves the incumbent on the original outer validation',
                'split': split, 'seed': 42, 'scores': scores, 'seconds': round(time.monotonic() - started, 2),
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'sklearn': sklearn.__version__, 'optuna': optuna.__version__,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input_sha256': {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                                 for name in ['train.csv', 'test.csv', 'sample_submission.csv']},
                'tuning': {'inner_split': inner_split, 'requested_trials': args.trials,
                           'completed_trials': sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
                           'timeout_seconds': args.timeout, 'timeout_note': 'applies to search, in-flight trial may finish; outer checks and final fit are extra',
                           'sampler': 'TPESampler(seed=42,n_startup_trials=4)', 'trial_jobs': 1,
                           'best_trial': study.best_trial.number, 'best_inner_score': study.best_value,
                           'candidate_parameters': candidate_params, 'incumbent_parameters': incumbent_params,
                           'outer_incumbent_score': incumbent_score, 'outer_candidate_score': candidate_score,
                           'selected': 'candidate' if improved else 'incumbent'}}
    if regression:
        import catboost
        metadata['catboost'] = catboost.__version__
    (out / 'metrics.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'OK: {out / "submission.csv"}', flush=True)


if __name__ == '__main__':
    main()
