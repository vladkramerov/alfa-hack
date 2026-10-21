"""Local learning baseline. Not the final LLM agent or platform entrypoint."""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['regression', 'multiclass'])
    parser.add_argument('--model', choices=['linear', 'forest', 'boosting'], default='linear')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data_for_participants', help='Folder containing both task directories')
    parser.add_argument('--out', type=Path, help='New output directory; existing directories are refused')
    args = parser.parse_args()
    started = time.monotonic()
    regression = args.task == 'regression'
    folder = 'Регрессия' if regression else 'Многоклассовая классификация'
    target = 'site_eui' if regression else 'attack_cat'
    data = args.data_dir.resolve() / folder
    out = args.out or ROOT / 'runs' / f'{args.task}-{args.model}-{time.strftime("%Y%m%d-%H%M%S")}'
    out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    X = train.drop(columns=['id', target])
    y = train[target]
    assert list(test.drop(columns='id')) == list(X)
    assert test['id'].is_unique and sample['id'].equals(test['id'])
    assert y.notna().all()
    print(f'{args.task}: train={train.shape}, test={test.shape}', flush=True)
    if regression:
        last_year = X['Year_Factor'].max()
        valid_idx = X.index[X['Year_Factor'] == last_year]
        fit_idx = X.index[X['Year_Factor'] < last_year]
        split = f'Year_Factor < {last_year} train; == {last_year} validation'
        estimator = Ridge(alpha=1.0, solver="lsqr", tol=1e-6)
        dummy = DummyRegressor(strategy='mean')
    else:
        fit_idx, valid_idx = train_test_split(X.index, test_size=0.2, random_state=42, stratify=y)
        split = 'stratified 80/20; random_state=42'
        estimator = LogisticRegression(solver='lbfgs', max_iter=3000, random_state=42)
        dummy = DummyClassifier(strategy='most_frequent')
    if args.model == 'forest':
        forest_type = RandomForestRegressor if regression else RandomForestClassifier
        estimator = forest_type(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                max_features=0.8, n_jobs=4, random_state=42)
    if args.model == 'boosting':
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            raise SystemExit('Install boosting dependencies: python -m pip install -r requirements-boosting.txt')
        boosting_type = CatBoostRegressor if regression else CatBoostClassifier
        estimator = boosting_type(iterations=500, depth=6, learning_rate=0.05,
                                  l2_leaf_reg=3.0, random_seed=42, thread_count=4,
                                  loss_function='RMSE' if regression else 'MultiClass',
                                  allow_writing_files=False, verbose=100)
    pd.DataFrame({'id': train.loc[fit_idx, 'id']}).to_csv(out / 'fit_ids.csv', index=False)
    pd.DataFrame({'id': train.loc[valid_idx, 'id']}).to_csv(out / 'validation_ids.csv', index=False)
    categorical = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numeric = [c for c in X if c not in categorical]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('fill', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    model = Pipeline([('prepare', preprocess), ('model', estimator)])
    dummy.fit(X.loc[fit_idx], y.loc[fit_idx])
    dummy_pred = dummy.predict(X.loc[valid_idx])
    print(f'Validation: {split}. Training {type(estimator).__name__}...', flush=True)
    # Fail visibly rather than report an unconverged classifier as a clean run.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(X.loc[fit_idx], y.loc[fit_idx])
    pred = np.asarray(model.predict(X.loc[valid_idx])).reshape(-1)
    if regression:
        scores = {'dummy_rmse': root_mean_squared_error(y.loc[valid_idx], dummy_pred),
                  'rmse': root_mean_squared_error(y.loc[valid_idx], pred)}
    else:
        scores = {'dummy_macro_f1': f1_score(y.loc[valid_idx], dummy_pred, average='macro'),
                  'macro_f1': f1_score(y.loc[valid_idx], pred, average='macro'),
                  'accuracy': accuracy_score(y.loc[valid_idx], pred),
                  'weighted_f1': f1_score(y.loc[valid_idx], pred, average='weighted')}
        (out / 'classification_report.txt').write_text(
            classification_report(y.loc[valid_idx], pred, zero_division=0), encoding='utf-8')
    print(json.dumps(scores, indent=2), flush=True)
    print('Refitting on all train rows and predicting test...', flush=True)
    final_model = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        final_model.fit(X, y)
    test_pred = np.asarray(final_model.predict(test[X.columns])).reshape(-1)
    submission = pd.DataFrame({'id': test['id'], target: test_pred})
    assert submission.columns.tolist() == sample.columns.tolist()
    assert len(submission) == len(test) and submission['id'].equals(test['id'])
    assert submission['id'].is_unique and submission.notna().all().all()
    if regression:
        assert np.isfinite(submission[target]).all()
    else:
        assert set(submission[target]) <= set(y)
    submission.to_csv(out / 'submission.csv', index=False)
    reopened = pd.read_csv(out / 'submission.csv')
    assert reopened.shape == sample.shape and reopened['id'].equals(test['id'])
    hashes = {f: hashlib.sha256((data / f).read_bytes()).hexdigest()
              for f in ['train.csv', 'test.csv', 'sample_submission.csv']}
    try:
        import subprocess
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    report = {'source_commit': source_commit, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'task': args.task, 'model': type(estimator).__name__, 'split': split,
              'seed': 42, 'features': list(X), 'categorical': categorical,
              'scores': {k: float(v) for k, v in scores.items()},
              'model_family': args.model,
              'model_parameters': estimator.get_params(),
              'hypothesis': {'linear': 'Linear model improves on a constant prediction',
                             'forest': 'Random Forest improves on the linear model using the same split',
                             'boosting': 'CatBoost improves on Random Forest using the same split and preprocessing'}[args.model],
              'seconds': round(time.monotonic() - started, 2),
              'python': platform.python_version(), 'sklearn': sklearn.__version__,
              'pandas': pd.__version__, 'numpy': np.__version__, 'input_sha256': hashes}
    if args.model == 'boosting':
        import catboost
        report['catboost'] = catboost.__version__
        report['preprocessing'] = 'median + StandardScaler; constant fill + OneHotEncoder(handle_unknown=ignore)'
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'OK: {out.resolve() / "submission.csv"}', flush=True)


if __name__ == '__main__':
    main()
