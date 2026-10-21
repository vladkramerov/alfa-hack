"""Sequential controlled hypotheses. No LLM, no API, no changes to source data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tune_models import preprocessing

ROOT = Path(__file__).resolve().parent
FLAGS = ['year_built', 'energy_star_rating', 'direction_max_wind_speed',
         'direction_peak_wind_speed', 'max_wind_speed', 'days_with_fog']


def branch_guard():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'sergey_branch':
        raise RuntimeError('Wrong branch. Stopping without switching branches.')


def engineer(X, variant):
    result = X.copy()
    if variant == 'missing_flags':
        for c in FLAGS:
            result[c + '_missing'] = X[c].isna().astype('int8')
    elif variant == 'zero_year':
        result['year_built'] = result['year_built'].mask(result['year_built'] == 0)
    elif variant == 'flow_ratios':
        result['byte_ratio'] = (result.sbytes + 1) / (result.dbytes + 1)
        result['packet_ratio'] = (result.spkts + 1) / (result.dpkts + 1)
        result['source_bytes_per_packet'] = result.sbytes / (result.spkts + 1)
        result['destination_bytes_per_packet'] = result.dbytes / (result.dpkts + 1)
    return result


class NativePreprocessor(TransformerMixin, BaseEstimator):
    """Keep numeric imputation/scaling identical; preserve categorical strings."""
    def fit(self, X, y=None):
        self.categories_ = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        self.numeric_ = [c for c in X if c not in self.categories_]
        self.numeric_pipeline_ = Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
        self.numeric_pipeline_.fit(X[self.numeric_])
        return self

    def transform(self, X):
        result = pd.DataFrame(self.numeric_pipeline_.transform(X[self.numeric_]),
                              index=X.index, columns=self.numeric_)
        for c in self.categories_:
            result[c] = X[c].astype('object').where(X[c].notna(), 'missing').astype(str)
        return result


def build(recipe, X, regression):
    categories = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    if recipe['family'] == 'forest':
        kind = RandomForestRegressor if regression else RandomForestClassifier
        options = dict(n_estimators=150, max_depth=20, min_samples_leaf=2,
                       max_features=0.8, n_jobs=4, random_state=42)
        if not regression:
            options['class_weight'] = recipe.get('class_weight')
        model = kind(**options)
    else:
        kind = CatBoostRegressor if regression else CatBoostClassifier
        options = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                       random_seed=42, thread_count=4, loss_function='RMSE' if regression else 'MultiClass',
                       allow_writing_files=False, verbose=False)
        if recipe.get('variant') == 'native':
            options['cat_features'] = categories
        model = kind(**options)
    prep = NativePreprocessor() if recipe.get('variant') == 'native' else preprocessing(X)
    return Pipeline([('prepare', prep), ('model', model)])


def split(X, y, regression, secondary=False):
    if regression:
        year = X.Year_Factor.max() - (1 if secondary else 0)
        fit, valid = X.index[X.Year_Factor < year], X.index[X.Year_Factor == year]
        description = f'Year_Factor < {year} train; == {year} validation'
    else:
        seed = 2026 if secondary else 42
        fit, valid = train_test_split(X.index, test_size=0.2, random_state=seed, stratify=y)
        description = f'stratified 80/20; random_state={seed}'
    assert not set(fit) & set(valid)
    return fit, valid, description


def metrics(y, pred, regression):
    if regression:
        return {'rmse': float(root_mean_squared_error(y, pred))}
    return {'macro_f1': float(f1_score(y, pred, average='macro')),
            'accuracy': float(accuracy_score(y, pred)),
            'weighted_f1': float(f1_score(y, pred, average='weighted'))}


def align_probabilities(probabilities, source_classes, classes):
    source_classes = list(map(str, source_classes))
    assert set(source_classes) == set(classes)
    result = probabilities[:, [source_classes.index(c) for c in classes]]
    assert np.isfinite(result).all() and np.allclose(result.sum(axis=1), 1)
    return result


def fit_predictions(recipe, X, y, test, fit, valid, regression, full):
    changed = engineer(X, recipe.get('variant', 'plain'))
    changed_test = engineer(test, recipe.get('variant', 'plain'))
    model = build(recipe, changed, regression)
    model.fit(changed.loc[fit], y.loc[fit])
    classes = sorted(map(str, y.unique())) if not regression else []
    prediction = np.asarray(model.predict(changed.loc[valid])).reshape(-1)
    probabilities = None if regression else align_probabilities(
        model.predict_proba(changed.loc[valid]), model.named_steps['model'].classes_, classes)
    test_prediction, test_probabilities = None, None
    if full:
        final = clone(model).fit(changed, y)
        test_prediction = np.asarray(final.predict(changed_test)).reshape(-1)
        if not regression:
            test_probabilities = align_probabilities(final.predict_proba(changed_test), final.named_steps['model'].classes_, classes)
    return {'prediction': prediction, 'probabilities': probabilities,
            'test_prediction': test_prediction, 'test_probabilities': test_probabilities,
            'classes': classes, 'model_parameters': model.named_steps['model'].get_params(),
            'feature_columns': list(changed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data_for_participants')
    args = parser.parse_args()
    branch_guard()
    suite = args.out.resolve()
    suite.mkdir(parents=True, exist_ok=False)
    report_dir = ROOT / 'reports' / suite.name
    report_dir.mkdir(parents=True, exist_ok=False)
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data = {}
    for task, folder, target in [('regression', 'Регрессия', 'site_eui'), ('multiclass', 'Многоклассовая классификация', 'attack_cat')]:
        path = args.data_dir.resolve() / folder
        train, test, sample = [pd.read_csv(path / f) for f in ['train.csv', 'test.csv', 'sample_submission.csv']]
        X, y = train.drop(columns=['id', target]), train[target]
        data[task] = dict(X=X, y=y, test=test.drop(columns='id'), ids=train.id,
                          test_ids=test.id, sample=sample, target=target,
                          hashes={f: hashlib.sha256((path/f).read_bytes()).hexdigest()
                                  for f in ['train.csv', 'test.csv', 'sample_submission.csv']})
    records, cached = {}, {}

    def run(run_id, task, recipe, hypothesis, change, reference=None, secondary=False, members=None):
        branch_guard()
        started = time.monotonic()
        regression = task == 'regression'
        d = data[task]
        fit, valid, split_description = split(d['X'], d['y'], regression, secondary)
        destination = suite / run_id
        destination.mkdir()
        full = not secondary
        print(f'START {run_id}: {change}; {split_description}', flush=True)
        if members is None:
            result = fit_predictions(recipe, d['X'], d['y'], d['test'], fit, valid, regression, full)
        else:
            if secondary:
                components = [fit_predictions(records[m]['recipe'], d['X'], d['y'], d['test'], fit, valid, regression, False) for m in members]
            else:
                components = [cached[m] for m in members]
            if regression:
                result = dict(prediction=np.mean([r['prediction'] for r in components], axis=0),
                              test_prediction=np.mean([r['test_prediction'] for r in components], axis=0) if full else None)
            else:
                classes = components[0]['classes']
                assert all(r['classes'] == classes for r in components)
                probabilities = np.mean([r['probabilities'] for r in components], axis=0)
                test_probabilities = np.mean([r['test_probabilities'] for r in components], axis=0) if full else None
                result = dict(prediction=np.asarray(classes)[probabilities.argmax(axis=1)],
                              test_prediction=np.asarray(classes)[test_probabilities.argmax(axis=1)] if full else None,
                              probabilities=probabilities, test_probabilities=test_probabilities, classes=classes)
            result.update(model_parameters={'equal_weights': [0.5, 0.5], 'members': members}, feature_columns=list(d['X']))
        score = metrics(d['y'].loc[valid], result['prediction'], regression)
        for label, ids in [('fit', fit), ('validation', valid)]:
            pd.DataFrame({'id': d['ids'].loc[ids]}).to_csv(destination / f'{label}_ids.csv', index=False)
        pd.DataFrame({'id': d['ids'].loc[valid], 'true': d['y'].loc[valid], 'prediction': result['prediction']}).to_csv(destination / 'validation_predictions.csv', index=False)
        if not regression:
            text = classification_report(d['y'].loc[valid], result['prediction'], zero_division=0)
            (destination / 'classification_report.txt').write_text(text, encoding='utf-8')
            (report_dir / f'{run_id}-classes.txt').write_text(text, encoding='utf-8')
        if full:
            submission = pd.DataFrame({'id': d['test_ids'], d['target']: result['test_prediction']})
            assert submission.shape == d['sample'].shape and list(submission) == list(d['sample'])
            assert submission.id.equals(d['test_ids']) and submission.id.is_unique and submission.notna().all().all()
            if regression:
                assert np.isfinite(submission[d['target']]).all()
            else:
                assert set(submission[d['target']]) <= set(d['y'])
            submission.to_csv(destination / 'submission.csv', index=False)
            assert pd.read_csv(destination / 'submission.csv').id.equals(d['test_ids'])
        if not regression and full:
            np.savez_compressed(destination / 'probabilities.npz', classes=np.asarray(result['classes']),
                                validation=result['probabilities'], test=result['test_probabilities'])
        elapsed = time.monotonic() - started
        comparison = None
        if reference:
            ref = records[reference]
            assert ref['split'] == split_description and ref['input_sha256'] == d['hashes']
            m = 'rmse' if regression else 'macro_f1'
            gain = ref['scores'][m] - score[m] if regression else score[m] - ref['scores'][m]
            comparison = {'reference': reference, 'reference_value': ref['scores'][m], 'gain': gain,
                          'conclusion': 'better' if gain > 1e-10 else ('worse' if gain < -1e-10 else 'same')}
        record = dict(kind='hypothesis_experiment', run_id=run_id, suite=suite.name, task=task,
                      hypothesis=hypothesis, change=change, recipe=recipe,
                      model='EqualWeightEnsemble' if members else ('CatBoostRegressor' if regression else 'CatBoostClassifier') if recipe['family']=='catboost' else ('RandomForestRegressor' if regression else 'RandomForestClassifier'),
                      model_parameters=result['model_parameters'], feature_columns=result['feature_columns'],
                      split=split_description, seed=2026 if secondary and not regression else 42,
                      validation_kind='secondary_sensitivity' if secondary else 'primary',
                      n_fit=len(fit), n_validation=len(valid), scores=score, comparison=comparison,
                      seconds=round(elapsed,2), submission_created=full,
                      source_commit=source_commit, script_sha256=script_sha, input_sha256=d['hashes'],
                      python=platform.python_version(), sklearn=sklearn.__version__, catboost=catboost.__version__,
                      created_at=datetime.now().astimezone().isoformat(), artifact_path=f'runs/{suite.name}/{run_id}')
        if members and full:
            record['component_seconds'] = sum(records[m]['seconds'] for m in members)
            record['pipeline_seconds'] = round(elapsed+record['component_seconds'],2)
        (destination/'metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (report_dir/f'{run_id}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        cached[run_id], records[run_id] = result, record
        print(f'DONE {run_id}: {score}; comparison={comparison}; seconds={elapsed:.2f}',flush=True)
        return record

    forest = {'family':'forest','variant':'plain'}
    cb = {'family':'catboost','variant':'plain'}
    run('C00-forest-control','multiclass',forest,'control','Повтор прежнего Random Forest; контроль воспроизводимости')
    run('H1-balanced-forest','multiclass',dict(forest,class_weight='balanced'),'H1','Только class_weight=balanced','C00-forest-control')
    run('R00-catboost-control','regression',cb,'control','Повтор CatBoost с OneHotEncoder; контроль')
    run('H2-native-regression','regression',dict(cb,variant='native'),'H2','Нативные категории CatBoost вместо OneHotEncoder','R00-catboost-control')
    run('C00-catboost-control','multiclass',cb,'control','Повтор CatBoost с OneHotEncoder; контроль')
    run('H2-native-multiclass','multiclass',dict(cb,variant='native'),'H2','Нативные категории CatBoost вместо OneHotEncoder','C00-catboost-control')
    run('H3-missing-flags','regression',dict(cb,variant='missing_flags'),'H3','Шесть индикаторов пропусков; прочее без изменений','R00-catboost-control')
    run('H4-zero-year','regression',dict(cb,variant='zero_year'),'H4','Только year_built=0 заменён на пропуск','R00-catboost-control')
    run('H5-flow-ratios','multiclass',dict(forest,variant='flow_ratios'),'H5','Четыре отношения объёмов/пакетов, знаменатель +1','C00-forest-control')
    run('R00-forest-control','regression',forest,'control','Повтор прежнего Random Forest для ансамбля')
    ensemble={'family':'ensemble','variant':'plain'}
    run('H6-ensemble-regression','regression',ensemble,'H6','Среднее прогнозов исходных леса и CatBoost, веса 50/50','R00-catboost-control',members=['R00-forest-control','R00-catboost-control'])
    run('H6-ensemble-multiclass','multiclass',ensemble,'H6','Среднее вероятностей исходных леса и CatBoost, веса 50/50','C00-forest-control',members=['C00-forest-control','C00-catboost-control'])
    # Freeze strongest new alternative for each task before evaluating secondary splits.
    selected={}
    for task in ['regression','multiclass']:
        candidates=[r for r in records.values() if r['task']==task and r['hypothesis']!='control']
        metric='rmse' if task=='regression' else 'macro_f1'
        selected[task]=(min if task=='regression' else max)(candidates,key=lambda r:r['scores'][metric])['run_id']
    (suite/'frozen_secondary_candidates.json').write_text(json.dumps(selected,indent=2)+'\n')
    for task,control in [('regression','R00-catboost-control'),('multiclass','C00-forest-control')]:
        candidate=selected[task]
        run('S-'+task+'-control',task,records[control]['recipe'],'stability','Контроль на дополнительном разделении',secondary=True)
        members=records[candidate]['model_parameters'].get('members')
        run('S-'+task+'-candidate',task,records[candidate]['recipe'],'stability',
            f'Кандидат {candidate} на дополнительном разделении','S-'+task+'-control',secondary=True,members=members)
    manifest=dict(kind='hypothesis_suite',source_commit=source_commit,selected_secondary_candidates=selected,
                  runs=list(records),completed=True)
    (suite/'suite.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (report_dir/'suite.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'ALL DONE: {len(records)} runs; reports: {report_dir}',flush=True)


if __name__=='__main__':
    main()
