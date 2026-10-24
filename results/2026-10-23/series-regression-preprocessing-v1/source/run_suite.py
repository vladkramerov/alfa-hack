"""Controlled preprocessing and regularization research, not a platform agent.

Choose configurations on inner validation, freeze the complete outer panel,
then evaluate that panel without using outer feedback to change any model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_squared_error)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from experiments.preprocessing import DataPreprocessor


ROOT = Path(__file__).resolve().parents[1]
MSK = timezone(timedelta(hours=3))
RECIPES = {
    'baseline': {},
    'correlation': {'drop_correlated': True},
    'vif': {'drop_correlated': True, 'drop_vif': True},
    'scaled': {'drop_correlated': True, 'drop_vif': True, 'scale': True},
    'gaussian': {'drop_correlated': True, 'drop_vif': True, 'scale': True, 'gaussian': True},
    'engineered': {'drop_correlated': True, 'drop_vif': True, 'scale': True, 'gaussian': True, 'engineer': True},
}
HYPOTHESES = {
    'baseline': 'HYP-001', 'correlation': 'HYP-001', 'vif': 'HYP-002',
    'scaled': 'HYP-003', 'gaussian': 'HYP-004', 'engineered': 'HYP-005',
    'native-l2': 'HYP-006', 'linear': 'HYP-007', 'final': 'HYP-005',
}


def now():
    return datetime.now(MSK).isoformat(timespec='seconds')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=_json_default) + '\n', encoding='utf-8')


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def build_splits(X, y, task):
    positions = np.arange(len(X))
    if task == 'regression':
        years = sorted(X['Year_Factor'].unique())
        if len(years) < 3:
            raise ValueError('At least three years are required for temporal inner/outer validation')
        outer_fit = positions[X.Year_Factor.to_numpy() < years[-1]]
        outer_valid = positions[X.Year_Factor.to_numpy() == years[-1]]
        inner_fit = positions[X.Year_Factor.to_numpy() < years[-2]]
        inner_valid = positions[X.Year_Factor.to_numpy() == years[-2]]
        description = f'temporal outer: years <{years[-1]} -> {years[-1]}; inner: <{years[-2]} -> {years[-2]}'
    else:
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        if len(np.unique(groups)) < len(groups):
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            outer_fit, outer_valid = next(splitter.split(X, y, groups))
            inner_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
            a, b = next(inner_splitter.split(X.iloc[outer_fit], y.iloc[outer_fit], groups[outer_fit]))
            inner_fit, inner_valid = outer_fit[a], outer_fit[b]
            description = 'stratified group outer5/inner4 first folds; exact raw feature groups; seed42'
        else:
            outer_fit, outer_valid = train_test_split(positions, test_size=.2, stratify=y, random_state=42)
            inner_fit, inner_valid = train_test_split(outer_fit, test_size=.25, stratify=y.iloc[outer_fit], random_state=42)
            description = 'stratified outer80/20, inner75/25 within outer fit; seed42'
        classes = set(y)
        for indices in (inner_fit, inner_valid, outer_fit, outer_valid):
            if set(y.iloc[indices]) != classes:
                raise ValueError('A split lacks a class; revise protocol before training')
        for fit, valid in ((inner_fit, inner_valid), (outer_fit, outer_valid)):
            if set(groups[fit]) & set(groups[valid]):
                raise ValueError('Exact feature duplicate leakage across split')
    for fit, valid in ((inner_fit, inner_valid), (outer_fit, outer_valid)):
        if not len(fit) or not len(valid) or set(fit) & set(valid):
            raise ValueError('Empty or overlapping validation split')
    return dict(inner_fit=np.asarray(inner_fit), inner_valid=np.asarray(inner_valid),
                outer_fit=np.asarray(outer_fit), outer_valid=np.asarray(outer_valid), description=description)


def metric_values(y, prediction, task):
    if task == 'regression':
        return {'rmse': float(np.sqrt(mean_squared_error(y, prediction)))}
    return {'macro_f1': float(f1_score(y, prediction, average='macro', labels=sorted(set(y)), zero_division=0)),
            'accuracy': float(accuracy_score(y, prediction)),
            'weighted_f1': float(f1_score(y, prediction, average='weighted', zero_division=0))}


def validate_submission(submission, test, sample, task, classes):
    target = 'site_eui' if task == 'regression' else 'attack_cat'
    if list(submission) != list(sample) or len(submission) != len(test):
        raise ValueError('Submission schema/row count differs from template')
    if not submission.id.reset_index(drop=True).equals(test.id.reset_index(drop=True)):
        raise ValueError('Submission IDs/order differ from test')
    if not submission.id.is_unique or submission.isna().any().any():
        raise ValueError('Submission has missing predictions or duplicate IDs')
    if task == 'regression' and not np.isfinite(submission[target].to_numpy(dtype=float)).all():
        raise ValueError('Regression predictions must be finite')
    if task != 'regression' and not set(submission[target]).issubset(set(classes)):
        raise ValueError('Unknown predicted class')


def create_run_dir(parent, hypothesis, slug):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    reservations = parent / '.run-index'
    reservations.mkdir(exist_ok=True)
    for number in range(1, 10000):
        prefix = f'{hypothesis}-{number:02d}-'
        if any(parent.glob(prefix + '*')):
            continue
        try:
            with (reservations / f'{hypothesis}-{number:02d}.lock').open('x', encoding='utf-8') as stream:
                stream.write(slug + '\n')
        except FileExistsError:
            continue
        directory = parent / (prefix + slug)
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory
    raise RuntimeError('No free experiment run identifier')


def build_model(task, config, threads):
    family = config['family']
    regression = task == 'regression'
    if family == 'catboost':
        cls = CatBoostRegressor if regression else CatBoostClassifier
        return cls(iterations=500, depth=6, learning_rate=.05, l2_leaf_reg=config.get('strength', 3),
                   random_seed=42, thread_count=threads, loss_function='RMSE' if regression else 'MultiClass',
                   allow_writing_files=False, use_best_model=False, verbose=False)
    if family == 'forest':
        cls = RandomForestRegressor if regression else RandomForestClassifier
        return cls(n_estimators=150, max_depth=20, min_samples_leaf=2, max_features=.8,
                   random_state=42, n_jobs=threads)
    if regression:
        if family == 'l1':
            return Lasso(alpha=config['strength'], max_iter=10000, tol=1e-3)
        return Ridge(alpha=config['strength'], solver='lsqr', tol=1e-6)
    return LogisticRegression(penalty=family, C=config['strength'], solver='saga',
                              max_iter=1000, tol=1e-3, random_state=42, n_jobs=1)


def config_key(config):
    return json.dumps(config, sort_keys=True)


class ExperimentSuite:
    def __init__(self, task, threads, tag):
        self.task, self.threads, self.tag = task, threads, tag
        self.target = 'site_eui' if task == 'regression' else 'attack_cat'
        self.data_dir = ROOT / 'data' / task
        self.train = pd.read_csv(self.data_dir / 'train.csv')
        self.test = pd.read_csv(self.data_dir / 'test.csv')
        self.sample = pd.read_csv(self.data_dir / 'sample_submission.csv')
        self.X = self.train.drop(columns=['id', self.target])
        self.y = self.train[self.target]
        self.classes = sorted(self.y.unique()) if task != 'regression' else None
        if list(self.test.drop(columns='id')) != list(self.X):
            raise ValueError('Train/test feature schema mismatch')
        self.splits = build_splits(self.X, self.y, task)
        self.root = ROOT / 'results' / datetime.now(MSK).strftime('%Y-%m-%d')
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory = self.root / f'series-{task}-{tag}'
        self.directory.mkdir(exist_ok=False)
        self.records, self.cache = [], {}
        hashes = {name: hashlib.sha256((self.data_dir / name).read_bytes()).hexdigest()
                  for name in ['train.csv', 'test.csv', 'sample_submission.csv']}
        code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / 'experiments').glob('*.py'))}
        self.environment = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                scipy=scipy.__version__, sklearn=sklearn.__version__, catboost=catboost.__version__,
                                joblib=joblib.__version__, threads=threads, data_sha256=hashes, code_sha256=code_hashes)
        save_json(self.directory / 'environment.json', self.environment)
        np.savez_compressed(self.directory / 'splits.npz', **{k:v for k,v in self.splits.items() if k != 'description'})
        split_info = {'description': self.splits['description'], 'sizes': {k:len(v) for k,v in self.splits.items() if k!='description'}}
        if self.classes:
            split_info['class_counts'] = {k: self.y.iloc[v].value_counts().to_dict() for k,v in self.splits.items() if k!='description'}
        save_json(self.directory / 'splits.json', split_info)
        print(f'{now()} {task}: {split_info}', flush=True)

    def utility(self, scores):
        return -scores['rmse'] if self.task == 'regression' else scores['macro_f1']

    def prepared(self, stage, recipe):
        key = stage, recipe
        if key not in self.cache:
            fit, valid = self.splits[stage + '_fit'], self.splits[stage + '_valid']
            preprocessor = DataPreprocessor(self.task, **RECIPES[recipe])
            A = preprocessor.fit_transform(self.X.iloc[fit])
            B = preprocessor.transform(self.X.iloc[valid])
            self.cache[key] = (preprocessor, A, B)
        return self.cache[key]

    def evaluate(self, config, stage, hypothesis=None, reference=None):
        cached = next((r for r in self.records if r['stage']==stage and r['config']==config and r['status']=='completed'), None)
        if cached:
            return cached
        family, recipe = config['family'], config['recipe']
        hypothesis = hypothesis or HYPOTHESES[recipe]
        suffix = f"-{config['strength']}" if 'strength' in config else ''
        directory = create_run_dir(self.root, hypothesis, f'{self.task}-{stage}-{recipe}-{family}{suffix}')
        record = dict(run_id=directory.name, path=str(directory.relative_to(ROOT)), hypothesis=hypothesis,
                      task=self.task, stage=stage, config=config, status='running', started=now(),
                      reference=reference['run_id'] if reference else None)
        save_json(directory / 'metrics.json', record)
        (directory / 'report.md').write_text(f'# {directory.name}\n\nСтатус: Выполняется.\n', encoding='utf-8')
        started = time.monotonic()
        try:
            preprocessor, A, B = self.prepared(stage, recipe)
            fit, valid = self.splits[stage+'_fit'], self.splits[stage+'_valid']
            model = build_model(self.task, config, self.threads)
            with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=self.threads):
                warnings.simplefilter('always')
                model.fit(A, self.y.iloc[fit])
                prediction = np.asarray(model.predict(B)).reshape(-1)
            scores = metric_values(self.y.iloc[valid].to_numpy(), prediction, self.task)
            record.update(status='completed', finished=now(), seconds=round(time.monotonic()-started,3), scores=scores,
                          n_fit=len(fit), n_validation=len(valid), n_features=A.shape[1],
                          model_parameters=model.get_params(), warnings=[str(w.message) for w in captured],
                          converged=not any(issubclass(w.category, ConvergenceWarning) for w in captured),
                          n_iter=getattr(model, 'n_iter_', None), environment=self.environment,
                          split=self.splits['description'])
            if reference:
                record['gain'] = self.utility(scores) - self.utility(reference['scores'])
            predictions = pd.DataFrame({'row_position': valid, 'id': self.train.iloc[valid].id.to_numpy(),
                                         'actual': self.y.iloc[valid].to_numpy(), 'prediction': prediction})
            predictions.to_csv(directory / 'validation_predictions.csv', index=False)
            if self.classes:
                (directory/'classification_report.txt').write_text(classification_report(
                    self.y.iloc[valid], prediction, labels=self.classes, zero_division=0, digits=6), encoding='utf-8')
            save_json(directory / 'preprocessing_audit.json', preprocessor.audit_)
            save_json(directory / 'metrics.json', record)
            self.write_report(directory, record)
            self.records.append(record)
            save_json(self.directory / 'records.json', self.records)
            print(f'{now()} {self.task} {stage} {family}/{recipe}{suffix}: {scores}, {record["seconds"]}s, converged={record["converged"]}', flush=True)
            del model
            return record
        except BaseException as exc:
            record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                          finished=now(), seconds=round(time.monotonic()-started,3), error=repr(exc))
            (directory/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
            save_json(directory/'metrics.json', record)
            self.write_report(directory, record)
            self.records.append(record)
            save_json(self.directory/'records.json', self.records)
            raise

    def write_report(self, directory, record):
        gain = record.get('gain')
        conclusion = 'Не оценивалась' if record['status']!='completed' else ('Недостаточно данных' if gain is None else
                      'Подтверждена в проверенных условиях' if gain>0 else 'Не подтверждена по заданному критерию')
        if record.get('converged') is False:
            conclusion = 'Недостаточно данных'
        text = (f'# {record["run_id"]}\n\n'
                f'- Начало: {record["started"]}\n- Окончание: {record.get("finished")}\n'
                f'- Статус: {record["status"]}\n- Вывод: {conclusion}\n'
                f'- Гипотеза: [описание](../../../docs/hypotheses/{record["hypothesis"]}.md)\n'
                '- Спецификация: [SPEC-002](../../../docs/specs/SPEC-002-preprocessing-model-comparison.md)\n'
                f'- Задача: {self.task}; stage: {record["stage"]}.\n'
                f'- Команда: `.venv-ml/Scripts/python.exe -m experiments.run_suite --task {self.task} --threads {self.threads} --tag {self.tag}`\n'
                f'- Контроль: {record.get("reference")}; выигрыш: {gain}.\n\n'
                'Критерий до запуска: RMSE меньше контроля / Macro F1 больше контроля; один и тот же split.\n'
                'Для контроля без парного сравнения или grid-кандидата без reference вывод по гипотезе не выставляется.\n\n'
                '[Полные параметры, версии, hashes и метрики](metrics.json).\n'
                + ('[Предобработка](preprocessing_audit.json), [validation predictions](validation_predictions.csv).\n' if record['status']=='completed' else '[Ошибка](error.log).\n')
                + '\nInner используется для выбора; outer — панель заранее замороженных конфигураций.\n'
                  'Outer сравнение полного recipe с baseline измеряет совокупный эффект pipeline, а не отдельный вклад feature engineering.\n'
                  'Результат ограничен выбранным split и бюджетом, не является оценкой скрытого test.\n')
        (directory/'report.md').write_text(text, encoding='utf-8')

    def run(self):
        # Each ablation adds one family of transformations. Compare to the
        # immediately preceding recipe, not whichever outer result looks best.
        ablations = {family: [] for family in ['catboost', 'forest']}
        for family in ablations:
            previous = None
            for recipe in RECIPES:
                record = self.evaluate(dict(family=family, recipe=recipe), 'inner', reference=previous)
                ablations[family].append(record)
                previous = record
        selected = {family: max(records, key=lambda r: self.utility(r['scores'])) for family,records in ablations.items()}
        l2_grid = []
        for strength in [1, 3, 10, 30]:
            cfg = dict(selected['catboost']['config'])
            if strength != 3:
                cfg['strength'] = strength
            l2_grid.append(self.evaluate(cfg, 'inner', 'HYP-006', selected['catboost']))
        cat_l2 = max(l2_grid, key=lambda r: self.utility(r['scores']))
        linear = {}
        for family in ['l1', 'l2']:
            strengths = ([.01, .1, 1] if family == 'l1' else [.1, 1, 10]) if self.task=='regression' else [.01, .1, 1]
            candidates = [self.evaluate(dict(family=family, recipe='engineered', strength=strength), 'inner', 'HYP-007')
                          for strength in strengths]
            converged = [r for r in candidates if r['converged']]
            linear[family] = max(converged or candidates, key=lambda r: self.utility(r['scores']))
        # Freeze all comparison and final-refit decisions before any outer fit.
        panel = []
        for family in ['catboost', 'forest']:
            panel += [dict(family=family, recipe='baseline'), dict(family=family, recipe='engineered'), selected[family]['config']]
        panel += [cat_l2['config'], linear['l1']['config'], linear['l2']['config']]
        unique_panel = list({config_key(config):config for config in panel}.values())
        best_tree = max([selected['forest'], cat_l2], key=lambda r:self.utility(r['scores']))
        frozen = dict(frozen_at=now(), selection='inner only; outer never changes configuration',
                      preprocessing={family:r['config'] for family,r in selected.items()},
                      catboost_native_l2=cat_l2['config'], linear={k:r['config'] for k,r in linear.items()},
                      linear_inner_converged={k:r['converged'] for k,r in linear.items()},
                      best_tree_for_final_refit=best_tree['config'], outer_panel=unique_panel)
        save_json(self.directory/'frozen_selection.json', frozen)
        outer_records = []
        for cfg in unique_panel:
            hyp = 'HYP-007' if cfg['family'] in ['l1','l2'] else 'HYP-006' if 'strength' in cfg else HYPOTHESES[cfg['recipe']]
            baseline = next((r for r in outer_records if r['config']==dict(family=cfg['family'],recipe='baseline')), None)
            if cfg['family']=='catboost' and 'strength' in cfg:
                baseline = next(r for r in outer_records if r['config']==selected['catboost']['config'])
            if cfg['family']=='l2':
                baseline = next(r for r in outer_records if r['config']==linear['l1']['config'])
            if baseline is not None and baseline['config']==cfg:
                baseline = None
            outer_records.append(self.evaluate(cfg,'outer',hyp,baseline))
        final = self.refit_final(best_tree['config'])
        summary = dict(task=self.task, completed_at=now(), directory=str(self.directory.relative_to(ROOT)),
                       split=self.splits['description'], inner_selected={k:r['run_id'] for k,r in selected.items()},
                       frozen=frozen, records=self.records, final=final,
                       limitation='Outer panel is descriptive held-out comparison of inner-frozen choices; no general superiority claim.')
        save_json(self.directory/'summary.json',summary)
        return summary

    def refit_final(self, config):
        directory = create_run_dir(self.root, 'HYP-005', self.task+'-full-refit')
        initial = dict(status='running', started=now(), task=self.task, config=config,
                       selection='frozen inner-only winner; full refit is not a new validation measurement')
        save_json(directory/'metrics.json', initial)
        (directory/'report.md').write_text('# Полное обучение\n\nСтатус: Выполняется.\n',encoding='utf-8')
        try:
            prep = DataPreprocessor(self.task, **RECIPES[config['recipe']])
            A = prep.fit_transform(self.X)
            B = prep.transform(self.test.drop(columns='id'))
            model = build_model(self.task, config, self.threads)
            started = time.monotonic()
            with threadpool_limits(limits=self.threads):
                model.fit(A,self.y)
                prediction = np.asarray(model.predict(B)).reshape(-1)
            submission = pd.DataFrame({'id':self.test.id,self.target:prediction})
            validate_submission(submission,self.test,self.sample,self.task,self.classes)
            submission.to_csv(directory/'submission.csv',index=False)
            validate_submission(pd.read_csv(directory/'submission.csv'),self.test,self.sample,self.task,self.classes)
            joblib.dump(Pipeline([('preprocess',prep),('model',model)]),directory/'model.joblib',compress=3)
            np.savez_compressed(directory/'preprocessed_data.npz',X_train=A,X_test=B,
                                y_train=self.y.to_numpy(),train_id=self.train.id.to_numpy(),test_id=self.test.id.to_numpy(),
                                feature_names=prep.get_feature_names_out().astype(str))
            save_json(directory/'preprocessing_audit.json',prep.audit_)
            result=dict(initial,status='completed',finished=now(),fit_seconds=round(time.monotonic()-started,3),
                        path=str(directory.relative_to(ROOT)),n_features=A.shape[1],environment=self.environment,
                        submission_sha256=hashlib.sha256((directory/'submission.csv').read_bytes()).hexdigest())
            save_json(directory/'metrics.json',result)
            (directory/'report.md').write_text('# Полное обучение и предсказания\n\n'
                'Конфигурация выбрана и заморожена по inner до оценки outer. Это техническое обучение на всём train, не новая проверка гипотезы.\n\n'
                '[Метаданные](metrics.json), [submission](submission.csv), [pipeline](model.joblib), '
                '[преобразованные массивы](preprocessed_data.npz), [аудит](preprocessing_audit.json).\n',encoding='utf-8')
            print(f'{now()} {self.task} FINAL: {config} -> {directory}',flush=True)
            return result
        except BaseException as exc:
            initial.update(status='error',finished=now(),error=repr(exc))
            save_json(directory/'metrics.json',initial)
            (directory/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
            (directory/'report.md').write_text('# Полное обучение\n\nОшибка: см. error.log.\n',encoding='utf-8')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['regression','multiclass'],required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--tag',default='preprocessing-v1')
    args=parser.parse_args()
    if args.threads<1 or args.threads>4:
        parser.error('threads must be between 1 and 4')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'matplotlib'))
    suite=ExperimentSuite(args.task,args.threads,args.tag)
    suite.run()


if __name__=='__main__':
    main()
