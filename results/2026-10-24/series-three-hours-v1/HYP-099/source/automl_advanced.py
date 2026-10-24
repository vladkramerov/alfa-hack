"""Flat-archive implementations of the measured HYP-035 and HYP-037 policies.

The legacy encoder is intentionally fitted independently for every model.
Imports of automl_models are delayed to keep its factory import acyclic.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


def calibrate_probabilities(probabilities, classes, mapping):
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[1] != len(classes) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p.sum(axis=1) <= 0)):
        raise ValueError('Invalid class probabilities')
    values = {str(label): float(value) for label, value in mapping.items()}
    if any(not np.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError('Class multipliers must be finite and positive')
    factors = np.array([values.get(str(label), 1.) for label in classes])
    result = p * factors
    return result / result.sum(axis=1, keepdims=True)


def calibration_folds(X, y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return [], 'single_class'
    if min(counts) < 3:
        return [], 'fewer_than_three_rows_in_a_class'
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if any(len(np.unique(groups[y == label])) < 3 for label in classes):
        return [], 'fewer_than_three_raw_groups_in_a_class'
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(X, y, groups))
    for fit, valid in folds:
        if not len(valid) or not np.array_equal(np.unique(y[fit]), classes):
            return [], 'inner_fit_missing_a_class'
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError('Calibration raw duplicate groups cross a fold boundary')
    return folds, None


def _choose_multipliers(y, probabilities, classes):
    # This is the frozen HYP-037 rule: one lexical coordinate pass, fixed
    # first-class anchor, and strict improvement (deterministic tie handling).
    factors = np.ones(len(classes))
    score = float(f1_score(y, classes[np.argmax(probabilities, axis=1)], average='macro', zero_division=0))
    for index in range(1, len(classes)):
        best = factors[index]
        for value in (.75, 1., 1.25, 1.5, 2.):
            candidate = factors.copy()
            candidate[index] = value
            candidate_score = float(f1_score(y, classes[np.argmax(probabilities * candidate, axis=1)], average='macro', zero_division=0))
            if candidate_score > score + 1e-12:
                score, best = candidate_score, value
        factors[index] = best
    return {str(label): float(value) for label, value in zip(classes, factors)}, score


class _LegacyBase:
    def __init__(self, kind, parameters, threads):
        self.kind, self.parameters, self.threads = kind, dict(parameters), threads

    def fit(self, X, y):
        from automl_models import Encoder
        target = np.asarray(y)
        self.encoder_ = Encoder(native=False, scale=False)
        values = self.encoder_.fit_transform(X, target)
        if self.kind == 'regression':
            from sklearn.ensemble import ExtraTreesRegressor
            numeric = np.asarray(target, dtype=float)
            if not np.isfinite(numeric).all() or np.any(numeric <= -1):
                raise ValueError('Invalid log1p target domain')
            settings = dict(n_estimators=200, max_depth=24, min_samples_leaf=2, max_features=.8)
            settings.update(self.parameters)
            self.model_ = ExtraTreesRegressor(**settings, random_state=42, n_jobs=self.threads)
            self.model_.fit(values, np.log1p(numeric))
        else:
            self.labels_ = LabelEncoder().fit(target)
            self.classes_ = self.labels_.classes_
            if len(self.classes_) == 1:
                self.model_ = None
                return self
            from xgboost import XGBClassifier
            settings = dict(n_estimators=600, max_depth=6, learning_rate=.05, reg_lambda=1, tree_method='hist', device='cpu')
            settings.update(self.parameters)
            self.model_ = XGBClassifier(**settings, random_state=42, n_jobs=self.threads)
            self.model_.fit(values, self.labels_.transform(target))
        return self

    def predict_proba(self, X):
        if self.kind == 'regression':
            raise ValueError('Regression has no class probabilities')
        if self.model_ is None:
            return np.ones((len(X), 1))
        result = np.asarray(self.model_.predict_proba(self.encoder_.transform(X)))
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite probabilities')
        return result

    def predict(self, X):
        if self.kind != 'regression':
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        result = np.expm1(np.asarray(self.model_.predict(self.encoder_.transform(X))).reshape(-1))
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite predictions')
        return result


class GroupExperts:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y):
        y = np.asarray(y)
        parameters = self.candidate.get('parameters', {})
        self.group_column_ = self.candidate.get('group_column', 'building_class')
        self.experts_, self.weights_ = {}, {}
        minimum = int(self.candidate.get('minimum_expert_rows', 500))
        shrinkage = float(self.candidate.get('shrinkage', 500.))
        if minimum < 500 or not np.isfinite(shrinkage) or shrinkage <= 0:
            raise ValueError('Invalid group expert minimum/shrinkage')
        self.global_ = _LegacyBase('regression', parameters, self.threads).fit(X, y)
        groups = sorted(X[self.group_column_].dropna().unique(), key=str) if self.group_column_ in X else []
        records = []
        if len(groups) >= 2:
            for group in groups:
                positions = np.flatnonzero(X[self.group_column_].eq(group).fillna(False).to_numpy(bool))
                weight = len(positions) / (len(positions) + shrinkage)
                trained = len(positions) >= minimum
                records.append({'group': str(group), 'fit_rows': len(positions), 'weight': weight, 'fitted': trained})
                if trained:
                    self.experts_[group] = _LegacyBase('regression', parameters, self.threads).fit(X.iloc[positions], y[positions])
                    self.weights_[group] = weight
        self.keep_mask_ = np.ones(len(X), dtype=bool)
        self.removed_positions_ = np.empty(0, dtype=int)
        self.audit_ = {'policy': 'group-experts-v1', 'strategy': 'group_experts', 'fit_original_rows': len(X),
                       'fit_retained_rows': len(X), 'target_transform': 'log1p', 'group_column': self.group_column_,
                       'minimum_expert_rows': minimum, 'shrinkage': shrinkage, 'experts': records,
                       'global_fit_rows': len(X), 'encoder': {'native': False, 'scale': False},
                       'fallback': 'global model for missing/unseen/undersized groups'}
        return self

    def predict(self, X):
        prediction = self.global_.predict(X)
        if self.group_column_ in X:
            for group, expert in self.experts_.items():
                positions = np.flatnonzero(X[self.group_column_].eq(group).fillna(False).to_numpy(bool))
                if len(positions):
                    weight = self.weights_[group]
                    prediction[positions] = weight * expert.predict(X.iloc[positions]) + (1 - weight) * prediction[positions]
        return prediction

    def predict_proba(self, X):
        raise ValueError('Regression has no class probabilities')


class MacroF1Calibrated:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        parameters = self.candidate.get('parameters', {})
        supplied = self.candidate.get('class_multipliers')
        folds_audit, score = [], None
        if supplied is not None:
            # Validate before fitting; freeze means no inner training on refit.
            calibrate_probabilities(np.ones((1, len(self.classes_))), self.classes_, supplied)
            mapping = {str(label): float(supplied.get(str(label), 1.)) for label in self.classes_}
            source, reason = 'frozen_candidate', None
        else:
            folds, reason = calibration_folds(X, y)
            mapping = {str(label): 1. for label in self.classes_}
            source = 'identity_insufficient_data'
            if folds:
                probabilities = np.full((len(X), len(self.classes_)), np.nan)
                for number, (fit, valid) in enumerate(folds):
                    model = _LegacyBase('classification', parameters, self.threads).fit(X.iloc[fit], y[fit])
                    p = model.predict_proba(X.iloc[valid])
                    for index, label in enumerate(model.classes_):
                        probabilities[valid, list(self.classes_).index(label)] = p[:, index]
                    folds_audit.append({'fold': number, 'fit_rows': len(fit), 'valid_rows': len(valid),
                                        'fit_positions': fit.tolist(), 'valid_positions': valid.tolist(), 'raw_groups_disjoint': True})
                if not np.isfinite(probabilities).all():
                    raise ValueError('Incomplete calibration OOF probabilities')
                mapping, score = _choose_multipliers(y, probabilities, self.classes_)
                source = 'training_grouped_oof'
        self.calibration_multipliers_ = mapping
        self.base_ = _LegacyBase('classification', parameters, self.threads).fit(X, y)
        self.classes_ = self.base_.classes_
        self.keep_mask_ = np.ones(len(X), dtype=bool)
        self.removed_positions_ = np.empty(0, dtype=int)
        self.audit_ = {'policy': 'macro-f1-v1', 'strategy': 'macro_f1_calibrated', 'fit_original_rows': len(X),
                       'fit_retained_rows': len(X), 'target_transform': 'none', 'encoder': {'native': False, 'scale': False},
                       'calibration_multipliers': dict(mapping), 'multipliers': dict(mapping), 'calibration_source': source,
                       'calibration_skip_reason': reason, 'inner_macro_f1': score, 'folds': folds_audit,
                       'calibration_grid': [.75, 1., 1.25, 1.5, 2.], 'calibration_passes': 1,
                       'fixed_anchor': str(self.classes_[0])}
        return self

    def predict_proba(self, X):
        return calibrate_probabilities(self.base_.predict_proba(X), self.classes_, self.calibration_multipliers_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_advanced(kind, candidate, threads=4):
    candidate = dict(candidate)
    parameters = dict(candidate.get('parameters', {}))
    for key in ('random_state', 'random_seed', 'seed', 'n_jobs', 'thread_count', 'verbose'):
        parameters.pop(key, None)
    strategy = candidate.get('strategy')
    allowed = ({'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'criterion'} if strategy == 'group_experts'
               else {'n_estimators', 'max_depth', 'learning_rate', 'reg_lambda', 'reg_alpha', 'subsample', 'colsample_bytree', 'tree_method', 'device'})
    if set(parameters) - allowed:
        raise ValueError('Unsupported advanced model parameters')
    for key, upper in (('n_estimators', 1200), ('max_depth', 32)):
        if key in parameters and (not isinstance(parameters[key], int) or not 1 <= parameters[key] <= upper):
            raise ValueError('Parameter exceeds worker resource bounds: ' + key)
    if parameters.get('device', 'cpu') != 'cpu':
        raise ValueError('Only CPU training is permitted')
    candidate['parameters'] = parameters
    if kind == 'regression' and strategy == 'group_experts':
        return GroupExperts(candidate, threads)
    if kind in {'classification', 'multiclass'} and strategy == 'macro_f1_calibrated':
        return MacroF1Calibrated(candidate, threads)
    raise ValueError('Advanced strategy is not applicable to this task')
