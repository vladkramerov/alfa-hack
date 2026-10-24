"""Portable SPEC-009 regression candidates with train-only fitted state."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from automl_models import Encoder
from automl_reg_combo import FeatureGroupExperts, _fit_cached_estimator, _rows, temporal_folds


def _configuration(model, policy, **extra):
    return {'model_class': type(model).__module__ + '.' + type(model).__name__,
            'parameters': model.get_params(deep=False), 'policy': policy, **extra}


class _WeightedEstimator:
    """Carry weights through the four-argument central fit registrar."""
    def __init__(self, estimator, sample_weight):
        self.estimator, self.sample_weight = estimator, np.asarray(sample_weight, dtype=float)

    def fit(self, X, y):
        self.estimator.fit(X, y, sample_weight=self.sample_weight)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class OriginalLGB:
    def __init__(self, parameters=None, threads=4):
        self.parameters, self.threads = dict(parameters or {}), threads

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        self.encoder_ = Encoder(native=True, scale=False)
        values = self.encoder_.fit_transform(X, y)
        parameters = dict(n_estimators=200, num_leaves=31, learning_rate=.05, min_child_samples=20, reg_lambda=1.)
        parameters.update(self.parameters)
        model = LGBMRegressor(**parameters, random_state=42, n_jobs=self.threads, verbosity=-1)
        config = _configuration(model, 'native-original-target', columns=list(X.columns), target_transform='none')
        self.model_ = _fit_cached_estimator(model, values, np.asarray(y, dtype=float), config)
        self.audit_ = {**config, 'fit_rows': len(X)}
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(self.encoder_.transform(X)), dtype=float)


class SplineGAM:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def _categorical(self, X):
        output = pd.DataFrame(index=X.index)
        for column in self.categorical_:
            values = ('V:' + X[column].astype('string')).fillna('M:').astype(str)
            output[column] = values.where(values.isin(self.category_levels_[column]), 'R:')
        return output

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, SplineTransformer
        from sklearn.linear_model import Ridge
        frame = X.replace([np.inf, -np.inf], np.nan)
        self.numeric_ = frame.select_dtypes(include='number').columns.tolist()
        self.categorical_ = [column for column in frame if column not in self.numeric_]
        self.numeric_medians_ = frame[self.numeric_].median().fillna(0.).to_dict()
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        numeric = self.imputer_.fit_transform(frame[self.numeric_]) if self.numeric_ else np.empty((len(X), 0))
        self.active_numeric_ = np.ptp(numeric, axis=0) > 0
        self.spline_ = None
        if self.active_numeric_.any():
            self.spline_ = SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant')
            self.spline_.fit(numeric[:, self.active_numeric_])
        self.category_levels_ = {}
        for column in self.categorical_:
            values = ('V:' + frame[column].astype('string')).fillna('M:').astype(str)
            counts = values.value_counts()
            levels = sorted(counts.index, key=lambda value: (-int(counts[value]), str(value)))[:127]
            self.category_levels_[column] = levels
        self.ohe_ = None
        if self.categorical_:
            categories = [sorted(self.category_levels_[column] + ['R:']) for column in self.categorical_]
            self.ohe_ = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=True, dtype=np.float64)
            self.ohe_.fit(self._categorical(frame))
        values = self.transform(X)
        model = Ridge(alpha=10., solver='sparse_cg')
        config = _configuration(model, 'spline-gam-v1', numeric_columns=self.numeric_, categorical_columns=self.categorical_,
                                n_knots=5, degree=3, knots='quantile', extrapolation='constant',
                                categorical_cap=128, target_transform='none')
        self.model_ = _fit_cached_estimator(model, values, np.asarray(y, dtype=float), config)
        _rows(self, X, 'spline_gam')
        self.audit_.update(configuration=config, categorical_level_counts={c: len(self.category_levels_[c]) + 1 for c in self.categorical_},
                           numeric_medians=self.numeric_medians_, inactive_numeric=[c for c, active in zip(self.numeric_, self.active_numeric_) if not active],
                           n_output_features=values.shape[1])
        return self

    def transform(self, X):
        frame = X.replace([np.inf, -np.inf], np.nan)
        parts = []
        if self.spline_ is not None:
            values = self.imputer_.transform(frame[self.numeric_])
            parts.append(sparse.csr_matrix(self.spline_.transform(values[:, self.active_numeric_])))
        if self.ohe_ is not None:
            parts.append(self.ohe_.transform(self._categorical(frame)))
        return sparse.hstack(parts, format='csr') if parts else sparse.csr_matrix(np.ones((len(X), 1)))

    def predict(self, X):
        return np.asarray(self.model_.predict(self.transform(X)), dtype=float)


class QuantileMean:
    QUANTILES = (.1, .3, .5, .7, .9)

    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        self.encoder_ = Encoder(native=True, scale=False)
        values = self.encoder_.fit_transform(X, y)
        self.models_, records = [], []
        for quantile in self.QUANTILES:
            parameters = dict(n_estimators=200, num_leaves=31, learning_rate=.05, min_child_samples=20, reg_lambda=1.)
            parameters.update(self.candidate.get('parameters', {}))
            parameters.update(objective='quantile', alpha=quantile)
            model = LGBMRegressor(**parameters, random_state=42, n_jobs=self.threads, verbosity=-1)
            config = _configuration(model, 'quantile-mean-v1', columns=list(X.columns), target_transform='none')
            self.models_.append(_fit_cached_estimator(model, values, np.asarray(y, dtype=float), config))
            records.append({'quantile': quantile, 'fit_rows': len(X), 'configuration': config})
        _rows(self, X, 'quantile_mean')
        self.audit_.update(quantiles=list(self.QUANTILES), components=records, weights=[.2] * 5,
                           target_transform='none', encoder={'native': True, 'scale': False})
        return self

    def predict(self, X):
        values = self.encoder_.transform(X)
        return np.column_stack([model.predict(values) for model in self.models_]).mean(axis=1)


def mask_training_copy(X, y):
    columns = [column for column in X.select_dtypes(include='number') if column != 'Year_Factor']
    mask = np.random.RandomState(42).random_sample((len(X), len(columns))) < .05
    copy = X.copy()
    for index, column in enumerate(columns):
        copy[column] = copy[column].astype(float).mask(mask[:, index])
    return pd.concat([X, copy], ignore_index=True), np.concatenate([np.asarray(y), np.asarray(y)]), mask


class MaskedRegression:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y):
        from automl_reg_combo import RegressionComponent
        augmented, target, self.mask_ = mask_training_copy(X, y)
        self.base_ = RegressionComponent(parameters=self.candidate.get('parameters'), threads=self.threads).fit(augmented, target)
        _rows(self, X, 'masked_regression')
        self.audit_.update(fit_model_rows=len(augmented), masked_columns=[c for c in X.select_dtypes(include='number') if c != 'Year_Factor'],
                           masked_values=int(self.mask_.sum()), mask_probability=.05, protected_columns=['Year_Factor'],
                           target_transform='log1p', augmentation='one masked copy per original training row')
        return self

    def predict(self, X):
        return self.base_.predict(X)


class CovariateShift:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.model_selection import StratifiedKFold
        years = pd.to_numeric(X['Year_Factor'], errors='raise').to_numpy()
        domain = (years == years.max()).astype(int)
        counts = np.bincount(domain, minlength=2)
        if counts.min() < 3:
            raise ValueError('Domain cross-fit needs three rows in both temporal domains')
        self.domain_columns_ = [column for column in X if column != 'Year_Factor']
        self.domain_probability_ = np.full(len(X), np.nan)
        records = []
        for number, (fit, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(X, domain)):
            frame = X.iloc[fit][self.domain_columns_]
            encoder = Encoder(native=False, scale=False)
            values = encoder.fit_transform(frame, domain[fit])
            parameters = dict(n_estimators=100, num_leaves=15, learning_rate=.05, min_child_samples=20, reg_lambda=1.)
            parameters.update(self.candidate.get('domain_parameters', {}))
            model = LGBMClassifier(**parameters, random_state=42, n_jobs=self.threads, verbosity=-1)
            config = _configuration(model, 'covariate-domain-v1', columns=self.domain_columns_, year_excluded=True)
            model = _fit_cached_estimator(model, values, domain[fit], config)
            self.domain_probability_[valid] = model.predict_proba(encoder.transform(X.iloc[valid][self.domain_columns_]))[:, list(model.classes_).index(1)]
            records.append({'fold': number, 'fit_positions': fit.tolist(), 'valid_positions': valid.tolist(),
                            'year_excluded': True, 'configuration': config})
        p = np.clip(self.domain_probability_, 1e-6, 1 - 1e-6)
        unnormalized = np.clip(p / (1 - p) * (counts[0] / counts[1]), .2, 5.)
        self.sample_weight_ = unnormalized / unnormalized.mean()
        self.encoder_ = Encoder(native=False, scale=False)
        values = self.encoder_.fit_transform(X, y)
        parameters = dict(n_estimators=200, max_depth=24, min_samples_leaf=2, max_features=.8)
        parameters.update(self.candidate.get('parameters', {}))
        model = ExtraTreesRegressor(**parameters, random_state=42, n_jobs=self.threads)
        config = _configuration(model, 'covariate-weighted-et-v1', columns=list(X.columns), target_transform='log1p',
                                weights_sha256=hashlib.sha256(self.sample_weight_.astype('<f8').tobytes()).hexdigest())
        wrapped = _WeightedEstimator(model, self.sample_weight_)
        self.model_ = _fit_cached_estimator(wrapped, values, np.log1p(np.asarray(y, dtype=float)), config)
        _rows(self, X, 'covariate_shift')
        self.audit_.update(domain_folds=records, domain_columns=self.domain_columns_, domain_counts=counts.tolist(),
                           latest_training_year=float(years.max()), domain_probability_source='three_fold_cross_fit_within_training',
                           weight_min=float(self.sample_weight_.min()), weight_max=float(self.sample_weight_.max()), weight_mean=float(self.sample_weight_.mean()),
                           weight_clip_before_normalization=[.2, 5.], weighted_configuration=config, target_transform='log1p')
        return self

    def predict(self, X):
        return np.expm1(self.model_.predict(self.encoder_.transform(X)))


def _h43_candidate(parameters=None):
    return {'strategy': 'reg_feature_experts', 'feature_numbers': [29, 30, 31], 'group_column': 'building_class',
            'minimum_expert_rows': 500, 'shrinkage': 500., 'parameters': dict(parameters or {})}


def _cached_base_fold(X, y, candidate, threads, cache_dir):
    if cache_dir is None:
        return FeatureGroupExperts(candidate, threads).fit(X, y), None
    import joblib
    from automl_combo_memory import _data_hash
    identity = {'X': _data_hash(X), 'y': _data_hash(np.asarray(y)), 'candidate': candidate}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / 'h43-temporal-models' / (key + '.joblib')
    if path.exists():
        return joblib.load(path), str(path)
    model = FeatureGroupExperts(candidate, threads).fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.joblib')
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)
    return model, str(path)


class H43Residual:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y, base_model=None, cache_dir=None):
        y = np.asarray(y, dtype=float)
        base_candidate = _h43_candidate(self.candidate.get('base_parameters'))
        self.oof_prediction_, fold_ids = np.full(len(X), np.nan), np.full(len(X), -1)
        records = []
        for number, (fit, valid) in enumerate(temporal_folds(X)):
            model, artifact = _cached_base_fold(X.iloc[fit], y[fit], base_candidate, self.threads, cache_dir)
            self.oof_prediction_[valid] = model.predict(X.iloc[valid])
            fold_ids[valid] = number
            records.append({'fold': number, 'fit_positions': fit.tolist(), 'valid_positions': valid.tolist(),
                            'fit_rows': len(fit), 'valid_rows': len(valid), 'model_artifact': artifact})
        self.residual_positions_ = np.flatnonzero(fold_ids >= 0)
        self.residual_target_ = y[self.residual_positions_] - self.oof_prediction_[self.residual_positions_]
        parameters = dict(n_estimators=200, max_depth=3, num_leaves=7, learning_rate=.03, min_child_samples=20, reg_lambda=10.)
        parameters.update(self.candidate.get('parameters', {}))
        self.residual_ = OriginalLGB(parameters, self.threads).fit(X.iloc[self.residual_positions_], self.residual_target_)
        self.base_ = base_model if base_model is not None else FeatureGroupExperts(base_candidate, self.threads).fit(X, y)
        _rows(self, X, 'h43_residual')
        self.audit_.update(base_oof_folds=records, residual_fit_rows=len(self.residual_positions_),
                           residual_target='original_y_minus_training_temporal_oof_prediction', correction_factor=1.,
                           base_outer_reused=base_model is not None, base_candidate=base_candidate, residual=self.residual_.audit_,
                           residual_fit_is_independent_validation=False)
        return self

    def predict(self, X):
        return np.asarray(self.base_.predict(X)) + self.residual_.predict(X)


def build_next_reg(candidate, threads=4):
    classes = {'spline_gam': SplineGAM, 'quantile_mean': QuantileMean, 'h43_residual': H43Residual,
               'covariate_shift': CovariateShift, 'masked_regression': MaskedRegression}
    if candidate.get('strategy') not in classes:
        raise ValueError('Unknown next regression strategy')
    return classes[candidate['strategy']](candidate, threads)
