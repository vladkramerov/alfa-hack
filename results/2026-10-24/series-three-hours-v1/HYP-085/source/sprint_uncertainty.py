"""H85 conditional lognormal mean with a jointly learned log-variance head.

The Gaussian model is for log1p(y), not y. Mean reconstruction is conditional
on that assumption; caps are fixed numerical/variance guards, never fitted to
outer labels. No baseline model or research cache is read or fitted here.
"""
from copy import deepcopy
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from automl_reg_combo import FeatureComposition


def conditional_lognormal_mean(raw_parameters, variance_cap=2.):
    """Return nonnegative exp(mu+.5*min(exp(2*log_sigma),cap))-1.

    Fixed log_sigma floor -350 and exponent bounds[0,700] prevent floating
    overflow and negative consumption; neither guard uses training/outer y.
    RawFormulaVal second column is log_sigma, not variance or log_variance.
    """
    raw = np.asarray(raw_parameters, dtype=float)
    if (raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all()
            or not np.isfinite(variance_cap) or variance_cap <= 0):
        raise ValueError('Expected finite N by 2 mu/log_sigma and a positive variance cap')
    bounded_log_sigma = np.clip(raw[:, 1], -350., .5 * np.log(variance_cap))
    variance = np.exp(2 * bounded_log_sigma)
    return np.expm1(np.clip(raw[:, 0] + .5 * variance, 0., 700.))


class ConditionalLognormalRegressor:
    """Fixed CatBoost800/depth6 learns mu(X) and log_sigma(X) of log1p(y)."""
    def __init__(self, threads=2):
        if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 4:
            raise ValueError('Need one to four CPU threads')
        self.threads = int(threads)
        self.parameters = dict(iterations=800, depth=6, learning_rate=.05, l2_leaf_reg=10.,
            loss_function='RMSEWithUncertainty', random_seed=42, thread_count=self.threads,
            verbose=False, allow_writing_files=False)

    def configuration(self):
        return {'hypothesis': 'HYP-085', 'strategy': 'conditional_lognormal_mean', 'version': 1,
                'family': 'catboost', 'parameters': deepcopy(self.parameters),
                'feature_numbers': [29, 30, 31], 'target_transform': 'log1p',
                'loss': 'Gaussian_negative_log_likelihood_on_log1p_y',
                'raw_parameters': ['conditional_log_mean_mu', 'conditional_log_standard_deviation'],
                'prediction': 'expm1(mu+0.5*min(exp(2*log_sigma),2))',
                'variance_cap': 2., 'numerical_log_sigma_floor': -350.,
                'numerical_log_mean_bounds': [0., 700.],
                'preprocessing': 'fit_only_feature_composition_numeric_inf_to_nan_native_tagged_categories',
                'class_or_global_smearing_factor': False, 'base_used': False,
                'query_statistics': False, 'row_deletion': False,
                'selection': 'fixed_all_rounds_no_validation_earlystop'}

    def _prepare(self, frame):
        out = frame.copy()
        for column in out:
            if column in self.categorical_:
                out[column] = ('V:' + out[column].astype('string')).fillna('M:').astype(str)
            else:
                out[column] = pd.to_numeric(out[column], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
        return out

    def fit(self, X, y, base_predictor=None):
        target = np.asarray(y, dtype=float)
        if (not isinstance(X, pd.DataFrame) or not len(X) or not X.columns.is_unique
                or not all(isinstance(c, str) for c in X)
                or target.shape != (len(X),) or not np.isfinite(target).all() or np.any(target < 0)):
            raise ValueError('Expected nonempty unique-column DataFrame and finite nonnegative aligned targets')
        self.columns_ = list(X.columns)
        self.feature_ = FeatureComposition((29, 30, 31)).fit(X)
        frame = self.feature_.transform(X)
        self.categorical_ = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
        self.model_ = CatBoostRegressor(**self.parameters)
        started = time.monotonic()
        self.model_.fit(self._prepare(frame), np.log1p(target), cat_features=self.categorical_)
        self.audit_ = {'fit_rows': len(X), 'fit_original_rows': len(X), 'fit_retained_rows': len(X),
                       'removed_rows': 0, 'raw_columns': self.columns_.copy(),
                       'feature_columns': list(frame.columns), 'categorical_columns': self.categorical_.copy(),
                       'configuration': self.configuration(), 'base_used': False,
                       'feature_audit': deepcopy(self.feature_.audit_),
                       'assumption': 'conditional_Gaussian_log1p_y_not_established_fact',
                       'objective_is_original_rmse': False, 'elapsed_seconds': time.monotonic() - started}
        return self

    def predict(self, X):
        if not hasattr(self, 'model_'):
            raise ValueError('Fit ConditionalLognormalRegressor before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query DataFrame must match fitted raw columns in order')
        if not len(X):
            return np.empty(0, dtype=float)
        raw = self.model_.predict(self._prepare(self.feature_.transform(X)), prediction_type='RawFormulaVal')
        return conditional_lognormal_mean(raw, variance_cap=2.)
