"""H90 original-scale expert gated by observed missing energy-star rating."""
from copy import deepcopy
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from automl_reg_combo import FeatureComposition


def missing_rating_mask(X):
    """A finite numeric rating, including zero, is observed; other values missing."""
    if not isinstance(X, pd.DataFrame) or 'energy_star_rating' not in X:
        raise ValueError('Explicit energy_star_rating is required for the observable gate')
    return ~np.isfinite(pd.to_numeric(X['energy_star_rating'], errors='coerce').to_numpy(float))


class MissingRatingExpert:
    """One missing-rating subset expert; explicit supplied H70 is never refit.

    fit_hook(estimator, subset_frame, subset_y, configuration, **fit_kwargs)
    can register the actual CatBoost fit on its transformed subset separately
    from an outer wrapper claim. The hook must return the fitted estimator.
    """
    def __init__(self, threads=2, fit_hook=None):
        if (isinstance(threads, bool) or not isinstance(threads, (int, np.integer))
                or not 1 <= threads <= 4 or (fit_hook is not None and not callable(fit_hook))):
            raise ValueError('Need one to four CPU threads and an optional callable fit hook')
        self.threads, self.fit_hook = int(threads), fit_hook
        self.parameters = dict(iterations=800, depth=6, learning_rate=.05, l2_leaf_reg=20.,
            loss_function='RMSE', random_seed=42, thread_count=self.threads,
            verbose=False, allow_writing_files=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['fit_hook'] = None
        return state

    def configuration(self):
        return {'hypothesis': 'HYP-090', 'strategy': 'observable_missing_rating_expert', 'version': 1,
                'family': 'catboost', 'parameters': deepcopy(self.parameters),
                'feature_numbers': [29, 30, 31], 'feature_statistics_source': 'missing_rating_train_subset_only',
                'target_transform': 'none', 'loss': 'original_scale_squared_error',
                'gate': 'nonfinite_numeric_energy_star_rating_including_coercion_failure',
                'zero_rating': 'observed', 'minimum_training_missing_rows': 300,
                'expert_weight': .7, 'base_weight': .3,
                'known_rating_or_insufficient_support': 'supplied_H70_unchanged',
                'negative_expert_prediction': 'clip_zero_before_blend',
                'preprocessing': 'numeric_inf_to_nan_native_tagged_categories',
                'base_refit': False, 'query_statistics': False, 'target_outlier_deletion': False,
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
                or not all(isinstance(c, str) for c in X) or target.shape != (len(X),)
                or not np.isfinite(target).all() or np.any(target < 0)):
            raise ValueError('Expected nonempty unique-column frame and finite nonnegative aligned target')
        if base_predictor is None or not (hasattr(base_predictor, 'predict') or callable(base_predictor)):
            raise ValueError('Explicit compatible H70 baseline predictor is required')
        self.columns_, self.base_predictor_ = list(X.columns), base_predictor
        self.fit_mask_ = missing_rating_mask(X)
        self.fit_positions_ = np.flatnonzero(self.fit_mask_)
        self.model_, self.categorical_ = None, []
        started = time.monotonic()
        if len(self.fit_positions_) >= 300:
            subset = X.iloc[self.fit_positions_]
            self.feature_ = FeatureComposition((29, 30, 31)).fit(subset)
            frame = self.feature_.transform(subset)
            self.categorical_ = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
            frame = self._prepare(frame)
            estimator = CatBoostRegressor(**self.parameters)
            if self.fit_hook is None:
                self.model_ = estimator.fit(frame, target[self.fit_positions_], cat_features=self.categorical_)
            else:
                configuration = {**self.configuration(), 'stage': 'missing_rating_actual_learner',
                                 'training_subset_rows': len(subset), 'feature_columns': list(frame.columns)}
                self.model_ = self.fit_hook(estimator, frame, target[self.fit_positions_], configuration,
                                            cat_features=self.categorical_)
        self.audit_ = {'fit_rows': len(X), 'fit_original_rows': len(X), 'fit_retained_rows': len(X),
                       'expert_fit_rows': len(self.fit_positions_), 'expert_fit_count': int(self.model_ is not None),
                       'expert_fit_mask': self.fit_mask_.tolist(),
                       'expert_fit_positions': self.fit_positions_.tolist(),
                       'expert_fit_original_indices': X.index[self.fit_positions_].tolist(),
                       'removed_rows': 0, 'target_outlier_deletion': False,
                       'feature_audit': deepcopy(self.feature_.audit_) if self.model_ is not None else None,
                       'categorical_columns': self.categorical_.copy(), 'base_refit': False,
                       'base_source': 'caller_supplied_H70_provenance', 'configuration': self.configuration(),
                       'elapsed_seconds': time.monotonic() - started}
        return self

    def predict(self, X):
        if not hasattr(self, 'base_predictor_'):
            raise ValueError('Fit MissingRatingExpert before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query frame must match fitted raw columns in order')
        predictor = self.base_predictor_.predict if hasattr(self.base_predictor_, 'predict') else self.base_predictor_
        prediction = np.asarray(predictor(X), dtype=float).copy()
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Compatible finite H70 predictions required')
        positions = np.flatnonzero(missing_rating_mask(X))
        used = 0
        if self.model_ is not None and len(positions):
            frame = self._prepare(self.feature_.transform(X.iloc[positions]))
            expert = np.asarray(self.model_.predict(frame), dtype=float)
            if expert.shape != (len(positions),) or not np.isfinite(expert).all():
                raise ValueError('Finite aligned missing-rating expert predictions required')
            prediction[positions] = .7 * np.maximum(expert, 0.) + .3 * prediction[positions]
            used = len(positions)
        self.last_prediction_audit_ = {'query_rows': len(X), 'query_missing_rows': len(positions),
                                      'expert_applied_rows': used, 'known_rating_unchanged': True}
        return prediction
