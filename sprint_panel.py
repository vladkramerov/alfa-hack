"""H76: supervised transitions between unambiguous, strictly past panel rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_regression import LINK_FIELDS, _keys, _ratings, _validate, _years


class PanelTransitionRegressor:
    """Predict original-unit consumption conditional on a known past observation.

    Only adjacent, strictly increasing training years form supervised pairs.
    Query rows never become donors, and the supplied baseline is never refit.
    """

    def __init__(self, parameters=None, alpha=.9, minimum_pairs=32, threads=4):
        self.parameters = dict(parameters or {})
        self.alpha, self.minimum_pairs = float(alpha), int(minimum_pairs)
        self.threads = min(4, max(1, int(threads)))
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1 or self.minimum_pairs < 2:
            raise ValueError('Invalid panel blending or minimum pair count')

    def configuration(self):
        parameters = {'iterations': 800, 'depth': 6, 'learning_rate': .05, 'l2_leaf_reg': 10.}
        parameters.update(self.parameters)
        parameters.update(loss_function='RMSE', random_seed=42, thread_count=self.threads,
                          verbose=False, allow_writing_files=False)
        return {'strategy': 'conditional_panel_transition', 'version': 1,
                'family': 'catboost', 'parameters': parameters, 'alpha': self.alpha,
                'minimum_pairs': self.minimum_pairs, 'target': 'current_original_units',
                'loss': 'original_scale_squared_error', 'link_fields': list(LINK_FIELDS),
                'ambiguity': 'reject_profile_with_any_same_year_collision',
                'training_pairs': 'adjacent_strictly_past', 'query_donor': 'latest_strictly_past',
                'current_features': 'raw_X_without_id_or_target',
                'transition_features': ['donor_target', 'donor_rating', 'rating_delta',
                                        'donor_rating_missing', 'current_rating_missing', 'age'],
                'weather_columns': 'numeric_except_link_fields_year_rating',
                'weather_features': 'donor_value_and_current_minus_donor',
                'matched_blend': 'alpha*transition+(1-alpha)*baseline',
                'unmatched_or_insufficient_pairs': 'baseline', 'negative_prediction': 'clip_zero'}

    @staticmethod
    def _excluded(column):
        return str(column).lower() in {'id', 'target', 'site_eui', 'label'}

    def _pair_frame(self, X, current_positions, donor_positions, donor_targets):
        current = X.iloc[current_positions]
        donor = self.fit_frame_.iloc[donor_positions]
        out = current.loc[:, self.feature_columns_].reset_index(drop=True).copy()
        current_rating, donor_rating = _ratings(current), _ratings(donor)
        out['panel_donor_target'] = np.asarray(donor_targets, float)
        out['panel_donor_rating'] = donor_rating
        out['panel_rating_delta'] = current_rating - donor_rating
        out['panel_donor_rating_missing'] = ~np.isfinite(donor_rating)
        out['panel_current_rating_missing'] = ~np.isfinite(current_rating)
        out['panel_age'] = _years(current) - _years(donor)
        for column in self.weather_columns_:
            previous = pd.to_numeric(donor[column], errors='coerce').to_numpy(float)
            present = pd.to_numeric(current[column], errors='coerce').to_numpy(float)
            out['panel_donor_' + column] = previous
            out['panel_delta_' + column] = present - previous
        return out

    def _prepare(self, frame):
        frame = frame.copy()
        for column in frame:
            if column in self.categorical_columns_:
                frame[column] = frame[column].astype('string').fillna('__MISSING__').astype(str)
            else:
                frame[column] = pd.to_numeric(frame[column], errors='coerce').astype(float)
                frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
        return frame

    def fit(self, X, y, base_predictor=None):
        from catboost import CatBoostRegressor

        y = _validate(X, y)
        if np.any(y < 0):
            raise ValueError('Nonnegative energy consumption required')
        years = _years(X)
        self.feature_columns_ = [column for column in X if not self._excluded(column)]
        self.fit_frame_ = X.loc[:, self.feature_columns_].reset_index(drop=True).copy()
        self.fit_target_ = y.copy()
        self.weather_columns_ = [column for column in self.feature_columns_
                                if column not in (*LINK_FIELDS, 'Year_Factor', 'energy_star_rating')
                                and pd.api.types.is_numeric_dtype(X[column].dtype)]
        self.base_predictor_, self.fallback_ = base_predictor, float(np.mean(y))
        groups = {}
        for position, key in enumerate(_keys(X)):
            groups.setdefault(key, []).append(position)
        self.records_, rejected, previous_positions, current_positions = {}, 0, [], []
        self.training_pair_audit_ = []
        for key in sorted(groups):
            ordered = sorted(groups[key], key=lambda position: years[position])
            if len({years[position] for position in ordered}) != len(ordered):
                rejected += 1
                continue
            self.records_[key] = ordered
            for previous, current in zip(ordered[:-1], ordered[1:]):
                previous_positions.append(previous)
                current_positions.append(current)
                self.training_pair_audit_.append({'donor_position': previous, 'current_position': current,
                    'donor_year': float(years[previous]), 'current_year': float(years[current])})
        self.fit_years_ = years.copy()
        self.model_ = None
        self.categorical_columns_ = []
        pair_count = len(current_positions)
        if pair_count >= self.minimum_pairs and np.ptp(y[current_positions]) > 0:
            frame = self._pair_frame(X, current_positions, previous_positions, y[previous_positions])
            self.categorical_columns_ = [column for column in frame
                                        if not pd.api.types.is_numeric_dtype(frame[column].dtype)]
            self.model_ = CatBoostRegressor(**self.configuration()['parameters'])
            self.model_.fit(self._prepare(frame), y[current_positions],
                            cat_features=self.categorical_columns_)
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'profiles': len(groups),
            'retained_profiles': len(self.records_), 'rejected_ambiguous_profiles': rejected,
            'training_pairs': pair_count, 'transition_fitted': self.model_ is not None,
            'pair_year_min': float(min(years[current_positions])) if pair_count else None,
            'pair_year_max': float(max(years[current_positions])) if pair_count else None,
            'weather_columns_fitted': list(self.weather_columns_),
            'strict_past_only': True, 'validation_target_used': False,
            'id_as_feature': False, 'row_deletion': False, 'base_refit': False}
        return self

    def predict(self, X):
        if self.base_predictor_ is None:
            prediction = np.full(len(X), self.fallback_, dtype=float)
        else:
            predictor = (self.base_predictor_.predict if hasattr(self.base_predictor_, 'predict')
                         else self.base_predictor_)
            prediction = np.asarray(predictor(X), float).copy()
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid compatible baseline prediction')
        current_positions, donor_positions, donor_years = [], [], [None] * len(X)
        for current, (key, year) in enumerate(zip(_keys(X), _years(X))):
            past = [position for position in self.records_.get(key, ())
                    if self.fit_years_[position] < year]
            if past:
                donor = past[-1]
                current_positions.append(current); donor_positions.append(donor)
                donor_years[current] = float(self.fit_years_[donor])
        used = 0
        if self.model_ is not None and current_positions:
            frame = self._pair_frame(X, current_positions, donor_positions, self.fit_target_[donor_positions])
            transition = np.asarray(self.model_.predict(self._prepare(frame)), float)
            if not np.isfinite(transition).all():
                raise ValueError('Nonfinite conditional transition prediction')
            transition = np.maximum(transition, 0.)
            positions = np.asarray(current_positions)
            prediction[positions] = self.alpha * transition + (1. - self.alpha) * prediction[positions]
            used = len(positions)
        self.last_prediction_audit_ = {'query_rows': len(X), 'linked_rows': len(current_positions),
                                      'transition_rows': used, 'donor_years': donor_years,
                                      'query_rows_added_to_memory': False}
        return prediction
