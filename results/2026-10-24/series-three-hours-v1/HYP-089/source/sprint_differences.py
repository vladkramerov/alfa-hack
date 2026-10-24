"""H89 within-entity differences, original-unit Ridge temporal transport."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_regression import LINK_FIELDS, _keys, _rating_score, _ratings, _validate, _years


class PanelDifferenceRegressor:
    """Learn changes from strict-past entity pairs, never baseline residuals."""

    def __init__(self, ridge_alpha=100., alpha=.95, minimum_pairs=32,
                 support_shrinkage=.25, age_decay=.25, log_change_cap=1.5, threads=4):
        self.ridge_alpha, self.alpha = float(ridge_alpha), float(alpha)
        self.minimum_pairs, self.threads = int(minimum_pairs), min(4, max(1, int(threads)))
        self.support_shrinkage, self.age_decay = float(support_shrinkage), float(age_decay)
        self.log_change_cap = float(log_change_cap)
        values = [self.ridge_alpha, self.alpha, self.support_shrinkage, self.age_decay, self.log_change_cap]
        if (not np.isfinite(values).all() or self.ridge_alpha <= 0 or not 0 <= self.alpha <= 1
                or self.minimum_pairs < 2 or self.support_shrinkage <= 0 or self.age_decay < 0
                or not 0 < self.log_change_cap < 700):
            raise ValueError('Invalid panel-difference configuration')

    def configuration(self):
        return {'strategy': 'within_entity_difference_ridge', 'version': 1,
            'link_fields': list(LINK_FIELDS), 'ambiguity': 'reject_profile_with_any_same_year_collision',
            'training_pairs': 'adjacent_strictly_past', 'donor': 'latest_strictly_past',
            'target': 'current_y_minus_donor_y_original_units', 'loss': 'original_unit_squared_error',
            'model': 'Ridge', 'ridge_alpha': self.ridge_alpha, 'solver': 'lsqr', 'tol': 1e-6,
            'fit_intercept': True, 'numeric_preprocessing': 'fit_only_median_then_StandardScaler',
            'categorical_preprocessing': 'fit_only_facility_OneHotEncoder_unknown_ignore',
            'features': ['donor_y', 'age', 'donor_y*delta_inverse_normal_rating_score',
                         'donor_rating_missing', 'current_rating_missing', 'numeric_weather_deltas', 'facility_type'],
            'weather_columns': 'numeric_X_except_link_fields_year_rating_id_target_label',
            'missing_rating_interaction': 0., 'current_absolute_X': False,
            'prediction': 'clip(donor_y+predicted_delta,donor_y*exp(-cap),donor_y*exp(cap))',
            'log_change_cap': self.log_change_cap, 'minimum_pairs': self.minimum_pairs,
            'alpha': self.alpha, 'support_shrinkage': self.support_shrinkage,
            'age_decay': self.age_decay, 'threads': self.threads}

    def _pair_features(self, X, current_positions, donor_positions, donor_targets):
        current, donor = X.iloc[current_positions], self.fit_frame_.iloc[donor_positions]
        previous_rating, current_rating = _ratings(donor), _ratings(current)
        known = np.isfinite(previous_rating) & np.isfinite(current_rating)
        interaction = np.zeros(len(current))
        donor_targets = np.asarray(donor_targets, float)
        interaction[known] = donor_targets[known] * (_rating_score(current_rating[known])
                                                   - _rating_score(previous_rating[known]))
        result = pd.DataFrame({'donor_y': donor_targets, 'age': _years(current)-_years(donor),
            'donor_y_delta_rating_score': interaction,
            'donor_rating_missing': (~np.isfinite(previous_rating)).astype(float),
            'current_rating_missing': (~np.isfinite(current_rating)).astype(float),
            'facility_type': current['facility_type'].astype('string').fillna('__MISSING__').astype(str).to_numpy()})
        for column in self.weather_columns_:
            result['delta_' + column] = (pd.to_numeric(current[column], errors='coerce').to_numpy(float)
                                        - pd.to_numeric(donor[column], errors='coerce').to_numpy(float))
        return result.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y, base_predictor=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from threadpoolctl import threadpool_limits

        y = _validate(X, y)
        if np.any(y < 0):
            raise ValueError('Nonnegative energy consumption required')
        years, keys = _years(X), _keys(X)
        excluded = {*LINK_FIELDS, 'Year_Factor', 'energy_star_rating'}
        self.weather_columns_ = [column for column in X if column not in excluded
            and str(column).lower() not in {'id', 'target', 'site_eui', 'label'}
            and pd.api.types.is_numeric_dtype(X[column].dtype)]
        self.fit_frame_ = X.drop(columns=[column for column in X
            if str(column).lower() in {'id', 'target', 'site_eui', 'label'}]).reset_index(drop=True).copy()
        self.fit_target_, self.fit_years_ = y.copy(), years.copy()
        self.base_predictor_, self.fallback_ = base_predictor, float(np.mean(y))
        grouped = {}
        for position, key in enumerate(keys):
            grouped.setdefault(key, []).append(position)
        self.records_, self.training_pair_audit_ = {}, []
        donors, currents, rejected = [], [], 0
        for key in sorted(grouped):
            ordered = sorted(grouped[key], key=lambda position: years[position])
            if len({years[position] for position in ordered}) != len(ordered):
                rejected += 1
                continue
            self.records_[key] = ordered
            for previous, current in zip(ordered[:-1], ordered[1:]):
                donors.append(previous); currents.append(current)
                self.training_pair_audit_.append({'donor_position': previous, 'current_position': current,
                    'donor_year': float(years[previous]), 'current_year': float(years[current])})
        self.model_, self.preprocessor_ = None, None
        if len(currents) >= self.minimum_pairs:
            features = self._pair_features(X, currents, donors, y[donors])
            numeric = [column for column in features if column != 'facility_type']
            numeric_pipeline = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                         ('scale', StandardScaler())])
            self.preprocessor_ = ColumnTransformer([
                ('numeric', numeric_pipeline, numeric),
                ('facility', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['facility_type'])],
                remainder='drop', sparse_threshold=0.)
            with threadpool_limits(limits=self.threads):
                values = self.preprocessor_.fit_transform(features)
                self.model_ = Ridge(alpha=self.ridge_alpha, fit_intercept=True, solver='lsqr', tol=1e-6)
                self.model_.fit(values, y[currents]-y[donors])
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'profiles': len(grouped),
            'retained_profiles': len(self.records_), 'rejected_ambiguous_profiles': rejected,
            'training_pairs': len(currents), 'delta_model_fitted': self.model_ is not None,
            'weather_columns_fitted': list(self.weather_columns_), 'strict_past_only': True,
            'baseline_residual_target': False, 'validation_target_used': False,
            'id_as_feature': False, 'row_deletion': False, 'base_refit': False}
        return self

    def predict(self, X):
        from threadpoolctl import threadpool_limits

        if self.base_predictor_ is None:
            prediction = np.full(len(X), self.fallback_, dtype=float)
        else:
            predictor = (self.base_predictor_.predict if hasattr(self.base_predictor_, 'predict')
                         else self.base_predictor_)
            prediction = np.asarray(predictor(X), float).copy()
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid compatible baseline prediction')
        currents, donors, supports, donor_years = [], [], [], [None]*len(X)
        years = _years(X)
        for current, (key, year) in enumerate(zip(_keys(X), years)):
            past = [position for position in self.records_.get(key, ()) if self.fit_years_[position] < year]
            if past:
                currents.append(current); donors.append(past[-1]); supports.append(len(past))
                donor_years[current] = float(self.fit_years_[past[-1]])
        used = 0
        if currents and self.model_ is not None:
            target = self.fit_target_[donors]
            frame = self._pair_features(X, currents, donors, target)
            with threadpool_limits(limits=self.threads):
                change = self.model_.predict(self.preprocessor_.transform(frame))
            transported = np.clip(target + change, target*np.exp(-self.log_change_cap),
                                  target*np.exp(self.log_change_cap))
            age = years[currents] - self.fit_years_[donors]
            support = np.asarray(supports, float)
            weight = self.alpha*support/(support+self.support_shrinkage)
            weight *= np.exp(-self.age_decay*np.maximum(0., age-1.))
            prediction[currents] = weight*transported + (1.-weight)*prediction[currents]
            used = len(currents)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite panel-difference prediction')
        self.last_prediction_audit_ = {'query_rows': len(X), 'linked_rows': len(currents),
            'delta_model_rows': used, 'donor_years': donor_years, 'query_rows_added_to_memory': False}
        return prediction
