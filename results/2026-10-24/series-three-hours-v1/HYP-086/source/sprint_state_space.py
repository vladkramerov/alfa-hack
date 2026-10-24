"""H86 causal local-level filtering of rating-normalized log consumption."""
from __future__ import annotations

import copy

import numpy as np

from sprint_regression import LINK_FIELDS, _keys, _rating_score, _ratings, _validate, _years


class RatingStateSpaceHistory:
    """Fixed-Q/R Gaussian state space; rating transport comes from frozen H70."""

    def __init__(self, process_variance=.01, measurement_variance=.04, alpha=.95,
                 support_shrinkage=.25, age_decay=.25, beta_cap=2.):
        self.process_variance, self.measurement_variance = float(process_variance), float(measurement_variance)
        self.alpha, self.support_shrinkage = float(alpha), float(support_shrinkage)
        self.age_decay, self.beta_cap = float(age_decay), float(beta_cap)
        values = [self.process_variance, self.measurement_variance, self.alpha,
                  self.support_shrinkage, self.age_decay, self.beta_cap]
        if (not np.isfinite(values).all() or min(self.process_variance, self.measurement_variance,
                                                self.support_shrinkage) <= 0
                or not 0 <= self.alpha <= 1 or min(self.age_decay, self.beta_cap) < 0):
            raise ValueError('Invalid state-space history configuration')

    def configuration(self):
        return {'strategy': 'rating_normalized_state_space_history', 'version': 1,
            'link_fields': list(LINK_FIELDS),
            'ambiguity': 'reject_profile_with_any_same_year_collision',
            'state': 'log_y_minus_frozen_facility_beta_times_rating_score',
            'transition': 'Gaussian_local_level_random_walk',
            'process_variance': self.process_variance, 'measurement_variance': self.measurement_variance,
            'variance_estimation': 'none_fixed_Q_R', 'initial_state': 'first_usable_z_P_equals_R',
            'observations': 'strictly_past_positive_y_finite_rating',
            'rating_score': 'inverse_normal_clipped_005_995',
            'transport': 'caller_frozen_h70_beta_no_refit', 'beta_cap': self.beta_cap,
            'query_mean': 'exp(mu+beta*q+.5*(P+Q*age+R))',
            'missing_query_rating': 'leave_compatible_H70_baseline_unchanged',
            'no_eligible_history': 'leave_baseline_unchanged',
            'alpha': self.alpha, 'support_shrinkage': self.support_shrinkage,
            'age_decay': self.age_decay, 'support': 'count_usable_filtered_observations'}

    def fit(self, X, y, base_predictor=None, transport_state=None):
        y = _validate(X, y)
        if np.any(y < 0):
            raise ValueError('Nonnegative consumption required')
        years, ratings, keys = _years(X), _ratings(X), _keys(X)
        facilities = X['facility_type'].astype('string').fillna('__MISSING__').to_numpy()
        state = copy.deepcopy(transport_state or {})
        beta = float(state.get('rating_beta', 0.))
        groups = {str(k): float(v) for k, v in state.get('group_rating_betas', {}).items()}
        if (not np.isfinite([beta, *groups.values()]).all() or beta < 0 or beta > self.beta_cap
                or any(value < 0 or value > self.beta_cap for value in groups.values())):
            raise ValueError('Finite bounded frozen transport beta required')
        self.transport_state_ = {'rating_beta': beta, 'group_rating_betas': groups}
        self.base_predictor_, self.fallback_ = base_predictor, float(np.mean(y))
        grouped = {}
        for position, key in enumerate(keys):
            grouped.setdefault(key, []).append(position)
        self.histories_, rejected, usable, without_history = {}, 0, 0, 0
        for key in sorted(grouped):
            positions = sorted(grouped[key], key=lambda position: years[position])
            if len({years[position] for position in positions}) != len(positions):
                rejected += 1
                continue
            history = []
            for position in positions:
                if y[position] <= 0 or not np.isfinite(ratings[position]):
                    continue
                facility = str(facilities[position])
                coefficient = groups.get(facility, beta)
                observation = float(np.log(y[position]) - coefficient * _rating_score(ratings[position]))
                year = float(years[position])
                if not history:
                    mean, variance, count = observation, self.measurement_variance, 1
                else:
                    previous_year, mean, variance, count, _ = history[-1]
                    predicted_variance = variance + self.process_variance * (year - previous_year)
                    gain = predicted_variance / (predicted_variance + self.measurement_variance)
                    mean += gain * (observation - mean)
                    variance = (1. - gain) * predicted_variance
                    count += 1
                history.append((year, float(mean), float(variance), count, facility))
                usable += 1
            if history:
                self.histories_[key] = history
            else:
                without_history += 1
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'profiles': len(grouped),
            'retained_filter_profiles': len(self.histories_), 'rejected_ambiguous_profiles': rejected,
            'profiles_without_eligible_observations': without_history,
            'usable_filter_observations': usable, 'rating_beta': beta,
            'group_rating_betas': dict(groups), 'transport_state_supplied': transport_state is not None,
            'transport_state_fitted_here': False, 'strict_past_only': True,
            'matching_target_used': False, 'validation_target_used': False,
            'row_deletion': False, 'id_as_feature': False, 'base_refit': False}
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
        filtered, missing, posterior_years = 0, 0, [None] * len(X)
        for i, (key, year, rating) in enumerate(zip(_keys(X), _years(X), _ratings(X))):
            if not np.isfinite(rating):
                missing += 1
                continue
            past = [posterior for posterior in self.histories_.get(key, ()) if posterior[0] < year]
            if not past:
                continue
            posterior_year, mean, variance, count, facility = past[-1]
            age = year - posterior_year
            coefficient = self.transport_state_['group_rating_betas'].get(
                facility, self.transport_state_['rating_beta'])
            predictive_variance = variance + self.process_variance * age + self.measurement_variance
            filtered_mean = float(np.exp(mean + coefficient * _rating_score(rating) + .5 * predictive_variance))
            weight = self.alpha * count / (count + self.support_shrinkage)
            weight *= np.exp(-self.age_decay * max(0., age - 1.))
            prediction[i] = weight * filtered_mean + (1. - weight) * prediction[i]
            filtered += 1; posterior_years[i] = posterior_year
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite state-space history prediction')
        self.last_prediction_audit_ = {'query_rows': len(X), 'filtered_rows': filtered,
            'missing_query_rating_rows': missing, 'posterior_years': posterior_years,
            'query_rows_added_to_memory': False}
        return prediction
