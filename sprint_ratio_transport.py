"""H83 original-unit least-squares monotone temporal consumption ratios."""
from __future__ import annotations

import numpy as np

from sprint_regression import LINK_FIELDS, _keys, _rating_score, _ratings, _validate, _years


class OriginalLossTransport:
    """Within-profile ratio regression, weighted by the squared past target."""

    def __init__(self, alpha=.95, support_shrinkage=.25, age_decay=.25,
                 group_minimum_pairs=100, group_shrinkage=200., log_change_cap=1.5):
        self.alpha, self.support_shrinkage = float(alpha), float(support_shrinkage)
        self.age_decay, self.group_shrinkage = float(age_decay), float(group_shrinkage)
        self.group_minimum_pairs, self.log_change_cap = int(group_minimum_pairs), float(log_change_cap)
        values = [self.alpha, self.support_shrinkage, self.age_decay,
                  self.group_shrinkage, self.log_change_cap]
        if (not np.isfinite(values).all() or not 0 <= self.alpha <= 1
                or self.support_shrinkage <= 0 or self.group_shrinkage <= 0
                or self.age_decay < 0 or not 0 < self.log_change_cap < 700
                or self.group_minimum_pairs < 1):
            raise ValueError('Invalid original-loss transport configuration')

    def configuration(self):
        return {'strategy': 'original_loss_temporal_ratio', 'version': 1,
            'link_fields': list(LINK_FIELDS),
            'ambiguity': 'reject_profile_with_any_same_year_collision',
            'donor': 'latest_strictly_past', 'training_pairs': 'adjacent_strictly_past',
            'rating_score': 'inverse_normal_clipped_005_995',
            'model': 'increasing_isotonic_delta_rating_score_to_consumption_ratio',
            'target': 'current_y/donor_y_unclipped', 'sample_weight': 'donor_y_squared',
            'loss': 'exact_original_unit_squared_error', 'out_of_bounds': 'clip',
            'curve_anchor': None, 'missing_rating_or_no_curve': 'ratio_one',
            'group_column': 'facility_type', 'group_minimum_pairs': self.group_minimum_pairs,
            'group_shrinkage': self.group_shrinkage, 'log_change_cap': self.log_change_cap,
            'ratio_min': float(np.exp(-self.log_change_cap)),
            'ratio_max': float(np.exp(self.log_change_cap)),
            'alpha': self.alpha, 'support_shrinkage': self.support_shrinkage,
            'age_decay': self.age_decay}

    @staticmethod
    def ratio_targets(donor, current):
        donor, current = np.asarray(donor, float), np.asarray(current, float)
        if (donor.ndim != 1 or donor.shape != current.shape or not len(donor)
                or not np.isfinite(donor).all() or not np.isfinite(current).all()
                or np.any(donor <= 0) or np.any(current < 0)):
            raise ValueError('Positive donors and nonnegative aligned finite current targets required')
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            ratio, weight = current / donor, donor ** 2
        if not np.isfinite(ratio).all() or not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError('Ratio or original-scale weights outside finite numerical range')
        return ratio, weight

    def _fit_curve(self, delta, ratio, weight):
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(increasing=True, y_min=float(np.exp(-self.log_change_cap)),
            y_max=float(np.exp(self.log_change_cap)), out_of_bounds='clip').fit(
                delta, ratio, sample_weight=weight)

    def fit(self, X, y, base_predictor=None):
        y = _validate(X, y)
        if np.any(y < 0):
            raise ValueError('Nonnegative energy consumption required')
        years, ratings, keys = _years(X), _ratings(X), _keys(X)
        facilities = X['facility_type'].astype('string').fillna('__MISSING__').to_numpy()
        self.base_predictor_, self.fallback_ = base_predictor, float(np.mean(y))
        grouped = {}
        for position, key in enumerate(keys):
            grouped.setdefault(key, []).append(position)
        self.records_, self.training_pair_audit_ = {}, []
        rejected, donors, currents, deltas, groups = 0, [], [], [], []
        for key in sorted(grouped):
            positions = sorted(grouped[key], key=lambda i: years[i])
            if len({years[i] for i in positions}) != len(positions):
                rejected += 1
                continue
            self.records_[key] = [(float(years[i]), float(ratings[i]), float(y[i]), str(facilities[i]))
                                  for i in positions]
            for previous, current in zip(positions[:-1], positions[1:]):
                eligible = np.isfinite([ratings[previous], ratings[current]]).all() and y[previous] > 0
                self.training_pair_audit_.append({'donor_position': previous, 'current_position': current,
                    'donor_year': float(years[previous]), 'current_year': float(years[current]),
                    'eligible_rating_ratio': bool(eligible)})
                if eligible:
                    donors.append(y[previous]); currents.append(y[current])
                    deltas.append(float(_rating_score(ratings[current]) - _rating_score(ratings[previous])))
                    groups.append(str(facilities[current]))
        self.global_curve_, self.group_curves_, self.group_counts_ = None, {}, {}
        weight_sum = 0.
        if donors:
            delta, group = np.asarray(deltas), np.asarray(groups)
            ratio, weight = self.ratio_targets(donors, currents)
            self.global_curve_ = self._fit_curve(delta, ratio, weight)
            weight_sum = float(np.sum(weight))
            for facility in sorted(set(groups)):
                mask = group == facility
                count = int(mask.sum())
                self.group_counts_[facility] = count
                if count >= self.group_minimum_pairs:
                    self.group_curves_[facility] = self._fit_curve(delta[mask], ratio[mask], weight[mask])
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'profiles': len(grouped),
            'retained_profiles': len(self.records_), 'rejected_ambiguous_profiles': rejected,
            'adjacent_training_pairs': len(self.training_pair_audit_),
            'training_ratio_pairs': len(donors), 'original_scale_weight_sum': weight_sum,
            'global_curve_fitted': self.global_curve_ is not None,
            'group_pair_counts': dict(self.group_counts_), 'group_curves_fitted': sorted(self.group_curves_),
            'strict_past_only': True, 'matching_target_used': False, 'validation_target_used': False,
            'id_as_feature': False, 'row_deletion': False, 'base_refit': False,
            'old_h70_model_refit': False}
        return self

    def _ratio(self, delta, facility):
        if self.global_curve_ is None:
            return 1.
        ratio = float(self.global_curve_.predict([delta])[0])
        curve = self.group_curves_.get(facility)
        if curve is not None:
            count = self.group_counts_[facility]
            weight = count / (count + self.group_shrinkage)
            ratio = weight * float(curve.predict([delta])[0]) + (1. - weight) * ratio
        return ratio

    def predict(self, X):
        if self.base_predictor_ is None:
            prediction = np.full(len(X), self.fallback_, dtype=float)
        else:
            predictor = (self.base_predictor_.predict if hasattr(self.base_predictor_, 'predict')
                         else self.base_predictor_)
            prediction = np.asarray(predictor(X), float).copy()
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid compatible baseline prediction')
        linked, transported, donor_years = 0, 0, [None] * len(X)
        for i, (key, year, rating) in enumerate(zip(_keys(X), _years(X), _ratings(X))):
            past = [row for row in self.records_.get(key, ()) if row[0] < year]
            if not past:
                continue
            donor_year, donor_rating, donor_target, facility = past[-1]
            ratio = 1.
            if np.isfinite([rating, donor_rating]).all() and self.global_curve_ is not None:
                ratio = self._ratio(float(_rating_score(rating) - _rating_score(donor_rating)), facility)
                transported += 1
            donor = donor_target * ratio
            weight = self.alpha * len(past) / (len(past) + self.support_shrinkage)
            weight *= np.exp(-self.age_decay * max(0., year - donor_year - 1.))
            prediction[i] = weight * donor + (1. - weight) * prediction[i]
            linked += 1; donor_years[i] = donor_year
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite original-loss transport prediction')
        self.last_prediction_audit_ = {'query_rows': len(X), 'linked_rows': linked,
            'rating_transport_rows': transported, 'donor_years': donor_years,
            'query_rows_added_to_memory': False}
        return prediction
