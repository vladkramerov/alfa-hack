"""H81 conservative fuzzy linkage; H70 predictions/state are supplied, never fit."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from sprint_regression import LINK_FIELDS, _keys, _rating_score, _ratings, _validate, _years


class FuzzyTemporalOverlay:
    """Recover only unmatched physical profiles with a unique nearby past donor."""

    def __init__(self, alpha=.5, area_tolerance=.001, year_tolerance=1., ambiguity_margin=.5,
                 support_shrinkage=.25, age_decay=.25, beta_cap=2., log_change_cap=1.5):
        self.alpha, self.area_tolerance = float(alpha), float(area_tolerance)
        self.year_tolerance, self.ambiguity_margin = float(year_tolerance), float(ambiguity_margin)
        self.support_shrinkage, self.age_decay = float(support_shrinkage), float(age_decay)
        self.beta_cap, self.log_change_cap = float(beta_cap), float(log_change_cap)
        values = [self.alpha, self.area_tolerance, self.year_tolerance, self.ambiguity_margin,
                  self.support_shrinkage, self.age_decay, self.beta_cap, self.log_change_cap]
        if (not np.isfinite(values).all() or not 0 <= self.alpha <= 1
                or not 0 < self.area_tolerance < 1 or self.year_tolerance <= 0
                or self.ambiguity_margin <= 0 or self.support_shrinkage <= 0
                or min(self.age_decay, self.beta_cap, self.log_change_cap) < 0):
            raise ValueError('Invalid fuzzy temporal configuration')

    def configuration(self):
        return {'strategy': 'fuzzy_temporal_overlay', 'version': 1, 'alpha': self.alpha,
            'link_fields': list(LINK_FIELDS), 'exact_block': list(LINK_FIELDS[:3]),
            'area_tolerance': self.area_tolerance, 'area_distance': 'absolute_gap_over_max_area',
            'year_tolerance': self.year_tolerance, 'missing_year': 'zero_or_nan_exact_unique_area_only',
            'distance': 'relative_area_gap/area_tolerance+absolute_year_gap/year_tolerance',
            'ambiguity_margin': self.ambiguity_margin, 'missing_year_distance': 1.,
            'ambiguity': 'reject_any_sameyear_collision_profile_then_distinct_profile_runnerup',
            'donor': 'latest_strictly_past', 'exact_h70_profile': 'leave_baseline_unchanged',
            'support_shrinkage': self.support_shrinkage, 'age_decay': self.age_decay,
            'beta_cap': self.beta_cap, 'log_change_cap': self.log_change_cap,
            'transport': 'caller_frozen_h70_beta_and_group_betas_no_refit',
            'rating_score': 'inverse_normal_clipped_005_995', 'id_as_feature': False}

    def fit(self, X, y, base_predictor=None, transport_state=None):
        y = _validate(X, y)
        if np.any(y < 0):
            raise ValueError('Nonnegative consumption required')
        years, ratings, keys = _years(X), _ratings(X), _keys(X)
        areas = pd.to_numeric(X['floor_area'], errors='coerce').to_numpy(float)
        built = pd.to_numeric(X['year_built'], errors='coerce').to_numpy(float)
        self.base_predictor_, self.fallback_ = base_predictor, float(np.mean(y))
        state = copy.deepcopy(transport_state or {})
        beta = float(state.get('rating_beta', 0.))
        group = {str(k): float(v) for k, v in state.get('group_rating_betas', {}).items()}
        if not np.isfinite([beta, *group.values()]).all():
            raise ValueError('Finite frozen rating transport state required')
        if beta < 0 or beta > self.beta_cap or any(v < 0 or v > self.beta_cap for v in group.values()):
            raise ValueError('Frozen beta exceeds fixed transport cap')
        self.transport_state_ = {'rating_beta': beta, 'group_rating_betas': group}
        grouped = {}
        for i, key in enumerate(keys):
            grouped.setdefault(key, []).append(i)
        self.known_keys_, self.records_, blocks = set(grouped), {}, {}
        rejected = invalid_area = 0
        for key in sorted(grouped):
            positions = sorted(grouped[key], key=lambda i: years[i])
            if len({years[i] for i in positions}) != len(positions):
                rejected += 1
                continue
            i = positions[0]
            if not np.isfinite(areas[i]) or areas[i] <= 0:
                invalid_area += 1
                continue
            block = tuple(json.loads(key)[:3])
            self.records_[key] = [(float(years[j]), float(ratings[j]), float(y[j])) for j in positions]
            blocks.setdefault(block, []).append((float(areas[i]), float(built[i]), key))
        self.blocks_ = {}
        for block, rows in blocks.items():
            rows.sort(key=lambda row: (row[0], row[2]))
            self.blocks_[block] = {'area': np.asarray([row[0] for row in rows]),
                                   'rows': rows}
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'profiles': len(grouped),
            'retained_profiles': len(self.records_), 'rejected_ambiguous_profiles': rejected,
            'rejected_invalid_area_profiles': invalid_area, 'strict_past_only': True,
            'validation_target_used': False, 'matching_target_used': False, 'row_deletion': False,
            'base_refit': False, 'transport_state_fitted_here': False,
            'transport_state_supplied': transport_state is not None,
            'rating_beta': beta, 'group_rating_betas': dict(group)}
        return self

    def _match(self, key, year, area, built):
        if key in self.known_keys_:
            return None, 'exact'
        if not np.isfinite(area) or area <= 0:
            return None, 'invalid_area'
        block = tuple(json.loads(key)[:3])
        index = self.blocks_.get(block)
        if index is None:
            return None, 'unmatched'
        lo = np.searchsorted(index['area'], area * (1. - self.area_tolerance), side='left')
        hi = np.searchsorted(index['area'], area / (1. - self.area_tolerance), side='right')
        candidates = []
        for donor_area, donor_built, profile in index['rows'][lo:hi]:
            past = [row for row in self.records_[profile] if row[0] < year]
            if not past:
                continue
            gap = abs(donor_area - area) / max(donor_area, area)
            known = np.isfinite([built, donor_built]).all() and min(built, donor_built) > 0
            if known:
                year_gap = abs(built - donor_built)
                if year_gap > self.year_tolerance:
                    continue
                distance = gap / self.area_tolerance + year_gap / self.year_tolerance
            else:
                if donor_area != area:
                    continue
                distance = 1.
            if gap <= self.area_tolerance:
                candidates.append((distance, profile, past[-1], len(past)))
        if not candidates:
            return None, 'unmatched'
        candidates.sort(key=lambda row: (row[0], row[1]))
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < self.ambiguity_margin:
            return None, 'margin'
        return (candidates[0][2], candidates[0][3], block[2]), 'matched'

    def predict(self, X):
        if self.base_predictor_ is None:
            prediction = np.full(len(X), self.fallback_)
        else:
            predictor = (self.base_predictor_.predict if hasattr(self.base_predictor_, 'predict')
                         else self.base_predictor_)
            prediction = np.asarray(predictor(X), float).copy()
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid compatible baseline prediction')
        areas = pd.to_numeric(X['floor_area'], errors='coerce').to_numpy(float)
        built = pd.to_numeric(X['year_built'], errors='coerce').to_numpy(float)
        linked, exact, margin, donor_years = 0, 0, 0, [None] * len(X)
        for i, (key, year, rating) in enumerate(zip(_keys(X), _years(X), _ratings(X))):
            match, reason = self._match(key, year, areas[i], built[i])
            exact += reason == 'exact'; margin += reason == 'margin'
            if match is None:
                continue
            (donor_year, donor_rating, donor_target), support, facility = match
            adjustment = 0.
            if np.isfinite([rating, donor_rating]).all():
                beta = self.transport_state_['group_rating_betas'].get(facility,
                                                           self.transport_state_['rating_beta'])
                adjustment = float(np.clip(beta * (_rating_score(rating) - _rating_score(donor_rating)),
                                            -self.log_change_cap, self.log_change_cap))
            transported = donor_target * np.exp(adjustment)
            weight = self.alpha * support / (support + self.support_shrinkage)
            weight *= np.exp(-self.age_decay * max(0., year - donor_year - 1.))
            prediction[i] = weight * transported + (1. - weight) * prediction[i]
            linked += 1; donor_years[i] = donor_year
        self.last_prediction_audit_ = {'query_rows': len(X), 'fuzzy_linked_rows': linked,
            'unchanged_exact_profiles': exact, 'rejected_margin': margin, 'donor_years': donor_years,
            'query_rows_added_to_memory': False}
        return prediction
