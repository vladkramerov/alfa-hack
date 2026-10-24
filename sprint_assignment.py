"""H99 cohort-dependent one-to-one past-year entity assignment."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sprint_regression import LINK_FIELDS, _keys, _rating_score, _ratings, _validate, _years


class GlobalEntityAssignment:
    """Fixed-cost joint matching, exact links reserved, no target-based matching."""

    def __init__(self, alpha=.35, age_decay=.25, max_assignment_cells=4000000):
        self.alpha, self.age_decay = float(alpha), float(age_decay)
        self.max_assignment_cells = int(max_assignment_cells)
        if (not np.isfinite([self.alpha, self.age_decay]).all() or not 0 <= self.alpha <= 1
                or self.age_decay < 0 or self.max_assignment_cells < 1):
            raise ValueError('Invalid global assignment configuration')

    def configuration(self):
        return {'strategy': 'global_temporal_entity_assignment', 'version': 1,
            'link_fields': list(LINK_FIELDS), 'exact_block': list(LINK_FIELDS[:3]),
            'donor_year': 'global_latest_training_year_strictly_before_query_year',
            'ambiguity': 'reject_training_sameyear_collision_and_query_duplicate_keys',
            'reserve_exact_H70_links': True, 'ambiguous_known_training_key': 'baseline_unchanged',
            'area_ratio_gate': [.95, 1.05], 'known_year_built_gap_cap': 20.,
            'cost': 'abs(log(areaQ/areaD))/.05+known_year_gap/20+known_elevation_gap/100',
            'unknown_or_zero_year_penalty': .5, 'unknown_elevation_penalty': .5,
            'maximum_eligible_cost': 1., 'unmatched_dummy_cost': 1.000001,
            'algorithm': 'linear_sum_assignment_with_one_dummy_per_query',
            'tie_order': 'canonical_physical_feature_key_no_ID_no_target',
            'max_assignment_cells': self.max_assignment_cells, 'cell_count': 'N*(M+N)_including_dummies',
            'oversized_block': 'skip_with_audit_no_alternate_algorithm',
            'transport': 'caller_frozen_H70_beta_no_refit',
            'rating_score': 'inverse_normal_clipped_005_995', 'beta_cap': 2., 'log_change_cap': 1.5,
            'alpha': self.alpha, 'support_factor': 'none', 'age_decay': self.age_decay,
            'query_cohort_dependent': True, 'batch_invariant': False,
            'physical_identity_proven': False, 'matching_uses_target': False}

    @staticmethod
    def _numeric(X, column):
        if column not in X:
            return np.full(len(X), np.nan)
        return pd.to_numeric(X[column], errors='coerce').to_numpy(float)

    def fit(self, X, y, base_predictor=None, transport_state=None):
        y = _validate(X, y)
        if np.any(y < 0):
            raise ValueError('Nonnegative consumption required')
        self.fit_years_, self.fit_rating_, self.fit_keys_ = _years(X), _ratings(X), _keys(X)
        self.fit_area_, self.fit_built_ = self._numeric(X, 'floor_area'), self._numeric(X, 'year_built')
        self.fit_elevation_, self.fit_target_ = self._numeric(X, 'ELEVATION'), y.copy()
        self.training_years_ = np.sort(np.unique(self.fit_years_))
        self.fit_blocks_ = [tuple(json.loads(key)[:3]) for key in self.fit_keys_]
        state = copy.deepcopy(transport_state or {})
        beta = float(state.get('rating_beta', 0.))
        group = {str(key): float(value) for key, value in state.get('group_rating_betas', {}).items()}
        if (not np.isfinite([beta, *group.values()]).all() or beta < 0 or beta > 2
                or any(value < 0 or value > 2 for value in group.values())):
            raise ValueError('Finite bounded frozen transport beta required')
        self.transport_state_ = {'rating_beta': beta, 'group_rating_betas': group}
        self.base_predictor_, self.fallback_ = base_predictor, float(np.mean(y))
        profiles = {}
        for i, key in enumerate(self.fit_keys_):
            profiles.setdefault(key, []).append(i)
        self.records_, self.rejected_keys_, self.donors_by_year_block_ = {}, set(), {}
        for key in sorted(profiles):
            rows = sorted(profiles[key], key=lambda i: self.fit_years_[i])
            if len({self.fit_years_[i] for i in rows}) != len(rows):
                self.rejected_keys_.add(key)
                continue
            self.records_[key] = rows
            for i in rows:
                index = (float(self.fit_years_[i]), self.fit_blocks_[i])
                self.donors_by_year_block_.setdefault(index, []).append(i)
        for rows in self.donors_by_year_block_.values():
            rows.sort(key=lambda i: self.fit_keys_[i])
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'profiles': len(profiles),
            'retained_profiles': len(self.records_), 'rejected_ambiguous_profiles': len(self.rejected_keys_),
            'training_years': self.training_years_.tolist(), 'rating_beta': beta,
            'group_rating_betas': dict(group), 'transport_state_supplied': transport_state is not None,
            'transport_state_fitted_here': False, 'validation_target_used': False,
            'matching_target_used': False, 'id_as_feature': False, 'row_deletion': False, 'base_refit': False}
        return self

    def _cost_matrix(self, X, query_positions, donor_positions):
        qa = self._numeric(X, 'floor_area')[query_positions, None]
        da = self.fit_area_[donor_positions][None, :]
        qb = self._numeric(X, 'year_built')[query_positions, None]
        db = self.fit_built_[donor_positions][None, :]
        qe = self._numeric(X, 'ELEVATION')[query_positions, None]
        de = self.fit_elevation_[donor_positions][None, :]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            ratio = qa / da
            area_cost = np.abs(np.log(ratio)) / .05
            known_year = np.isfinite(qb) & np.isfinite(db) & (qb > 0) & (db > 0)
            year_gap = np.abs(qb - db)
            year_cost = np.where(known_year, year_gap / 20., .5)
            known_elevation = np.isfinite(qe) & np.isfinite(de)
            elevation_cost = np.where(known_elevation, np.abs(qe - de) / 100., .5)
            cost = area_cost + year_cost + elevation_cost
        valid = (np.isfinite(qa) & np.isfinite(da) & (qa > 0) & (da > 0)
                 & (ratio >= .95) & (ratio <= 1.05) & (~known_year | (year_gap <= 20.))
                 & np.isfinite(cost) & (cost <= 1.))
        return np.where(valid, cost, np.inf)

    def predict(self, X):
        if self.base_predictor_ is None:
            prediction = np.full(len(X), self.fallback_, dtype=float)
        else:
            predictor = (self.base_predictor_.predict if hasattr(self.base_predictor_, 'predict')
                         else self.base_predictor_)
            prediction = np.asarray(predictor(X), float).copy()
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid compatible baseline prediction')
        keys, years, rating = _keys(X), _years(X), _ratings(X)
        blocks = [tuple(json.loads(key)[:3]) for key in keys]
        donor_positions, donor_years, costs = [None]*len(X), [None]*len(X), [None]*len(X)
        assigned = reserved_total = duplicate_total = exact_total = rejected_total = 0
        skipped = []
        for year in np.sort(np.unique(years)):
            queries = np.flatnonzero(years == year).tolist()
            past_years = self.training_years_[self.training_years_ < year]
            if not len(past_years):
                continue
            donor_year = float(past_years[-1])
            exact_keys = {key for key in {keys[i] for i in queries} if key in self.records_
                          and any(self.fit_years_[j] < year for j in self.records_[key])}
            reserved_total += len(exact_keys)
            counts = {}
            for i in queries:
                counts[keys[i]] = counts.get(keys[i], 0) + 1
            candidate_blocks = {}
            for i in queries:
                if keys[i] in exact_keys:
                    exact_total += 1
                elif keys[i] in self.rejected_keys_:
                    rejected_total += 1
                elif counts[keys[i]] > 1:
                    duplicate_total += 1
                else:
                    candidate_blocks.setdefault(blocks[i], []).append(i)
            for block in sorted(candidate_blocks, key=lambda value: json.dumps(value, default=str)):
                rows = sorted(candidate_blocks[block], key=lambda i: keys[i])
                donors = [j for j in self.donors_by_year_block_.get((donor_year, block), ())
                          if self.fit_keys_[j] not in exact_keys]
                if not donors:
                    continue
                n, m = len(rows), len(donors)
                cells = n*(m+n)
                if cells > self.max_assignment_cells:
                    skipped.append({'query_year': float(year), 'block': list(block), 'query_rows': n,
                                    'donor_rows': m, 'cells': cells, 'reason': 'matrix_cell_cap'})
                    continue
                cost = self._cost_matrix(X, rows, donors)
                matrix = np.full((n, m+n), 1.000001)
                matrix[:, :m] = cost
                selected_rows, selected_columns = linear_sum_assignment(matrix)
                for row, column in zip(selected_rows, selected_columns):
                    if column >= m:
                        continue
                    i, j = rows[row], donors[column]
                    donor = self.fit_target_[j]
                    adjustment = 0.
                    if np.isfinite([rating[i], self.fit_rating_[j]]).all():
                        coefficient = self.transport_state_['group_rating_betas'].get(
                            str(block[2]), self.transport_state_['rating_beta'])
                        adjustment = float(np.clip(coefficient * (_rating_score(rating[i])
                                                   - _rating_score(self.fit_rating_[j])), -1.5, 1.5))
                    weight = self.alpha*np.exp(-self.age_decay*max(0., float(year)-donor_year-1.))
                    prediction[i] = weight*donor*np.exp(adjustment) + (1.-weight)*prediction[i]
                    donor_positions[i], donor_years[i], costs[i] = int(j), donor_year, float(cost[row, column])
                    assigned += 1
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite global assignment prediction')
        self.last_prediction_audit_ = {'query_rows': len(X), 'assigned_rows': assigned,
            'reserved_donor_profiles': reserved_total, 'exact_query_rows_unchanged': exact_total,
            'duplicate_query_rows': duplicate_total, 'ambiguous_training_key_rows_unchanged': rejected_total,
            'donor_positions': donor_positions, 'donor_years': donor_years, 'assignment_costs': costs,
            'skipped_blocks': skipped, 'query_cohort_dependent': True,
            'matching_target_used': False, 'query_rows_added_to_memory': False}
        return prediction
