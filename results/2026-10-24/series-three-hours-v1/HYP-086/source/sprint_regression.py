"""SPEC011 regression candidates; all learned state is fit-only, no API or I/O."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.special import ndtri


LINK_FIELDS = ('State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built')


def _validate(X, y):
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected nonempty finite aligned targets')
    return y


def _keys(X):
    missing = set(LINK_FIELDS) - set(X.columns)
    if missing:
        raise ValueError('Missing linkage fields: ' + str(sorted(missing)))
    values = []
    for column in LINK_FIELDS:
        if column in ('floor_area', 'year_built'):
            v = pd.to_numeric(X[column], errors='coerce')
            values.append(v.map(lambda x: None if pd.isna(x) else format(float(x), '.17g')).tolist())
        else:
            values.append(X[column].astype('string').where(X[column].notna(), None).tolist())
    return [json.dumps(row, ensure_ascii=False, default=str) for row in zip(*values)]


def _years(X):
    value = pd.to_numeric(X['Year_Factor'], errors='raise').to_numpy(float)
    if not np.isfinite(value).all():
        raise ValueError('Finite explicit temporal year required')
    return value


def _ratings(X):
    return pd.to_numeric(X['energy_star_rating'], errors='coerce').to_numpy(float)


def _rating_score(rating):
    return ndtri(np.clip((100.5 - np.asarray(rating, float)) / 100., .005, .995))


class TemporalEntityLinker:
    """Latest unique past donor, with training-pair rating transport and fallback."""
    def __init__(self, alpha=.8, pair_ridge=20., group_shrinkage=100.,
                 support_shrinkage=.25, age_decay=.25, beta_cap=2., log_change_cap=1.5):
        self.alpha = float(alpha)
        self.pair_ridge, self.group_shrinkage = float(pair_ridge), float(group_shrinkage)
        self.support_shrinkage, self.age_decay = float(support_shrinkage), float(age_decay)
        self.beta_cap, self.log_change_cap = float(beta_cap), float(log_change_cap)
        if not 0 <= self.alpha <= 1 or min(self.pair_ridge, self.group_shrinkage, self.support_shrinkage) <= 0:
            raise ValueError('Invalid linkage confidence parameters')

    def configuration(self):
        return {'strategy': 'temporal_entity_linking', 'version': 1, 'alpha': self.alpha,
            'link_fields': list(LINK_FIELDS), 'ambiguity': 'reject_profile_with_any_same_year_collision',
            'donor': 'latest_strictly_past', 'rating_score': 'inverse_normal_clipped_005_995',
            'pair_ridge': self.pair_ridge, 'group_shrinkage': self.group_shrinkage,
            'support_shrinkage': self.support_shrinkage, 'age_decay': self.age_decay,
            'beta_cap': self.beta_cap, 'log_change_cap': self.log_change_cap}

    def fit(self, X, y, base_predictor=None):
        y = _validate(X, y)
        if np.any(y < 0):
            raise ValueError('Rating transport requires nonnegative consumption')
        years, ratings = _years(X), _ratings(X)
        self.base_predictor_ = base_predictor
        self.fallback_ = float(np.mean(y))
        records = {}
        facilities = X['facility_type'].astype('string').fillna('__MISSING__').to_numpy()
        for key, year, rating, target, facility in zip(_keys(X), years, ratings, y, facilities):
            records.setdefault(key, []).append((year, rating, target, str(facility)))
        self.records_, rejected, pairs = {}, 0, []
        for key, rows in records.items():
            ordered = sorted(rows, key=lambda r: r[0])
            if len({row[0] for row in ordered}) != len(ordered):
                rejected += 1
                continue
            self.records_[key] = ordered
            for previous, current in zip(ordered[:-1], ordered[1:]):
                if np.isfinite([previous[1], current[1]]).all() and min(previous[2], current[2]) > 0:
                    delta = float(_rating_score(current[1]) - _rating_score(previous[1]))
                    response = float(np.clip(np.log(current[2] / previous[2]), -self.log_change_cap, self.log_change_cap))
                    pairs.append((delta, response, current[3]))
        xx = sum(a * a for a, b, c in pairs)
        xy = sum(a * b for a, b, c in pairs)
        self.beta_ = float(np.clip(xy / (xx + self.pair_ridge), 0., self.beta_cap))
        group_sums = {}
        for delta, response, group in pairs:
            sums = group_sums.setdefault(group, [0., 0., 0])
            sums[0] += delta * delta; sums[1] += delta * response; sums[2] += 1
        self.group_beta_ = {}
        for group, (gx, gy, count) in group_sums.items():
            local = float(np.clip(gy / (gx + self.pair_ridge), 0., self.beta_cap))
            weight = count / (count + self.group_shrinkage)
            self.group_beta_[group] = weight * local + (1 - weight) * self.beta_
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'profiles': len(records),
            'retained_profiles': len(self.records_), 'rejected_ambiguous_profiles': rejected,
            'training_rating_pairs': len(pairs), 'rating_beta': self.beta_,
            'group_rating_betas': self.group_beta_, 'strict_past_only': True,
            'id_as_feature': False, 'row_deletion': False, 'base_refit': False}
        return self

    def predict(self, X):
        if self.base_predictor_ is None:
            result = np.full(len(X), self.fallback_, dtype=float)
        else:
            predictor = self.base_predictor_.predict if hasattr(self.base_predictor_, 'predict') else self.base_predictor_
            result = np.asarray(predictor(X), dtype=float).copy()
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError('Invalid compatible baseline prediction')
        linked = 0
        for i, (key, year, rating) in enumerate(zip(_keys(X), _years(X), _ratings(X))):
            past = [row for row in self.records_.get(key, ()) if row[0] < year]
            if not past:
                continue
            donor_year, donor_rating, donor_target, facility = past[-1]
            adjustment = 0.
            if np.isfinite([rating, donor_rating]).all():
                beta = self.group_beta_.get(facility, self.beta_)
                adjustment = float(np.clip(beta * (_rating_score(rating) - _rating_score(donor_rating)),
                                           -self.log_change_cap, self.log_change_cap))
            donor = donor_target * np.exp(adjustment)
            weight = self.alpha * len(past) / (len(past) + self.support_shrinkage)
            weight *= np.exp(-self.age_decay * max(0., year - donor_year - 1.))
            result[i] = weight * donor + (1 - weight) * result[i]
            linked += 1
        self.last_prediction_audit_ = {'query_rows': len(X), 'linked_rows': linked}
        return result


class RatingNormalizer:
    """Empirical monotone score curve with global -> class -> facility shrinkage."""
    def __init__(self, minimum_rows=30, class_shrinkage=200., facility_shrinkage=100.,
                 reference_scale=100., scale_floor=.05, scale_cap=10.):
        self.minimum_rows = int(minimum_rows)
        self.class_shrinkage, self.facility_shrinkage = float(class_shrinkage), float(facility_shrinkage)
        self.reference_scale, self.scale_floor, self.scale_cap = float(reference_scale), float(scale_floor), float(scale_cap)

    def _curve(self, rating, target):
        from sklearn.isotonic import IsotonicRegression
        if len(target) < self.minimum_rows or len(np.unique(rating)) < 2:
            return None
        return IsotonicRegression(increasing=False, y_min=self.reference_scale * self.scale_floor,
            y_max=self.reference_scale * self.scale_cap, out_of_bounds='clip').fit(rating, target)

    def fit(self, X, y):
        y = _validate(X, y); rating = _ratings(X)
        known = np.isfinite(rating)
        self.global_ = self._curve(rating[known], y[known])
        self.default_ = float(np.mean(y[known])) if known.any() else self.reference_scale
        self.groups_ = {}
        for column in ('building_class', 'facility_type'):
            labels = X[column].astype('string').fillna('__MISSING__').to_numpy()
            for group in np.unique(labels):
                mask = known & (labels == group)
                curve = self._curve(rating[mask], y[mask])
                if curve is not None:
                    self.groups_[(column, str(group))] = (int(mask.sum()), curve)
        return self

    def transform(self, X):
        rating = _ratings(X); known = np.isfinite(rating)
        result = np.ones(len(X))
        if not known.any():
            return result
        positions = np.flatnonzero(known)
        values = self.global_.predict(rating[known]) if self.global_ is not None else np.full(known.sum(), self.default_)
        for column, shrinkage in [('building_class', self.class_shrinkage), ('facility_type', self.facility_shrinkage)]:
            labels = X[column].astype('string').fillna('__MISSING__').to_numpy()[known]
            for group in np.unique(labels):
                fitted = self.groups_.get((column, str(group)))
                if fitted is not None:
                    count, curve = fitted; mask = labels == group
                    weight = count / (count + shrinkage)
                    values[mask] = weight * curve.predict(rating[known][mask]) + (1 - weight) * values[mask]
        result[positions] = np.clip(values / self.reference_scale, self.scale_floor, self.scale_cap)
        return result


class RatingNormalizedRegressor:
    """Original-unit MSE via target normalization and exactly equivalent weights."""
    def __init__(self, parameters=None, features=(29, 30, 31), threads=4):
        self.parameters = dict(parameters or {})
        self.features, self.threads = tuple(features), min(4, int(threads))

    def configuration(self):
        parameters = {'n_estimators': 200, 'max_depth': 24, 'min_samples_leaf': 2, 'max_features': .8}
        parameters.update(self.parameters)
        return {'strategy': 'rating_normalized_regression', 'version': 1, 'family': 'extra_trees',
            'parameters': parameters, 'features': list(self.features), 'loss': 'original_scale_squared_error',
            'sample_weight': 'g_squared', 'normalizer': 'temporal_past_only_isotonic_hierarchy',
            'normalizer_minimum_rows': 30, 'class_shrinkage': 200., 'facility_shrinkage': 100.,
            'reference_scale': 100., 'scale_floor': .05, 'scale_cap': 10., 'first_year_scale': 1.}

    @staticmethod
    def normalized_targets(y, scales):
        y, scales = np.asarray(y, float), np.asarray(scales, float)
        if y.shape != scales.shape or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Positive finite aligned normalization scales required')
        return y / scales, scales ** 2

    def fit(self, X, y, base_predictor=None):
        from sklearn.ensemble import ExtraTreesRegressor
        from automl_models import Encoder
        from automl_reg_combo import FeatureComposition
        y = _validate(X, y); years = _years(X)
        if np.any(y < 0):
            raise ValueError('Nonnegative consumption required')
        self.training_scales_ = np.ones(len(X)); folds = []
        for year in sorted(np.unique(years))[1:]:
            fit, valid = np.flatnonzero(years < year), np.flatnonzero(years == year)
            normalizer = RatingNormalizer().fit(X.iloc[fit], y[fit])
            self.training_scales_[valid] = normalizer.transform(X.iloc[valid])
            folds.append({'year': float(year), 'fit_rows': len(fit), 'valid_rows': len(valid)})
        self.normalizer_ = RatingNormalizer().fit(X, y)
        self.feature_ = FeatureComposition(self.features).fit(X)
        frame = self.feature_.transform(X)
        frame['sprint_rating_missing'] = ~np.isfinite(_ratings(X))
        self.encoder_ = Encoder(native=False, scale=False).fit(frame)
        values = self.encoder_.transform(frame)
        target, weight = self.normalized_targets(y, self.training_scales_)
        parameters = self.configuration()['parameters']
        for field in ('n_jobs', 'random_state'):
            parameters.pop(field, None)
        self.model_ = ExtraTreesRegressor(**parameters, n_jobs=self.threads, random_state=42)
        self.model_.fit(values, target, sample_weight=weight)
        self.audit_ = {**self.configuration(), 'fit_rows': len(X), 'temporal_normalizer_folds': folds,
            'training_scale_min': float(self.training_scales_.min()), 'training_scale_max': float(self.training_scales_.max()),
            'row_deletion': False, 'validation_target_used': False, 'base_refit': False,
            'weight_min': float(weight.min()), 'weight_max': float(weight.max())}
        return self

    def predict(self, X):
        frame = self.feature_.transform(X)
        frame['sprint_rating_missing'] = ~np.isfinite(_ratings(X))
        prediction = np.asarray(self.model_.predict(self.encoder_.transform(frame)), float)
        prediction *= self.normalizer_.transform(X)
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite regression prediction')
        return prediction
