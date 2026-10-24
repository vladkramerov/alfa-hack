"""Past-only panel retrieval with explicit collision and uncertainty accounting.

All identities and statistics are fitted exclusively on the training partition.
Identifiers and query targets are never inputs to this module.
"""
import json
import numpy as np
import pandas as pd

PANEL_FIELDS = ['State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built']


def panel_keys(X, fields):
    if not fields or any(c not in X for c in fields):
        raise ValueError('Panel fields are unavailable')
    values = []
    for c in fields:
        if pd.api.types.is_numeric_dtype(X[c]):
            values.append(pd.to_numeric(X[c], errors='coerce').map(
                lambda x: 'M:' if not np.isfinite(x) else 'N:' + format(float(x), '.17g')).to_numpy())
        else:
            values.append(('V:' + X[c].astype('string')).fillna('M:').to_numpy())
    return [json.dumps(list(row), ensure_ascii=False, separators=(',', ':')) for row in zip(*values)]


class PanelHistory:
    def __init__(self, method='stable', fields=None, time_column='Year_Factor'):
        self.method = method
        self.fields = list(fields or PANEL_FIELDS)
        self.time_column = time_column

    def fit(self, X, y):
        if self.method not in {'stable', 'reliability'}:
            raise ValueError('Unsupported panel estimator')
        years = pd.to_numeric(X[self.time_column], errors='raise').to_numpy(float)
        y = np.asarray(y, float)
        if y.shape != (len(X),) or not np.isfinite(y).all() or not np.isfinite(years).all():
            raise ValueError('Invalid panel training data')
        frame = pd.DataFrame({'key': panel_keys(X, self.fields), 'year': years, 'target': y})
        self.records_ = {k: part[['year', 'target']].to_numpy(float)
                         for k, part in frame.groupby('key', sort=False)}
        moments = frame.groupby(['key', 'year']).target.agg(['mean', 'count', 'var']).reset_index()
        differences = moments.sort_values(['key', 'year']).groupby('key')['mean'].diff().dropna().to_numpy()
        # For a persistent latent level with independent yearly disturbances,
        # Var(y_t-y_t-1)/2 estimates disturbance variance. Robust trimming here
        # is a fixed 95th percentile of training pair squared differences only.
        squared = differences ** 2
        cap = float(np.quantile(squared, .95)) if len(squared) else 0.
        self.noise_ = float(np.mean(np.minimum(squared, cap)) / 2) if len(squared) else np.inf
        self.between_ = float(frame.groupby('key').target.mean().var(ddof=0)) if len(frame) else 0.
        self.audit_ = {'fit_original_rows': len(X), 'fit_retained_rows': len(X),
                       'profile_fields': self.fields, 'time_column': self.time_column,
                       'method': self.method, 'profile_count': len(self.records_),
                       'same_profile_year_multiple_rows': int((moments['count'] > 1).sum()),
                       'profile_is_proven_building_identity': False, 'strict_past_only': True,
                       'id_as_key': False, 'training_pair_count': len(differences),
                       'training_noise_variance': self.noise_, 'training_between_variance': self.between_}
        return self

    def lookup(self, X):
        years = pd.to_numeric(X[self.time_column], errors='raise').to_numpy(float)
        if not np.isfinite(years).all():
            raise ValueError('Invalid panel query time')
        mean, alpha = np.zeros(len(X)), np.zeros(len(X))
        collision_rows = 0
        for i, (key, year) in enumerate(zip(panel_keys(X, self.fields), years)):
            records = self.records_.get(key)
            if records is None:
                continue
            past = records[records[:, 0] < year]
            if not len(past):
                continue
            latest = float(past[:, 0].max())
            gap = max(0., year - latest - 1.)
            if self.method == 'stable':
                weights = .5 ** (latest - past[:, 0])
                mean[i] = np.average(past[:, 1], weights=weights)
                alpha[i] = min(.9, len(past) / (len(past) + 1.)) * np.exp(-.25 * gap)
            else:
                last = past[past[:, 0] == latest, 1]
                mean[i] = last.mean()
                ambiguity = float(np.var(last, ddof=0))
                disturbance = self.noise_ + ambiguity
                reliability = self.between_ / (self.between_ + disturbance + 1e-12)
                alpha[i] = min(.95, max(0., reliability)) * np.exp(-.25 * gap)
            collision_rows += int(len(past) != len(np.unique(past[:, 0])))
        audit = {'query_rows': len(X), 'matched_rows': int((alpha > 0).sum()),
                 'matched_fraction': float((alpha > 0).mean()) if len(X) else 0.,
                 'mean_memory_weight': float(alpha.mean()) if len(X) else 0.,
                 'queries_with_same_profile_year_collisions': collision_rows,
                 'no_query_targets_used': True}
        return mean, alpha, audit

    def features(self, X):
        """Every statistic is computed from strictly earlier training rows."""
        years = pd.to_numeric(X[self.time_column], errors='raise').to_numpy(float)
        values = np.full((len(X), 8), np.nan)
        matched = np.zeros(len(X), dtype=bool)
        for i, (key, year) in enumerate(zip(panel_keys(X, self.fields), years)):
            records = self.records_.get(key)
            if records is None:
                continue
            past = records[records[:, 0] < year]
            if not len(past):
                continue
            latest = past[:, 0].max()
            y = past[:, 1]
            weights = .5 ** (latest-past[:, 0])
            last = past[past[:, 0] == latest, 1]
            values[i] = [np.average(y, weights=weights), last.mean(), y.std(ddof=0),
                         y.min(), y.max(), len(y), year-latest, len(last)]
            matched[i] = True
        names = ['mean', 'last', 'std', 'min', 'max', 'count', 'gap', 'collision_count']
        return pd.DataFrame(values, index=X.index, columns=['__panel_'+n for n in names]), matched


class PanelDynamics:
    """Learn annual residuals from causal past-profile features, not query labels."""
    def __init__(self, threads=4):
        self.threads = threads

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        from automl_models import Encoder
        from automl_combo_memory import fit_once
        self.history_ = PanelHistory().fit(X, y)
        temporal, matched = self.history_.features(X)
        if matched.sum() < 500:
            raise ValueError('Insufficient causal training pairs')
        frame = pd.concat([X.drop(columns=['Year_Factor']), temporal], axis=1)
        if not frame.columns.is_unique:
            raise ValueError('Panel feature collision')
        self.encoder_ = Encoder(native=True, scale=False)
        values = self.encoder_.fit_transform(frame.loc[matched])
        target = np.asarray(y, float)[matched]-temporal.loc[matched, '__panel_mean'].to_numpy()
        settings = dict(n_estimators=600, learning_rate=.03, num_leaves=15,
                        min_child_samples=40, reg_lambda=10., verbosity=-1,
                        random_state=42, n_jobs=self.threads, force_col_wise=True)
        self.model_ = fit_once(LGBMRegressor(**settings), values, target,
            {'strategy': 'causal_panel_dynamics', 'parameters': settings,
             'target': 'current_minus_strict_past_ew_mean', 'columns': list(frame.columns)})
        self.nonnegative_ = bool(np.all(np.asarray(y) >= 0))
        self.audit_ = {**self.history_.audit_, 'strategy': 'causal_panel_dynamics',
                       'causal_model_rows': int(matched.sum()), 'parameters': settings,
                       'validation_targets_used': False, 'id_feature_used': False}
        return self

    def predict_with_fallback(self, X, fallback):
        fallback = np.asarray(fallback, float)
        if fallback.shape != (len(X),) or not np.isfinite(fallback).all():
            raise ValueError('Invalid panel fallback')
        temporal, matched = self.history_.features(X)
        output = fallback.copy()
        if matched.any():
            frame = pd.concat([X.drop(columns=['Year_Factor']), temporal], axis=1)
            output[matched] = temporal.loc[matched, '__panel_mean'].to_numpy()+self.model_.predict(
                self.encoder_.transform(frame.loc[matched]))
            if self.nonnegative_:
                output[matched] = np.maximum(output[matched], 0.)
        return output, matched


class PanelStateForest(PanelDynamics):
    """Randomized original-scale conditional mean over causal panel states."""
    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesRegressor
        from automl_models import Encoder
        from automl_combo_memory import fit_once
        self.history_ = PanelHistory().fit(X, y)
        temporal, matched = self.history_.features(X)
        if matched.sum() < 500:
            raise ValueError('Insufficient causal training pairs')
        frame = pd.concat([X.drop(columns=['Year_Factor']), temporal], axis=1)
        if not frame.columns.is_unique:
            raise ValueError('Panel feature collision')
        self.encoder_ = Encoder(native=False, scale=False)
        values = self.encoder_.fit_transform(frame.loc[matched])
        settings = dict(n_estimators=300, max_depth=24, min_samples_leaf=4,
                        max_features=1., random_state=42, n_jobs=self.threads)
        self.model_ = fit_once(ExtraTreesRegressor(**settings), values, np.asarray(y)[matched],
            {'strategy': 'causal_panel_state_forest', 'parameters': settings,
             'target': 'current_original_scale', 'columns': list(frame.columns)})
        self.nonnegative_ = bool(np.all(np.asarray(y) >= 0))
        self.audit_ = {**self.history_.audit_, 'strategy': 'causal_panel_state_forest',
                       'causal_model_rows': int(matched.sum()), 'parameters': settings,
                       'validation_targets_used': False, 'id_feature_used': False}
        return self

    def predict_with_fallback(self, X, fallback):
        temporal, matched = self.history_.features(X)
        fallback = np.asarray(fallback, float)
        if fallback.shape != (len(X),) or not np.isfinite(fallback).all():
            raise ValueError('Invalid panel fallback')
        output = fallback.copy()
        if matched.any():
            frame = pd.concat([X.drop(columns=['Year_Factor']), temporal], axis=1)
            output[matched] = self.model_.predict(self.encoder_.transform(frame.loc[matched]))
        return output, matched


class StablePanelModel:
    def __init__(self, candidate=None, threads=4):
        self.candidate, self.threads = dict(candidate or {}), threads

    def fit(self, X, y):
        from automl_reg_combo import FeatureGroupExperts
        self.base_ = FeatureGroupExperts({'feature_numbers': [29, 30, 31],
            'group_column': 'building_class', 'minimum_expert_rows': 500, 'shrinkage': 500.,
            'parameters': self.candidate.get('parameters', {})}, self.threads).fit(X, y)
        self.memory_ = PanelHistory(method=self.candidate.get('panel_method', 'stable')).fit(X, y)
        self.audit_ = {**self.memory_.audit_, 'strategy': 'stable_panel_memory', 'base': self.base_.audit_}
        self.keep_mask_ = np.ones(len(X), dtype=bool)
        self.removed_positions_ = np.empty(0, dtype=int)
        return self

    def predict(self, X):
        mean, alpha, _ = self.memory_.lookup(X)
        return alpha * mean + (1. - alpha) * self.base_.predict(X)


class PanelHybridModel(StablePanelModel):
    """H73: fixed 50/50 stable lookup and causal annual-dynamics expert."""
    def fit(self, X, y):
        super().fit(X, y)
        self.dynamics_ = PanelDynamics(self.threads).fit(X, y)
        self.audit_.update(strategy='panel_hybrid_memory', fixed_blend_weights=[.5, .5],
                           dynamics=self.dynamics_.audit_)
        return self

    def predict(self, X):
        base = self.base_.predict(X)
        mean, alpha, _ = self.memory_.lookup(X)
        stable = alpha * mean + (1-alpha) * base
        dynamics, _ = self.dynamics_.predict_with_fallback(X, base)
        return .5*stable+.5*dynamics
