"""Generic causal residual model for recurring entities, with an explicit cold fallback."""
import numpy as np
import pandas as pd
from automl_panel import CausalPanelFeatures, stable_panel_columns


class CausalResidual:
    def __init__(self, time_column, threads=4, strict_key=False):
        self.time_column, self.threads = time_column, threads
        self.strict_key = strict_key

    def _features(self, X):
        names = ['__causal_mean', '__causal_last', '__causal_std', '__causal_min',
                 '__causal_max', '__causal_count', '__causal_gap']
        if set(names) & set(X.columns):
            raise ValueError('Causal residual feature collision')
        features = np.full((len(X), len(names)), np.nan)
        periods = pd.to_numeric(X[self.time_column], errors='coerce').to_numpy(float)
        for i, (key, period) in enumerate(zip(self.history_._keys(X, self.history_.columns_), periods)):
            records = self.history_.history_.get(key)
            if records is None or not np.isfinite(period):
                continue
            stop = np.searchsorted(records[:, 0], period, side='left')
            if stop:
                past = records[:stop]
                weights = .5 ** ((past[-1, 0] - past[:, 0]) / self.step_) * past[:, 2]
                y = past[:, 1]
                features[i] = [np.average(y, weights=weights), y[-1], y.std(), y.min(), y.max(),
                               past[:, 2].sum(), (period - past[-1, 0]) / self.step_]
        return pd.concat([X.drop(columns=[self.time_column]),
                          pd.DataFrame(features, index=X.index, columns=names)], axis=1)

    def fit(self, X, y):
        from automl_models import CandidateModel
        self.history_ = CausalPanelFeatures(self.time_column).fit(X, y)
        if self.strict_key and self.history_.columns_:
            self.history_.columns_ = stable_panel_columns(X, self.time_column, self.history_.columns_)
            keys = self.history_._keys(X, self.history_.columns_)
            frame = pd.DataFrame({'key': keys, 'time': pd.to_numeric(X[self.time_column]), 'y': np.asarray(y)})
            grouped = frame.groupby(['key', 'time']).y.agg(['mean', 'count']).reset_index()
            self.history_.history_ = {key: part.sort_values('time')[['time', 'mean', 'count']].to_numpy(float)
                                     for key, part in grouped.groupby('key', sort=False)}
        self.model_ = None
        self.audit_ = {'key_columns': self.history_.columns_, 'fit_rows': len(X),
                       'causal_fit_rows': 0, 'strict_past': True, 'query_discovery': False}
        if not self.history_.columns_:
            return self
        periods = np.sort(pd.to_numeric(X[self.time_column], errors='coerce').dropna().unique())
        self.step_ = max(float(np.median(np.diff(periods))), 1e-9)
        frame = self._features(X)
        matched = frame['__causal_count'].notna().to_numpy()
        self.audit_['causal_fit_rows'] = int(matched.sum())
        if matched.sum() < 500:
            return self
        target = np.asarray(y, float)[matched] - frame.loc[matched, '__causal_mean'].to_numpy()
        self.model_ = CandidateModel('regression', {'family': 'lightgbm', 'parameters': {
            'n_estimators': 600, 'learning_rate': .03, 'num_leaves': 15,
            'min_child_samples': 40, 'reg_lambda': 10}}, self.threads).fit(frame.loc[matched], target)
        return self

    def predict(self, X, fallback):
        output = np.asarray(fallback, float).copy()
        if self.model_ is None:
            return output
        frame = self._features(X)
        matched = frame['__causal_count'].notna().to_numpy()
        if matched.any():
            output[matched] = frame.loc[matched, '__causal_mean'].to_numpy() + self.model_.predict(frame.loc[matched])
        return output

    def stable(self, X, fallback):
        output = np.asarray(fallback, float).copy()
        if not self.history_.columns_:
            return output
        frame = self._features(X)
        keys = self.history_._keys(X, self.history_.columns_)
        ambiguous = {key for key, records in self.history_.history_.items() if (records[:, 2] > 1).any()}
        matched = frame['__causal_count'].notna().to_numpy() & ~np.isin(keys, list(ambiguous))
        count = frame.loc[matched, '__causal_count'].to_numpy()
        gap = frame.loc[matched, '__causal_gap'].to_numpy()
        alpha = np.minimum(.9, count / (count + 1)) * np.exp(-.25 * np.maximum(0, gap - 1))
        output[matched] = alpha * frame.loc[matched, '__causal_mean'].to_numpy() + (1 - alpha) * output[matched]
        return output


class CausalHybridModel:
    """Strong global fallback plus complementary stable levels and annual changes."""
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = candidate, threads

    def fit(self, X, y):
        from automl_models import CandidateModel
        base = {k: v for k, v in self.candidate.items() if k != 'panel_strategy'}
        self.base_ = CandidateModel('regression', base, self.threads).fit(X, y)
        self.residual_ = CausalResidual(self.candidate['panel_time_column'], self.threads, strict_key=True).fit(X, y)
        self.feature_code_, self.classes_ = self.base_.feature_code_, None
        self.audit_ = {**self.base_.audit_, 'causal_hybrid': self.residual_.audit_,
                       'hybrid_weights': [.5, .5], 'public_fitted_state': False}
        return self

    def predict(self, X):
        base = self.base_.predict(X)
        return .5 * self.residual_.stable(X, base) + .5 * self.residual_.predict(X, base)
