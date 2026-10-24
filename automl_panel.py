"""Train-only discovery of repeated entities and strictly-past numeric history."""
import numpy as np
import pandas as pd


class CausalPanelFeatures:
    def __init__(self, time_column):
        self.time_column = time_column

    def _keys(self, X, columns):
        # Shared representation avoids train/test numeric dtype differences.
        frame = pd.DataFrame(index=X.index)
        for col in columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                frame[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
            else:
                frame[col] = X[col].astype('string').fillna('__MISSING__')
        return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)

    def _quality(self, X, columns, periods):
        keys = self._keys(X, columns)
        pairs = pd.DataFrame({'key': keys, 'time': periods}).drop_duplicates()
        counts = pairs.groupby('key').time.nunique()
        coverage = float(pd.Series(keys).map(counts).gt(1).mean())
        uniqueness = len(pairs) / max(1, len(X))
        return 2 * coverage + uniqueness, coverage, uniqueness

    def fit(self, X, y):
        self.columns_ = []
        self.history_ = {}
        periods = pd.to_numeric(X[self.time_column], errors='coerce').to_numpy(float)
        if not np.isfinite(periods).all() or len(np.unique(periods)) < 2:
            return self
        options = [c for c in X if c != self.time_column and 2 <= X[c].nunique() <= len(X) * .9]
        anchors = [c for c in options if X[c].nunique() >= max(20, len(X) * .01)]
        ranked = [(self._quality(X, [c], periods), c) for c in anchors]
        if not ranked:
            return self
        quality, first = max(ranked, key=lambda item: item[0][0])
        if quality[1] < .3 or quality[2] < .3:
            return self
        self.columns_ = [first]
        for _ in range(4):
            choices = [(self._quality(X, self.columns_ + [c], periods), c) for c in options if c not in self.columns_]
            if not choices:
                break
            new_quality, column = max(choices, key=lambda item: item[0][0])
            if new_quality[0] < quality[0] + .005 or new_quality[1] < .3:
                break
            self.columns_.append(column)
            quality = new_quality
        keys = self._keys(X, self.columns_)
        frame = pd.DataFrame({'key': keys, 'time': periods, 'y': np.asarray(y, float)})
        grouped = frame.groupby(['key', 'time']).y.agg(['mean', 'count']).reset_index()
        for key, part in grouped.groupby('key', sort=False):
            self.history_[key] = part.sort_values('time')[['time', 'mean', 'count']].to_numpy(float)
        self.audit_ = {'key_columns': self.columns_, 'cross_period_coverage': quality[1],
                       'within_period_uniqueness': quality[2], 'uses_test_for_discovery': False}
        return self

    def transform(self, X):
        if not self.columns_:
            return X.copy()
        names = ['__panel_last', '__panel_mean', '__panel_periods', '__panel_gap', '__panel_std']
        if set(names) & set(X.columns):
            raise ValueError('Causal feature name collision')
        periods = pd.to_numeric(X[self.time_column], errors='coerce').to_numpy(float)
        result = np.full((len(X), len(names)), np.nan)
        result[:, 2] = 0
        for i, (key, period) in enumerate(zip(self._keys(X, self.columns_), periods)):
            records = self.history_.get(key)
            if records is None or not np.isfinite(period):
                continue
            stop = np.searchsorted(records[:, 0], period, side='left')
            if stop:
                history = records[:stop]
                values = history[:, 1]
                result[i] = [values[-1], np.average(values, weights=history[:, 2]), len(values),
                             period - history[-1, 0], np.std(values)]
        return pd.concat([X, pd.DataFrame(result, columns=names, index=X.index)], axis=1)
