"""Train-only discovery of a monotone change driver for recurring entities.

Adapted mechanism: Ivan H83 original-unit weighted ratio isotonic transport.
Column/key/driver discovery is generic; no public schema or learned state.
Research candidate, not included in the verified delivery until CLI validation.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.isotonic import IsotonicRegression
from automl_panel import CausalPanelFeatures, stable_panel_columns


class DiscoveredRatio:
    def __init__(self, time_column):
        self.time_column = time_column

    def _values(self, X, column):
        values = pd.to_numeric(X[column], errors='coerce').to_numpy(float)
        reference, percentage = self.references_[column]
        if percentage:
            quantile = (values + .5) / 100
        else:
            quantile = (np.searchsorted(reference, values, side='left') +
                        np.searchsorted(reference, values, side='right')) / (2 * len(reference))
        return np.where(np.isfinite(values), ndtri(np.clip(quantile, .005, .995)), np.nan)

    @staticmethod
    def _curve(delta, ratio, weight):
        direction = np.sum(weight * delta * (ratio-1)) >= 0
        return IsotonicRegression(increasing=bool(direction), y_min=np.exp(-1.5),
                                  y_max=np.exp(1.5), out_of_bounds='clip').fit(delta, ratio, sample_weight=weight)

    def fit(self, X, y):
        y = np.asarray(y, float)
        if y.shape != (len(X),) or not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError('Nonnegative finite aligned targets required')
        self.helper_ = CausalPanelFeatures(self.time_column).fit(X, y)
        self.columns_ = stable_panel_columns(X, self.time_column, self.helper_.columns_) if self.helper_.columns_ else []
        self.records_, self.references_, self.driver_, self.curve_ = {}, {}, None, None
        self.groups_, self.group_counts_, self.group_column_ = {}, {}, None
        self.audit_ = {'key_columns': self.columns_, 'strict_past': True, 'query_discovery': False,
                       'source_mechanism': 'Ivan H83; generic roles, finite-pair discovery, percentile support'}
        if not self.columns_:
            return self
        periods = pd.to_numeric(X[self.time_column], errors='raise').to_numpy(float)
        unique = np.unique(periods)
        self.step_ = max(1e-9, float(np.median(np.diff(unique))))
        keys = self.helper_._keys(X, self.columns_)
        frame = pd.DataFrame({'key': keys, 'period': periods, 'position': np.arange(len(X))})
        pairs = []
        for key, part in frame.groupby('key', sort=False):
            if part.period.duplicated().any():
                continue
            positions = part.sort_values('period').position.to_numpy(int)
            self.records_[key] = (periods[positions], y[positions], positions)
            pairs.extend(zip(positions[:-1], positions[1:]))
        self.audit_['pairs'] = len(pairs)
        if len(pairs) < 300:
            return self
        previous, current = np.asarray(pairs).T
        eligible = (y[previous] > 0) & (y[current] > 0)
        previous, current = previous[eligible], current[eligible]
        response = np.clip(np.log(y[current] / y[previous]), -1.5, 1.5)
        choices = []
        for column in X:
            if column in self.columns_ + [self.time_column] or not pd.api.types.is_numeric_dtype(X[column]):
                continue
            raw = pd.to_numeric(X[column], errors='coerce').to_numpy(float)
            reference = np.sort(raw[np.isfinite(raw)])
            if len(np.unique(reference)) < 10:
                continue
            percentage = bool(reference.min() >= 0 and 95 <= reference.max() <= 100 and np.mean(reference == np.rint(reference)) >= .99)
            self.references_[column] = (reference, percentage)
            values = self._values(X, column)
            delta = values[current] - values[previous]
            finite = np.isfinite(delta)
            if finite.sum() < 300 or np.std(delta[finite]) < 1e-9:
                continue
            correlation = abs(float(np.corrcoef(delta[finite], response[finite])[0, 1]))
            if np.isfinite(correlation):
                choices.append((correlation, column))
        if not choices:
            return self
        correlation, self.driver_ = max(choices)
        self.train_values_ = self._values(X, self.driver_)
        delta = self.train_values_[current] - self.train_values_[previous]
        finite = np.isfinite(delta)
        previous, current, delta = previous[finite], current[finite], delta[finite]
        ratio, weight = y[current] / y[previous], y[previous] ** 2
        self.curve_ = self._curve(delta, ratio, weight)
        # Hierarchical curve uses the most informative-sized categorical partition,
        # selected on X/support only; no validation target selects its identity.
        options = [c for c in X if not pd.api.types.is_numeric_dtype(X[c]) and 2 <= X[c].nunique() <= 128]
        if options:
            self.group_column_ = max(options, key=lambda c: X[c].nunique())
            values = X[self.group_column_].astype('string').fillna('__MISSING__').to_numpy()[current]
            for group in np.unique(values):
                mask = values == group
                self.group_counts_[group] = int(mask.sum())
                if mask.sum() >= 100:
                    self.groups_[group] = self._curve(delta[mask], ratio[mask], weight[mask])
        self.audit_.update(driver=self.driver_, correlation=correlation, finite_pairs=len(delta),
                           percentile_support=self.references_[self.driver_][1], group_column=self.group_column_)
        return self

    def predict(self, X, fallback):
        output = np.asarray(fallback, float).copy()
        if self.curve_ is None:
            return output
        values = self._values(X, self.driver_)
        groups = X[self.group_column_].astype('string').fillna('__MISSING__').to_numpy() if self.group_column_ else [None]*len(X)
        periods = pd.to_numeric(X[self.time_column], errors='raise').to_numpy(float)
        linked = 0
        for i, (key, period) in enumerate(zip(self.helper_._keys(X, self.columns_), periods)):
            record = self.records_.get(key)
            if record is None:
                continue
            times, targets, positions = record
            stop = np.searchsorted(times, period, side='left')
            if not stop:
                continue
            ratio = 1.
            delta = values[i] - self.train_values_[positions[stop-1]]
            if np.isfinite(delta):
                ratio = float(self.curve_.predict([delta])[0])
                if groups[i] in self.groups_:
                    count = self.group_counts_[groups[i]]
                    alpha = count / (count+200)
                    ratio = alpha*float(self.groups_[groups[i]].predict([delta])[0]) + (1-alpha)*ratio
            confidence = .95 * stop / (stop+.25) * np.exp(-.25*max(0, (period-times[stop-1])/self.step_-1))
            output[i] = confidence*targets[stop-1]*ratio + (1-confidence)*output[i]
            linked += 1
        self.audit_['last_query_linked'] = linked
        return output
