"""Task-local temporal donor transport and supervised leaf retrieval experiments."""
import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.linear_model import Ridge
from automl_panel import CausalPanelFeatures


class TemporalDonor:
    """Discover a collision-resistant key; learn changes from strictly past pairs."""
    def __init__(self, time_column, transport=True):
        self.time_column, self.transport = time_column, transport

    def fit(self, X, y):
        self.key_ = CausalPanelFeatures(self.time_column).fit(X, y)
        self.columns_ = list(self.key_.columns_)
        self.records_, self.curves_, self.regressor_ = {}, {}, None
        periods = pd.to_numeric(X[self.time_column], errors='coerce').to_numpy(float)
        if not self.columns_:
            self.audit_ = {'key_columns': [], 'pairs': 0}
            return self
        # Prefer disambiguation to aggressive coverage: false links are harmful.
        def quality(cols):
            _, coverage, unique = self.key_._quality(X, cols, periods)
            return coverage * unique ** 4, coverage
        current = quality(self.columns_)
        for _ in range(4):
            options = [c for c in X if c not in self.columns_ + [self.time_column]
                       and 2 <= X[c].nunique() <= 256]
            choices = [(quality(self.columns_ + [c]), c) for c in options]
            if not choices:
                break
            best, column = max(choices, key=lambda t: t[0][0])
            if best[0] < current[0] + .002 or best[1] < .3:
                break
            self.columns_.append(column)
            current = best
        keys = self.key_._keys(X, self.columns_)
        if isinstance(self.transport, str):
            frame = X.copy()
            frame['__key'] = keys
            order = np.lexsort((periods, keys))
            previous, current = order[:-1], order[1:]
            same = (keys[previous] == keys[current]) & (periods[previous] != periods[current])
            previous, current = previous[same], current[same]
            for col in X:
                if col in self.columns_ + [self.time_column] or not 2 <= X[col].nunique() <= 256:
                    continue
                values = X[col].astype('string').fillna('__MISSING__').to_numpy()
                agreement = np.mean(values[previous] == values[current]) if len(current) else 0
                chance = np.sum(X[col].value_counts(normalize=True, dropna=False).to_numpy() ** 2)
                stability = (agreement - chance) / max(1e-12, 1 - chance)
                if stability >= .99:
                    self.columns_.append(col)
            keys = self.key_._keys(X, self.columns_)
        y = np.asarray(y, float)
        rows = pd.DataFrame({'key': keys, 'time': periods, 'position': np.arange(len(X))})
        pairs, rejected = [], 0
        for key, group in rows.groupby('key', sort=False):
            if group.time.duplicated().any():
                rejected += 1
                continue
            order = group.sort_values('time').position.to_numpy(int)
            self.records_[key] = (periods[order], y[order], order)
            pairs.extend(zip(order[:-1], order[1:]))
        numeric = [c for c in X if c not in self.columns_ + [self.time_column]
                   and pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() >= 4]
        for col in numeric:
            values = pd.to_numeric(X[col], errors='coerce').to_numpy(float)
            finite = np.sort(values[np.isfinite(values)])
            if len(finite):
                self.curves_[col] = finite
        self.train_numeric_ = self._numeric(X)
        self.step_ = float(np.median(np.diff(np.unique(periods))))
        self.step_ = max(self.step_, 1e-9)
        self.audit_ = {'key_columns': self.columns_, 'rejected_keys': rejected,
                       'retained_keys': len(self.records_), 'pairs': len(pairs),
                       'uses_query_for_discovery': False, 'strict_past': True,
                       'transport': self.transport}
        if self.transport and len(pairs) >= 300 and self.curves_:
            previous, current = np.asarray(pairs, int).T
            eligible = (y[previous] > 1) & (y[current] >= 0)
            previous, current = previous[eligible], current[eligible]
            matrix = self.train_numeric_[current] - self.train_numeric_[previous]
            response = y[current] / y[previous] - 1
            weight = y[previous] ** 2
            weight /= weight.mean()
            if isinstance(self.transport, str):
                log_response = np.clip(np.log(np.maximum(y[current], 1e-6) / y[previous]), -1.5, 1.5)
                norm = np.sqrt(np.sum(matrix ** 2, axis=0) * np.sum(log_response ** 2))
                correlation = np.abs(matrix.T @ log_response) / np.maximum(norm, 1e-12)
                self.driver_ = int(np.argmax(correlation))
                self.audit_['driver'] = list(self.curves_)[self.driver_]
                self.audit_['driver_correlation'] = float(correlation[self.driver_])
                if self.transport == 'log':
                    coef = np.zeros(matrix.shape[1])
                    delta = matrix[:, self.driver_]
                    coef[self.driver_] = (delta @ log_response) / (delta @ delta + 20)
                    self.regressor_ = Ridge()
                    self.regressor_.coef_ = coef
                else:
                    from sklearn.isotonic import IsotonicRegression
                    increasing = bool(matrix[:, self.driver_] @ log_response >= 0)
                    self.regressor_ = IsotonicRegression(increasing=increasing, y_min=.2, y_max=3,
                        out_of_bounds='clip').fit(matrix[:, self.driver_], response + 1, sample_weight=weight)
            else:
                self.regressor_ = Ridge(alpha=100, fit_intercept=False).fit(matrix, response, sample_weight=weight)
                self.audit_['coefficients'] = dict(zip(self.curves_, self.regressor_.coef_.tolist()))
        return self

    def _numeric(self, X):
        columns = []
        for col, values in self.curves_.items():
            raw = pd.to_numeric(X[col], errors='coerce').to_numpy(float)
            quantile = (np.searchsorted(values, raw, side='left') + .5) / len(values)
            columns.append(np.where(np.isfinite(raw), ndtri(np.clip(quantile, .005, .995)), 0))
        return np.column_stack(columns) if columns else np.empty((len(X), 0))

    def predict(self, X, fallback, strength=.95):
        result = np.asarray(fallback, float).copy()
        if not self.columns_:
            return result
        periods = pd.to_numeric(X[self.time_column], errors='coerce').to_numpy(float)
        numeric = self._numeric(X)
        linked = 0
        for i, (key, period) in enumerate(zip(self.key_._keys(X, self.columns_), periods)):
            record = self.records_.get(key)
            if record is None or not np.isfinite(period):
                continue
            times, target, positions = record
            stop = np.searchsorted(times, period, side='left')
            if not stop:
                continue
            donor = float(target[stop - 1])
            if self.regressor_ is not None:
                delta = numeric[i] - self.train_numeric_[positions[stop - 1]]
                if self.transport == 'log':
                    ratio = np.exp(np.clip(delta @ self.regressor_.coef_, -1.5, 1.5))
                elif self.transport == 'isotonic':
                    ratio = float(self.regressor_.predict([delta[self.driver_]])[0])
                else:
                    ratio = 1 + float(delta @ self.regressor_.coef_)
                donor *= np.clip(ratio, .2, 3)
            confidence = strength * stop / (stop + .25)
            confidence *= np.exp(-.25 * max(0, (period - times[stop - 1]) / self.step_ - 1))
            result[i] = confidence * donor + (1 - confidence) * result[i]
            linked += 1
        self.last_linked_ = linked
        return result


class LeafRetrieval:
    """Retrieve training labels using supervised tree partitions, no query labels."""
    def __init__(self, neighbours=30, trees=128):
        self.neighbours, self.trees = neighbours, trees

    def fit(self, model, X, y):
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.neighbors import NearestNeighbors
        self.base_ = model
        leaves = model.model_.booster_.predict(model._matrix(X), pred_leaf=True)
        self.selected_ = np.linspace(0, leaves.shape[1] - 1, min(self.trees, leaves.shape[1]), dtype=int)
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
        matrix = self.encoder_.fit_transform(leaves[:, self.selected_])
        self.index_ = NearestNeighbors(n_neighbors=min(self.neighbours, len(X)), metric='cosine', n_jobs=1).fit(matrix)
        self.classes_, self.target_ = np.unique(np.asarray(y), return_inverse=True)
        self.priors_ = np.bincount(self.target_, minlength=len(self.classes_)) / len(y)
        return self

    def predict_proba(self, X):
        leaves = self.base_.model_.booster_.predict(self.base_._matrix(X), pred_leaf=True)
        matrix = self.encoder_.transform(leaves[:, self.selected_])
        result = np.zeros((len(X), len(self.classes_)))
        for start in range(0, len(X), 128):
            distance, indices = self.index_.kneighbors(matrix[start:start + 128])
            weight = np.exp(-10 * distance)
            for cls in range(len(self.classes_)):
                result[start:start + len(indices), cls] = np.sum(weight * (self.target_[indices] == cls), axis=1)
        result += .5 * self.priors_
        return result / result.sum(axis=1, keepdims=True)
