"""Independent exact, bounded-memory nearest neighbors for mixed tables.

Numeric values and their original missing indicators form the numeric component.
Each value uses a training median and IQR; pairwise absolute differences are
capped at three. The categorical component is a mean lexical mismatch. Available
components have equal weight. Neighbor ties use input row order; vote ties use
the order in which classes first appeared during fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class MixedNeighborsClassifier:
    """Exact mixed-distance classifier without an all-pairs allocation.

    Blocks change memory use, never the mathematical neighbor ordering. There
    are no learned parameters or vocabulary updates during prediction.
    """

    def __init__(self, n_neighbors=25, *, query_block_size=128, train_block_size=8192):
        self.n_neighbors = n_neighbors
        self.query_block_size = query_block_size
        self.train_block_size = train_block_size

    def configuration(self):
        return {
            "model": "MixedNeighborsClassifier", "n_neighbors": self.n_neighbors,
            "query_block_size": self.query_block_size,
            "train_block_size": self.train_block_size,
            "numeric": "train_median_iqr_capped_l1_3_with_all_original_missing_flags",
            "numeric_denominator": "2 * number_of_numeric_columns",
            "categorical": "mean_lexical_mismatch_separate_missing_and_unknown",
            "type_weighting": "equal_available_components",
            "weights": "uniform", "neighbor_tie": "training_row_order",
            "class_tie": "first_appearance_in_training",
        }

    @staticmethod
    def _frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.has_duplicates or not len(X.columns):
            raise ValueError("X needs unique, nonempty feature columns")
        return X

    @staticmethod
    def _category(value):
        # Tagged tuples prevent actual strings such as '<MISSING>' or 'NA'
        # from colliding with absent cells; strings retain leading zeroes.
        return ("missing",) if pd.isna(value) else ("value", str(value))

    def fit(self, X, y):
        X = self._frame(X)
        for name in ("n_neighbors", "query_block_size", "train_block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(X) or not len(X):
            raise ValueError("y must contain one class per nonempty training row")
        if pd.isna(labels).any():
            raise ValueError("Training classes must not be missing")
        codes, classes = pd.factorize(labels, sort=False)
        self.classes_ = np.asarray(classes)
        self._label_codes = np.asarray(codes, dtype=np.intp)
        self._columns = list(X.columns)
        self._numeric_columns = [c for c in X if is_numeric_dtype(X[c].dtype) and not is_bool_dtype(X[c].dtype)]
        self._categorical_columns = [c for c in X if c not in self._numeric_columns]
        self._medians, self._scales, self._category_maps = {}, {}, {}
        for col in self._numeric_columns:
            values = X[col].to_numpy(dtype=np.float64, na_value=np.nan)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            if len(finite):
                with np.errstate(over="ignore", invalid="ignore"):
                    q25, q75 = np.percentile(finite, [25, 75])
                    scale = float(q75 - q25)
            else:
                scale = 1.0
            self._medians[col] = median if np.isfinite(median) else 0.0
            self._scales[col] = scale if np.isfinite(scale) and scale > 0 else 1.0
        for col in self._categorical_columns:
            mapping = {}
            for value in X[col]:
                key = self._category(value)
                if key not in mapping:
                    mapping[key] = len(mapping)
            self._category_maps[col] = mapping
        self._train_numeric, self._train_missing, self._train_categories = self._transform(X)
        self.n_features_in_ = len(X.columns)
        self.audit_ = {
            "n_training_rows": len(X), "numeric_columns": list(self._numeric_columns),
            "categorical_columns": list(self._categorical_columns),
            "numeric_medians": dict(self._medians), "numeric_scales": dict(self._scales),
            "numeric_missing_flags": list(self._numeric_columns),
            "categorical_vocabulary_sizes": {c: len(m) for c, m in self._category_maps.items()},
            "fit_only": True, "configuration": self.configuration(),
        }
        return self

    def _transform(self, X):
        numeric = np.empty((len(self._numeric_columns), len(X)), dtype=np.float64)
        missing = np.empty_like(numeric, dtype=np.bool_)
        for j, col in enumerate(self._numeric_columns):
            values = X[col].to_numpy(dtype=np.float64, na_value=np.nan).copy()
            absent = ~np.isfinite(values)
            missing[j] = absent
            values[absent] = self._medians[col]
            with np.errstate(over="ignore", invalid="ignore"):
                values = (values - self._medians[col]) / self._scales[col]
            # Protect finite output even when finite inputs overflow arithmetic.
            numeric[j] = np.nan_to_num(values, nan=0., posinf=np.finfo(float).max / 4,
                                       neginf=-np.finfo(float).max / 4)
        categories = np.empty((len(self._categorical_columns), len(X)), dtype=np.int32)
        for j, col in enumerate(self._categorical_columns):
            mapping = self._category_maps[col]
            categories[j] = [mapping.get(self._category(value), -1) for value in X[col]]
        return numeric, missing, categories

    @staticmethod
    def _topk(distances, indices, k):
        """Select exact top k with deterministic boundary ties in linear time."""
        if distances.shape[1] > k:
            chosen = np.argpartition(distances, k - 1, axis=1)[:, :k]
            selected = np.take_along_axis(distances, chosen, axis=1)
            boundary = selected.max(axis=1)
            # Argpartition is unspecified for equal elements. Fix only rows
            # where a boundary tie extends beyond its initially chosen set.
            ties_total = np.count_nonzero(distances == boundary[:, None], axis=1)
            ties_chosen = np.count_nonzero(selected == boundary[:, None], axis=1)
            for row in np.flatnonzero(ties_total > ties_chosen):
                strict = np.flatnonzero(distances[row] < boundary[row])
                tied = np.flatnonzero(distances[row] == boundary[row])
                tied = tied[np.argsort(indices[row, tied], kind="stable")]
                chosen[row] = np.concatenate((strict, tied[:k - len(strict)]))
            distances = np.take_along_axis(distances, chosen, axis=1)
            indices = np.take_along_axis(indices, chosen, axis=1)
        order = np.lexsort((indices, distances), axis=1)
        return np.take_along_axis(distances, order, axis=1), np.take_along_axis(indices, order, axis=1)

    def _distance_block(self, query_numeric, query_missing, query_categories, start, stop):
        rows = query_numeric.shape[1] if len(self._numeric_columns) else query_categories.shape[1]
        distance = np.zeros((rows, stop - start), dtype=np.float64)
        scratch = np.empty_like(distance)
        numeric_count = len(self._numeric_columns)
        categorical_count = len(self._categorical_columns)
        components = int(numeric_count > 0) + int(categorical_count > 0)
        if numeric_count:
            weight = 1.0 / (components * 2 * numeric_count)
            for j in range(numeric_count):
                with np.errstate(over="ignore"):
                    np.subtract(query_numeric[j, :, None], self._train_numeric[j, None, start:stop], out=scratch)
                np.abs(scratch, out=scratch)
                np.minimum(scratch, 3.0, out=scratch)
                scratch += query_missing[j, :, None] != self._train_missing[j, None, start:stop]
                scratch *= weight
                distance += scratch
        if categorical_count:
            weight = 1.0 / (components * categorical_count)
            for j in range(categorical_count):
                np.not_equal(query_categories[j, :, None], self._train_categories[j, None, start:stop], out=scratch)
                scratch *= weight
                distance += scratch
        return distance

    def predict_proba(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError("Classifier must be fitted before prediction")
        X = self._frame(X)
        if len(X.columns) != len(self._columns) or set(X.columns) != set(self._columns):
            raise ValueError("Prediction columns must match the training columns")
        numeric, missing, categories = self._transform(X)
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        k = min(self.n_neighbors, len(self._label_codes))
        for query_start in range(0, len(X), self.query_block_size):
            query_stop = min(query_start + self.query_block_size, len(X))
            query_size = query_stop - query_start
            best_distances = np.empty((query_size, 0), dtype=np.float64)
            best_indices = np.empty((query_size, 0), dtype=np.intp)
            for start in range(0, len(self._label_codes), self.train_block_size):
                stop = min(start + self.train_block_size, len(self._label_codes))
                distances = self._distance_block(numeric[:, query_start:query_stop],
                                                 missing[:, query_start:query_stop],
                                                 categories[:, query_start:query_stop], start, stop)
                indices = np.broadcast_to(np.arange(start, stop), distances.shape)
                distances, indices = self._topk(distances, indices, k)
                best_distances, best_indices = self._topk(
                    np.concatenate((best_distances, distances), axis=1),
                    np.concatenate((best_indices, indices), axis=1), k)
            codes = self._label_codes[best_indices]
            for row in range(query_size):
                out[query_start + row] = np.bincount(codes[row], minlength=len(self.classes_)) / k
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
