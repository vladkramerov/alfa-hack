"""Portable mandatory full-v1 preprocessing and training-only outlier filtering.

Callers must exclude identifiers and targets before supplying X. Neither
component infers target columns or learns anything from transform batches.
"""
from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted


MAX_DENSE_CELLS = 32_000_000
MAX_ONEHOT_CATEGORIES = 64
FLOAT32_LIMIT = float(np.finfo(np.float32).max)


def _frame(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame without identifiers or target")
    if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
        raise ValueError("Feature names must be unique strings")
    return X


def _numeric(X, columns):
    values = X[columns].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _category_values(X, columns):
    # Prefixing every real value keeps literal missing markers distinct.
    output = pd.DataFrame(index=X.index)
    for column in columns:
        output[column] = ("v:" + X[column].astype("string")).fillna("m:").astype(object)
    return output


def _finite(values):
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=FLOAT32_LIMIT,
                                 neginf=-FLOAT32_LIMIT), -FLOAT32_LIMIT, FLOAT32_LIMIT)


def _check_dense(rows, columns):
    if rows * columns > MAX_DENSE_CELLS:
        raise MemoryError("Dense preprocessing output exceeds 32 million cells; use bounded batches or native categories")


def _skewness(values, columns):
    output = {}
    for i, column in enumerate(columns):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = float(pd.Series(values[:, i]).skew()) if len(values) > 2 else 0.0
        output[column] = value if np.isfinite(value) else 0.0
    return output


class FullPreprocessor(TransformerMixin, BaseEstimator):
    """Fit median/quantiles/scaler/categories on training only.

    Native output is a DataFrame preserving X.index. Other models receive a
    finite dense float32 array. One-hot category count and dense allocations
    are bounded. Empty numeric input falls back to a zero-valued bias feature.
    """

    def __init__(self, native=False):
        self.native = native

    def fit(self, X, y=None):
        _frame(X)
        if not len(X):
            raise ValueError("Cannot fit preprocessing on empty training data")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.raw_numeric_ = X.select_dtypes(include="number").columns.tolist()
        self.raw_categorical_ = [c for c in X if c not in self.raw_numeric_]
        raw = _numeric(X, self.raw_numeric_)
        missing_counts = {c: int(np.isnan(raw[:, i]).sum()) for i, c in enumerate(self.raw_numeric_)}
        kept = []
        dropped = []
        for i, column in enumerate(self.raw_numeric_):
            finite = raw[np.isfinite(raw[:, i]), i]
            if not finite.size:
                dropped.append({"column": column, "reason": "all_null"})
            elif np.unique(finite).size <= 1:
                dropped.append({"column": column, "reason": "constant"})
            else:
                duplicate = next((j for j in kept if np.array_equal(raw[:, i], raw[:, j], equal_nan=True)), None)
                if duplicate is not None:
                    dropped.append({"column": column, "reason": "exact_duplicate",
                                    "duplicate_of": self.raw_numeric_[duplicate]})
                else:
                    kept.append(i)
        self.numeric_ = [self.raw_numeric_[i] for i in kept]
        raw_categories = _category_values(X, self.raw_categorical_)
        self.categorical_ = []
        for column in self.raw_categorical_:
            keys = raw_categories[column]
            if keys.eq("m:").all():
                dropped.append({"column": column, "reason": "all_null"})
            elif keys.nunique() <= 1:
                dropped.append({"column": column, "reason": "constant"})
            else:
                duplicate = next((c for c in self.categorical_
                                  if np.array_equal(keys.to_numpy(), raw_categories[c].to_numpy())), None)
                if duplicate is not None:
                    dropped.append({"column": column, "reason": "exact_duplicate", "duplicate_of": duplicate})
                else:
                    self.categorical_.append(column)
        # Retain variable missingness even if the observed numeric value is constant.
        duplicate_columns = {d["column"] for d in dropped if d["reason"] == "exact_duplicate"}
        self.missing_flag_columns_ = [c for c in self.raw_numeric_
                                      if 0 < missing_counts[c] < len(X) and c not in duplicate_columns]
        self.missing_flag_names_ = [c + "__missing" for c in self.missing_flag_columns_]
        if set(self.missing_flag_names_).intersection(X.columns):
            raise ValueError("Generated missing indicator names collide with input columns")
        self.numeric_output_columns_ = self.numeric_ + self.missing_flag_names_
        self.fallback_ = not self.numeric_output_columns_ and not self.categorical_
        if self.fallback_:
            self.numeric_output_columns_ = ["__full_v1_bias"]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True) if self.numeric_ else None
        if self.numeric_:
            values = self.imputer_.fit_transform(raw[:, kept])
        else:
            values = np.empty((len(X), 0), dtype=float)
        self.continuous_columns_ = []
        self.continuous_indices_ = []
        for i, column in enumerate(self.numeric_):
            _, counts = np.unique(values[:, i], return_counts=True)
            if len(counts) >= 20 and counts.max() / len(X) <= .8:
                self.continuous_columns_.append(column)
                self.continuous_indices_.append(i)
        self.quantile_ = None
        before = values.copy()
        if self.continuous_indices_:
            self.quantile_ = QuantileTransformer(output_distribution="normal",
                                                 n_quantiles=min(512, len(X)),
                                                 subsample=None, random_state=42)
            values[:, self.continuous_indices_] = self.quantile_.fit_transform(values[:, self.continuous_indices_])
        flags = self._missing_flags(X)
        before = np.column_stack([before, flags])
        values = np.column_stack([values, flags])
        if self.fallback_:
            before = values = np.zeros((len(X), 1), dtype=float)
        self.scaler_ = StandardScaler().fit(values) if values.shape[1] else None
        scaled = self.scaler_.transform(values) if self.scaler_ is not None else values
        if not np.isfinite(scaled).all():
            raise ValueError("Nonfinite training statistics; numeric magnitudes exceed supported range")
        categories = _category_values(X, self.categorical_)
        self.levels_ = {c: sorted(set(categories[c]).union({"m:"})) for c in self.categorical_}
        self.onehot_retained_ = {}
        self.ohe_ = None
        category_names = []
        if self.categorical_:
            if self.native:
                category_names = list(self.categorical_)
            else:
                onehot_levels = []
                for c in self.categorical_:
                    levels = self.levels_[c]
                    if len(levels) > MAX_ONEHOT_CATEGORIES:
                        counts = categories[c].value_counts().to_dict()
                        frequent = sorted((v for v in levels if v != "m:"),
                                          key=lambda v: (-counts.get(v, 0), v))[:MAX_ONEHOT_CATEGORIES - 2]
                        self.onehot_retained_[c] = set(frequent + ["m:"])
                        levels = sorted(frequent + ["m:", "r:"])
                    onehot_levels.append(levels)
                self.ohe_ = OneHotEncoder(categories=onehot_levels,
                                          handle_unknown="ignore",
                                          sparse_output=False, dtype=np.float32)
                self.ohe_.fit(self._onehot_values(categories))
                category_names = self.ohe_.get_feature_names_out(self.categorical_).tolist()
        self.output_columns_ = self.numeric_output_columns_ + category_names
        if not self.native:
            _check_dense(len(X), len(self.output_columns_))
        self.audit_ = {"policy": "full-v1", "n_fit": len(X), "normalization_enabled": True,
                       "numeric_columns": list(self.numeric_), "categorical_columns": list(self.categorical_),
                       "numeric_output_columns": list(self.numeric_output_columns_), "dropped_columns": dropped,
                       "continuous_columns": list(self.continuous_columns_),
                       "missing_counts": missing_counts, "missing_flag_columns": list(self.missing_flag_columns_),
                       "scaler_mean": self.scaler_.mean_.tolist() if self.scaler_ is not None else [],
                       "scaler_scale": self.scaler_.scale_.tolist() if self.scaler_ is not None else [],
                       "skewness_before": _skewness(before, self.numeric_output_columns_),
                       "skewness_after": _skewness(scaled, self.numeric_output_columns_),
                       "quantile_parameters": ({"output_distribution": "normal", "n_quantiles": min(512, len(X)),
                                                "subsample": None, "random_state": 42} if self.quantile_ is not None else None),
                       "native_categories": bool(self.native), "onehot_max_categories": MAX_ONEHOT_CATEGORIES,
                       "max_dense_cells": MAX_DENSE_CELLS, "fallback_bias": self.fallback_}
        return self

    def _missing_flags(self, X):
        if not self.missing_flag_columns_:
            return np.empty((len(X), 0), dtype=float)
        return np.isnan(_numeric(X, self.missing_flag_columns_)).astype(float)

    def _onehot_values(self, categories):
        output = categories.copy()
        for c, retained in self.onehot_retained_.items():
            rare_known = output[c].isin(self.levels_[c]) & ~output[c].isin(retained)
            output.loc[rare_known, c] = "r:"
        return output

    def transform(self, X):
        check_is_fitted(self, "output_columns_")
        _frame(X)
        if any(c not in X for c in self.feature_names_in_):
            raise ValueError("Transform data is missing training columns")
        if not self.native:
            _check_dense(len(X), len(self.output_columns_))
        if not len(X):
            if self.native:
                out = pd.DataFrame(index=X.index, columns=self.numeric_output_columns_, dtype=float)
                for c in self.categorical_:
                    out[c] = pd.Categorical([], categories=self.levels_[c] + ["u:"])
                return out
            return np.empty((0, len(self.output_columns_)), dtype=np.float32)
        values = self.imputer_.transform(_numeric(X, self.numeric_)) if self.numeric_ else np.empty((len(X), 0))
        if self.quantile_ is not None:
            values[:, self.continuous_indices_] = self.quantile_.transform(values[:, self.continuous_indices_])
        values = np.column_stack([values, self._missing_flags(X)])
        if self.fallback_:
            values = np.zeros((len(X), 1), dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            values = self.scaler_.transform(values) if self.scaler_ is not None else values
        values = _finite(values)
        categories = _category_values(X, self.categorical_)
        if self.native:
            output = pd.DataFrame(values, index=X.index, columns=self.numeric_output_columns_)
            for c in self.categorical_:
                known = categories[c].where(categories[c].isin(self.levels_[c]), "u:")
                output[c] = pd.Categorical(known, categories=self.levels_[c] + ["u:"])
            return output
        if self.ohe_ is not None:
            values = np.column_stack([values, self.ohe_.transform(self._onehot_values(categories))])
        return values.astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


class TrainingOutlierFilter(BaseEstimator):
    """IsolationForest candidates with independent global/per-class hard caps.

    Detection sees numeric X only. A classification label only restricts which
    negative-score candidates may be removed; regression y is never inspected.
    ``fit_filter`` is exclusively a training operation, never a transform.
    """

    def __init__(self, max_fraction=.05, random_state=42, n_jobs=4):
        self.max_fraction = max_fraction
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_filter(self, X, y, classification=False):
        _frame(X)
        target = np.asarray(y)
        if not len(X) or target.ndim != 1 or len(target) != len(X):
            raise ValueError("Training X and one-dimensional y must be nonempty and aligned")
        if (not isinstance(self.max_fraction, numbers.Real) or not np.isfinite(self.max_fraction)
                or not 0 <= self.max_fraction <= .05):
            raise ValueError("max_fraction must be finite and in [0, .05]")
        if (not isinstance(self.n_jobs, numbers.Integral) or isinstance(self.n_jobs, bool)
                or not 1 <= self.n_jobs <= 4):
            raise ValueError("n_jobs must be an integer between 1 and 4")
        n = len(X)
        cap = int(np.floor(self.max_fraction * n))
        self.keep_mask_ = np.ones(n, dtype=bool)
        self.scores_ = np.zeros(n, dtype=float)
        self.removed_positions_ = np.empty(0, dtype=int)
        self.numeric_ = X.select_dtypes(include="number").columns.tolist()
        self.detector_ = None
        self.detector_imputer_ = None
        raw = _numeric(X, self.numeric_)
        varying = [i for i in range(raw.shape[1]) if np.unique(raw[np.isfinite(raw[:, i]), i]).size > 1]
        parameters = {"n_estimators": 100, "max_samples": min(256, n), "contamination": "auto",
                      "random_state": self.random_state, "n_jobs": int(self.n_jobs)}
        codes = None
        labels = []
        counts = np.empty(0, dtype=int)
        class_caps = np.empty(0, dtype=int)
        class_removed = np.empty(0, dtype=int)
        if classification:
            if pd.isna(target).any():
                raise ValueError("Classification targets must not be missing")
            codes, labels = pd.factorize(target, sort=False)
            counts = np.bincount(codes)
            class_caps = np.minimum(np.floor(self.max_fraction * counts).astype(int),
                                    counts - np.minimum(50, counts))
            class_removed = np.zeros(len(counts), dtype=int)
        if not varying:
            reason = "no_varying_numeric_features"
        elif not cap:
            reason = "zero_removal_budget"
        else:
            # All-null numeric columns receive the separately fitted detector's zero fill.
            self.detector_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            values = self.detector_imputer_.fit_transform(raw)
            self.detector_ = IsolationForest(**parameters).fit(values)
            self.scores_ = self.detector_.decision_function(values)
            removed = []
            for position in np.argsort(self.scores_, kind="stable"):
                if self.scores_[position] >= 0 or len(removed) >= cap:
                    break
                if classification and class_removed[codes[position]] >= class_caps[codes[position]]:
                    continue
                removed.append(int(position))
                if classification:
                    class_removed[codes[position]] += 1
            self.removed_positions_ = np.asarray(removed, dtype=int)
            self.keep_mask_[self.removed_positions_] = False
            reason = "filtered_negative_scores" if removed else "no_eligible_negative_scores"
        classes = {str(label): {"before": int(counts[i]), "removed": int(class_removed[i]),
                                "retained": int(counts[i] - class_removed[i]),
                                "max_removed": int(class_caps[i]),
                                "removed_fraction": float(class_removed[i] / counts[i])}
                   for i, label in enumerate(labels)}
        self.audit_ = {"n_input": n, "n_retained": int(self.keep_mask_.sum()),
                       "n_removed": int((~self.keep_mask_).sum()),
                       "removed_fraction": float((~self.keep_mask_).sum() / n),
                       "max_fraction": float(self.max_fraction), "max_removed": cap,
                       "classification": bool(classification), "classes": classes,
                       "reason": reason, "numeric_columns": list(self.numeric_),
                       "detector_numeric_columns": list(self.numeric_) if self.detector_ is not None else [],
                       "negative_score_candidates": int((self.scores_ < 0).sum()),
                       "detector_parameters": parameters, "target_used_in_detector": False}
        positions = np.flatnonzero(self.keep_mask_)
        y_retained = y.iloc[positions].copy() if isinstance(y, (pd.Series, pd.DataFrame)) else target[positions].copy()
        return X.iloc[positions].copy(), y_retained, self.keep_mask_.copy()
