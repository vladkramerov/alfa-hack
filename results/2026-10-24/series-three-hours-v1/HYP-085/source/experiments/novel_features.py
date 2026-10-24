"""Train-only experimental transformations for HYP-015, HYP-016 and HYP-017.

These components consume DataFrames and preserve positional row order and index.
They do not select a model, read public data or run experiments on import.
"""
import numbers

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted


def _frame(X, columns=None):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("X must be a DataFrame with unique columns")
    if columns is not None and any(c not in X.columns for c in columns):
        raise ValueError("X is missing required columns")
    return X


def _keys(series):
    # Tagged values cannot collide with a literal '__MISSING__' or 'M:'.
    return ("V:" + series.astype("string")).fillna("M:")


def _joint_keys(X, columns):
    # Length prefixes keep delimiters inside a category unambiguous.
    parts = [_keys(X[c]).map(lambda s: str(len(s)) + ":" + s) for c in columns]
    result = parts[0]
    for part in parts[1:]:
        result = result + "|" + part
    return result.astype(str)


def _remember_columns(estimator, X, output_columns):
    estimator.feature_names_in_ = np.asarray(X.columns, dtype=object)
    estimator.n_features_in_ = X.shape[1]
    estimator.output_columns_ = list(output_columns)


class GroupTargetEncoder(TransformerMixin, BaseEstimator):
    """Smoothed category means with grouped out-of-fold training encodings.

    ``fit_transform`` excludes the entire raw-X duplicate group of each row.
    ``fit``/``transform`` use full training mappings for outer validation/test.
    Missing and unseen categories use the mean of the applicable fit rows.
    Numeric columns are copied without imputation or scaling.
    """

    def __init__(self, n_splits=5, smoothing=20.0):
        self.n_splits = n_splits
        self.smoothing = smoothing

    def _validate_target(self, X, y):
        _frame(X)
        target = np.asarray(y)
        if (target.ndim != 1 or len(target) != len(X) or not len(X)
                or not pd.api.types.is_numeric_dtype(target.dtype)
                or pd.api.types.is_bool_dtype(target.dtype)
                or np.iscomplexobj(target)):
            raise ValueError("y must be a nonempty one-dimensional numeric regression target")
        target = target.astype(float)
        if not np.isfinite(target).all():
            raise ValueError("y must contain only finite values")
        if (not isinstance(self.n_splits, numbers.Integral) or isinstance(self.n_splits, bool)
                or self.n_splits < 2):
            raise ValueError("n_splits must be an integer >= 2")
        if (not isinstance(self.smoothing, numbers.Real) or not np.isfinite(self.smoothing)
                or self.smoothing < 0):
            raise ValueError("smoothing must be finite and nonnegative")
        return target

    def _mapping(self, keys, target, prior):
        known = keys.ne("M:").to_numpy()
        stats = pd.DataFrame({"key": keys.to_numpy()[known], "target": target[known]})
        stats = stats.groupby("key", sort=False)["target"].agg(["sum", "count"])
        return ((stats["sum"] + self.smoothing * prior)
                / (stats["count"] + self.smoothing)).to_dict()

    def fit(self, X, y):
        target = self._validate_target(X, y)
        self.numeric_ = X.select_dtypes(include="number").columns.tolist()
        self.categorical_ = [c for c in X if c not in self.numeric_]
        output = self.numeric_ + [c + "__target_mean" for c in self.categorical_]
        if len(set(output)) != len(output):
            raise ValueError("Generated target feature names collide with numeric inputs")
        _remember_columns(self, X, output)
        self.prior_ = float(target.mean())
        self.mappings_ = {c: self._mapping(_keys(X[c]), target, self.prior_)
                          for c in self.categorical_}
        self.audit_ = {"numeric_columns": list(self.numeric_),
                       "categorical_columns": list(self.categorical_),
                       "smoothing": float(self.smoothing), "n_splits": int(self.n_splits),
                       "n_fit": len(X), "folds": [], "no_group_overlap": None,
                       "no_self_target_leakage": False}
        for attribute in ("fold_assignment_", "fold_indices_"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            raise ValueError("GroupTargetEncoder does not consume fit metadata")
        target = self._validate_target(X, y)
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < self.n_splits:
            raise ValueError("Grouped target encoding needs at least n_splits distinct raw-X groups")
        self.fit(X, target)
        output = X[self.numeric_].copy()
        encoded = {c: np.empty(len(X), dtype=float) for c in self.categorical_}
        keys = {c: _keys(X[c]) for c in self.categorical_}
        self.fold_assignment_ = np.full(len(X), -1, dtype=int)
        self.fold_indices_ = []
        folds = []
        for fold, (train, valid) in enumerate(GroupKFold(self.n_splits).split(X, target, groups)):
            prior = float(target[train].mean())
            train_groups, valid_groups = np.unique(groups[train]), np.unique(groups[valid])
            overlap = int(np.intersect1d(train_groups, valid_groups).size)
            row_overlap = int(np.intersect1d(train, valid).size)
            if overlap or row_overlap:
                raise RuntimeError("Target encoding folds overlap")
            for c in self.categorical_:
                mapping = self._mapping(keys[c].iloc[train], target[train], prior)
                encoded[c][valid] = keys[c].iloc[valid].map(mapping).fillna(prior).to_numpy(float)
            self.fold_assignment_[valid] = fold
            self.fold_indices_.append((train.copy(), valid.copy()))
            folds.append({"fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                          "train_groups": len(train_groups), "valid_groups": len(valid_groups),
                          "group_overlap": overlap, "row_overlap": row_overlap, "prior": prior})
        for c in self.categorical_:
            output[c + "__target_mean"] = encoded[c]
        self.audit_.update({"n_groups": n_groups, "folds": folds, "no_group_overlap": True,
                            "no_self_target_leakage": bool((self.fold_assignment_ >= 0).all())})
        return output

    def transform(self, X):
        check_is_fitted(self, "mappings_")
        _frame(X, self.feature_names_in_)
        output = X[self.numeric_].copy()
        for c in self.categorical_:
            output[c + "__target_mean"] = _keys(X[c]).map(self.mappings_[c]).fillna(self.prior_).to_numpy(float)
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


class CrossFeatureTransformer(TransformerMixin, BaseEstimator):
    """Append a collision-safe string joint category; learn no category state."""

    def __init__(self, columns=("proto", "service", "state"), output_column="proto_service_state"):
        self.columns = columns
        self.output_column = output_column

    def _validate(self, X):
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be nonempty and distinct")
        _frame(X, self.columns)
        if self.output_column in X:
            raise ValueError("Joint feature name collides with an input column")

    def fit(self, X, y=None):
        self._validate(X)
        _remember_columns(self, X, list(X.columns) + [self.output_column])
        return self

    def transform(self, X):
        self._validate(X)
        output = X.copy()
        output[self.output_column] = _joint_keys(X, self.columns)
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


class FrequencyFeatureTransformer(TransformerMixin, BaseEstimator):
    """Append individual and joint count/n_fit features without target access."""

    def __init__(self, columns=("proto", "service", "state"), joint_name="proto_service_state"):
        self.columns = columns
        self.joint_name = joint_name

    def _all_keys(self, X):
        _frame(X, self.columns)
        keys = {c: _keys(X[c]) for c in self.columns}
        keys[self.joint_name] = _joint_keys(X, self.columns)
        return keys

    def fit(self, X, y=None):
        if (not self.columns or len(set(self.columns)) != len(self.columns)
                or self.joint_name in self.columns):
            raise ValueError("Frequency source names must be nonempty and distinct")
        keys = self._all_keys(X)
        if not len(X):
            raise ValueError("Cannot fit frequencies on empty data")
        additions = [c + "__frequency" for c in keys]
        if any(c in X for c in additions):
            raise ValueError("Frequency feature name collides with an input column")
        self.n_fit_ = len(X)
        self.frequency_maps_ = {c: (key.value_counts(dropna=False) / self.n_fit_).to_dict()
                                for c, key in keys.items()}
        _remember_columns(self, X, list(X.columns) + additions)
        self.audit_ = {"n_fit": self.n_fit_, "source_columns": list(self.columns),
                       "features": additions, "uses_target": False}
        return self

    def transform(self, X):
        check_is_fitted(self, "frequency_maps_")
        _frame(X, self.feature_names_in_)
        output = X.copy()
        for c, key in self._all_keys(X).items():
            output[c + "__frequency"] = key.map(self.frequency_maps_[c]).fillna(0.0).to_numpy(float)
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)
