"""Deterministic train-only transformations for the two tabular experiments.

Inputs must omit identifiers and targets. Domain features use only a row's
predictors; monthly spread is max minus min of monthly average temperatures.
Packet denominators are clipped to one, so an empty flow has finite ratios.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted


class DataPreprocessor(TransformerMixin, BaseEstimator):
    """Fit numeric/category state on training data and return dense float32.

    The target is accepted for Pipeline compatibility and never inspected.
    Correlation pruning retains the first column in input order and compares
    each later numeric column only with retained columns, before imputation.
    """

    def __init__(self, task, drop_correlated=False, scale=False, gaussian=False,
                 engineer=False, correlation_threshold=0.95, drop_vif=False,
                 vif_threshold=10.0):
        self.task = task
        self.drop_correlated = drop_correlated
        self.scale = scale
        self.gaussian = gaussian
        self.engineer = engineer
        self.correlation_threshold = correlation_threshold
        self.drop_vif = drop_vif
        self.vif_threshold = vif_threshold

    @staticmethod
    def _validate_frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame without identifier or target columns")
        if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
            raise ValueError("Feature names must be unique strings")
        forbidden = {"id", "site_eui", "attack_cat", "label", "target"}
        present = sorted(forbidden.intersection(c.lower() for c in X.columns))
        if present:
            raise ValueError(f"Remove identifier/target columns before preprocessing: {present}")

    def _engineer_frame(self, X):
        frame = X.copy(deep=True)
        for column in frame.select_dtypes(include="number").columns:
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
        if not self.engineer:
            return frame
        if self.task == "regression":
            if "year_built" in frame:
                frame["year_built"] = frame["year_built"].replace(0, np.nan)
            monthly = [c for c in frame if c.endswith(("_avg_temp_c", "_avg_temp"))]
            if monthly:
                frame["monthly_temp_mean"] = frame[monthly].mean(axis=1)
                frame["monthly_temp_spread"] = frame[monthly].max(axis=1) - frame[monthly].min(axis=1)
            for suffix in ("_c", ""):
                heat, cool = "heating_degree_days" + suffix, "cooling_degree_days" + suffix
                if heat in frame and cool in frame:
                    frame["heating_cooling_total"] = frame[heat] + frame[cool]
                    break
            if "floor_area" in frame:
                frame["floor_area_log1p"] = np.log1p(frame["floor_area"].clip(lower=0))
        else:
            for byte_col, packet_col, output in (
                ("sbytes", "spkts", "src_bytes_per_packet"),
                ("dbytes", "dpkts", "dst_bytes_per_packet"),
                ("src_bytes", "src_packets", "src_bytes_per_packet"),
                ("dst_bytes", "dst_packets", "dst_bytes_per_packet"),
            ):
                if byte_col in frame and packet_col in frame:
                    frame[output] = frame[byte_col] / frame[packet_col].clip(lower=1)
            for column in ("sbytes", "dbytes", "spkts", "dpkts", "src_bytes",
                           "dst_bytes", "src_packets", "dst_packets"):
                if column in frame:
                    frame[column + "_log1p"] = np.log1p(frame[column].clip(lower=0))
        return frame.replace([np.inf, -np.inf], np.nan)

    def _with_missing_flags(self, frame):
        for column in self.missing_flag_columns_:
            frame[column + "_missing"] = frame[column].isna().astype(float)
        return frame

    @staticmethod
    def _skewness(matrix, names):
        result = {}
        for i, column in enumerate(names):
            values = matrix[:, i]
            value = float(pd.Series(values).skew()) if np.unique(values).size > 1 else 0.0
            result[column] = value if np.isfinite(value) else 0.0
        return result

    def _categoricals(self, frame):
        # A string conversion avoids mixed category types while retaining NaNs
        # until the sentinel has been inserted.
        return frame[self.categorical_columns_].astype("string").fillna(self.missing_sentinel_).astype(object)

    @staticmethod
    def _vif_values(matrix, names):
        """VIF from a correlation eigendecomposition, including exact dependencies.

        A pseudoinverse alone would hide singular dependencies. Predictors with
        nonzero participation in the numerical nullspace receive infinite VIF.
        Constants receive infinity and are removed by the optional pruning step.
        """
        if not names:
            return {}, set()
        constant_mask = np.ptp(matrix, axis=0) == 0
        constants = {name for name, is_constant in zip(names, constant_mask) if is_constant}
        values = {name: float("inf") for name in constants}
        variable_indices = np.flatnonzero(~constant_mask)
        if variable_indices.size:
            standardized = StandardScaler().fit_transform(matrix[:, variable_indices])
            correlation = standardized.T @ standardized / len(standardized)
            correlation = (correlation + correlation.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            tolerance = max(1e-12, np.finfo(float).eps * len(variable_indices)
                            * max(float(eigenvalues[-1]), 1.0) * 10)
            positive = eigenvalues > tolerance
            null_participation = np.sum(eigenvectors[:, ~positive] ** 2, axis=1)
            finite_diagonal = np.sum(eigenvectors[:, positive] ** 2 / eigenvalues[positive], axis=1)
            for i, source_index in enumerate(variable_indices):
                values[names[source_index]] = (float("inf") if null_participation[i] > 1e-10
                                                else max(1.0, float(finite_diagonal[i])))
        return {name: values[name] for name in names}, constants

    @staticmethod
    def _json_vif(values):
        return {name: value if np.isfinite(value) else "infinity" for name, value in values.items()}

    def fit(self, X, y=None):
        self._validate_frame(X)
        if self.task not in {"regression", "multiclass", "classification"}:
            raise ValueError("task must be regression, multiclass, or classification")
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if not np.isfinite(self.vif_threshold) or self.vif_threshold < 1:
            raise ValueError("vif_threshold must be finite and at least one")
        if len(X) == 0:
            raise ValueError("Cannot fit preprocessing on an empty training set")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        frame = self._engineer_frame(X)
        self.missing_flag_columns_ = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])
                                      and frame[c].isna().any()] if self.engineer else []
        frame = self._with_missing_flags(frame)
        all_numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical_columns_ = [c for c in frame if c not in all_numeric]
        self.numeric_columns_ = []
        dropped = []
        if self.drop_correlated and all_numeric:
            correlations = {method: frame[all_numeric].corr(method=method)
                            for method in ("pearson", "spearman")}
            for column in all_numeric:
                match = None
                for retained in self.numeric_columns_:
                    for method, matrix in correlations.items():
                        coefficient = matrix.loc[column, retained]
                        if np.isfinite(coefficient) and abs(coefficient) >= self.correlation_threshold:
                            match = {"column": column, "correlated_with": retained,
                                     "coefficient": float(coefficient), "method": method}
                            break
                    if match:
                        break
                if match:
                    dropped.append(match)
                else:
                    self.numeric_columns_.append(column)
        else:
            self.numeric_columns_ = all_numeric
        vif_matrix = np.empty((len(frame), 0), dtype=float)
        if self.numeric_columns_:
            vif_matrix = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
                frame[self.numeric_columns_])
        vif_before, constants = self._vif_values(vif_matrix, self.numeric_columns_)
        vif_after = vif_before
        vif_dropped = []
        while self.drop_vif and self.numeric_columns_:
            largest = max(vif_after.values())
            if np.isfinite(largest) and largest <= self.vif_threshold:
                break
            # eigendecomposition can produce tiny rounding differences for equal
            # finite VIFs; treat numerical ties consistently with infinite ties.
            tied = [i for i, column in enumerate(self.numeric_columns_)
                    if (np.isinf(largest) and np.isinf(vif_after[column]))
                    or (np.isfinite(largest)
                        and np.isclose(vif_after[column], largest, rtol=1e-10, atol=1e-12))]
            index = tied[-1]
            column = self.numeric_columns_[index]
            vif_dropped.append({
                "column": column,
                "vif": largest if np.isfinite(largest) else "infinity",
                "reason": "constant" if column in constants else (
                    "singular_dependency" if np.isinf(largest) else "vif_above_threshold"),
            })
            self.numeric_columns_.pop(index)
            vif_matrix = np.delete(vif_matrix, index, axis=1)
            vif_after, constants = self._vif_values(vif_matrix, self.numeric_columns_)
        residual = {}
        for method in ("pearson", "spearman"):
            matrix = (correlations[method].loc[self.numeric_columns_, self.numeric_columns_]
                      if self.drop_correlated and all_numeric
                      else frame[self.numeric_columns_].corr(method=method)).to_numpy()
            upper = np.abs(matrix[np.triu_indices_from(matrix, k=1)])
            finite = upper[np.isfinite(upper)]
            residual[method] = float(finite.max()) if finite.size else 0.0
        self.numeric_imputer_ = None
        numeric = np.empty((len(frame), 0), dtype=float)
        if self.numeric_columns_:
            self.numeric_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            numeric = self.numeric_imputer_.fit_transform(frame[self.numeric_columns_])
        before = self._skewness(numeric, self.numeric_columns_)
        self.power_transformers_ = {}
        skipped = {}
        if self.gaussian:
            for i, column in enumerate(self.numeric_columns_):
                if np.unique(numeric[:, i]).size <= 20 or abs(before[column]) <= 1:
                    continue
                try:
                    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                    with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
                        warnings.simplefilter("error", RuntimeWarning)
                        values = transformer.fit_transform(numeric[:, [i]])
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite Yeo-Johnson training output")
                    numeric[:, i] = values[:, 0]
                    self.power_transformers_[column] = transformer
                except (ValueError, RuntimeError, FloatingPointError, RuntimeWarning) as exc:
                    skipped[column] = type(exc).__name__
        after = self._skewness(numeric, self.numeric_columns_)
        self.scaler_ = StandardScaler().fit(numeric) if self.scale and numeric.shape[1] else None
        self.missing_sentinel_ = "__MISSING__"
        if self.categorical_columns_:
            observed = set(frame[self.categorical_columns_].astype("string").stack().tolist())
            while self.missing_sentinel_ in observed:
                self.missing_sentinel_ += "_"
        self.encoder_ = None
        categorical_names = []
        if self.categorical_columns_:
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            self.encoder_.fit(self._categoricals(frame))
            categorical_names = list(self.encoder_.get_feature_names_out(self.categorical_columns_))
        self.transformed_feature_names_ = np.asarray(self.numeric_columns_ + categorical_names, dtype=object)
        self.audit_ = {
            "numeric_columns": list(self.numeric_columns_),
            "categorical_columns": list(self.categorical_columns_),
            "dropped_columns": dropped,
            "residual_max_abs_correlation": residual,
            "vif_dropped_columns": vif_dropped,
            "numeric_vif_before": self._json_vif(vif_before),
            "numeric_vif_after": self._json_vif(vif_after),
            "missing_flag_columns": list(self.missing_flag_columns_),
            "skewness_before": before,
            "skewness_after": after,
            "gaussian_columns": list(self.power_transformers_),
            "gaussian_lambdas": {c: float(t.lambdas_[0]) for c, t in self.power_transformers_.items()},
            "gaussian_skipped": skipped,
            "feature_names": self.transformed_feature_names_.tolist(),
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "audit_")
        self._validate_frame(X)
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing fitted feature columns: {sorted(missing)}")
        frame = self._with_missing_flags(self._engineer_frame(X.loc[:, self.feature_names_in_]))
        pieces = []
        if self.numeric_columns_:
            numeric = self.numeric_imputer_.transform(frame[self.numeric_columns_])
            for column, transformer in self.power_transformers_.items():
                i = self.numeric_columns_.index(column)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    numeric[:, i] = transformer.transform(numeric[:, [i]])[:, 0]
            if self.scaler_ is not None:
                # Replace numerical overflow from pathological validation rows
                # before StandardScaler's finite-input validation.
                numeric = np.nan_to_num(numeric, nan=0.0, posinf=1e30, neginf=-1e30)
                numeric = self.scaler_.transform(numeric)
            pieces.append(numeric)
        if self.encoder_ is not None:
            pieces.append(self.encoder_.transform(self._categoricals(frame)))
        output = np.concatenate(pieces, axis=1) if pieces else np.empty((len(frame), 0))
        limit = np.finfo(np.float32).max
        output = np.nan_to_num(output, nan=0.0, posinf=limit, neginf=-limit)
        return np.clip(output, -limit, limit).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "audit_")
        if input_features is not None and not np.array_equal(input_features, self.feature_names_in_):
            raise ValueError("input_features must match the fitted feature names")
        return self.transformed_feature_names_.copy()
