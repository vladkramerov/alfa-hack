"""Independent sparse second-order factorization regression for SPEC010.

Preprocessing is learned exclusively by ``fit``. Categorical columns receive
lexical one-hot codes (unknown values have no active code); numeric columns
receive median imputation, standardization, and original missing indicators.
No dense sample-by-encoded-feature or sample-by-sample matrix is allocated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _fm_prediction(X, bias, linear, factors, squared=None, projected=None):
    """Compute all distinct feature pairs without materializing those pairs."""
    if squared is None:
        squared = X.multiply(X)
    if projected is None:
        projected = X @ factors
    interactions = .5 * np.sum(projected * projected - squared @ (factors * factors), axis=1)
    return bias + np.asarray(X @ linear).reshape(-1) + interactions


def fm_loss_gradient(X, y, bias, linear, factors, l2=0.001):
    """Return ``(loss, d_bias, d_linear, d_factors)`` for one minibatch.

    ``X`` is a finite CSR matrix of shape (n, p), ``y`` a finite vector (n,),
    ``bias`` a scalar, ``linear`` a vector (p,), and ``factors`` an array
    (p, rank). Dense X is also accepted and converted to CSR. The objective is
    ``0.5 * mean((FM(X)-y)**2) + 0.5*l2*(sum(linear**2)+sum(factors**2))``;
    the intercept is unpenalized. Target scaling is the caller's concern.
    Gradients are exact, including L2 on every parameter in each minibatch.
    The only dense feature-shaped arrays are model parameters and gradients.
    """
    X = sparse.csr_matrix(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    factors = np.asarray(factors, dtype=np.float64)
    if (X.shape[0] == 0 or y.shape != (X.shape[0],)
            or linear.shape != (X.shape[1],) or factors.ndim != 2
            or factors.shape[0] != X.shape[1]):
        raise ValueError("Incompatible FM parameter, target, or matrix shapes")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    squared = X.multiply(X)
    projected = X @ factors
    residual = _fm_prediction(X, bias, linear, factors, squared, projected) - y
    count = X.shape[0]
    loss = .5 * float(residual @ residual) / count
    loss += .5 * l2 * float(np.sum(linear * linear) + np.sum(factors * factors))
    d_bias = float(residual.mean())
    d_linear = np.asarray(X.T @ residual).reshape(-1) / count + l2 * linear
    d_factors = np.asarray(X.T @ (residual[:, None] * projected)) / count
    correction = np.asarray(squared.T @ residual).reshape(-1) / count
    d_factors -= correction[:, None] * factors
    d_factors += l2 * factors
    return loss, d_bias, d_linear, d_factors


def _category_key(value):
    # A prefix makes missing distinct from every literal string, including "m:".
    return "m:" if pd.isna(value) else "v:" + str(value)


class FactorizationRegressor:
    """Second-order FM with deterministic shuffled minibatch Adam.

    L2 acts in standardized target units. Predictions undo the affine target
    transform and remain in the input target's original units. Dtype determines
    numeric versus categorical treatment; strings such as ``"001"`` stay lexical.
    """

    def __init__(self, rank=8, epochs=20, batch_size=512, learning_rate=.005,
                 l2=.001, random_state=42):
        self.rank = rank
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.random_state = random_state

    def configuration(self):
        """JSON-serializable recipe, independent of any fitted data."""
        return {
            "family": "factorization_machine", "order": 2,
            "rank": int(self.rank), "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate), "l2": float(self.l2),
            "random_state": int(self.random_state), "factor_init_std": .01,
            "optimizer": "Adam", "adam_beta1": .9, "adam_beta2": .999,
            "adam_epsilon": 1e-8,
            "loss": "half_mean_squared_error_plus_half_l2",
            "numeric": "train_median_standardization_and_missing_flags",
            "categorical": "train_lexical_one_hot_unknown_ignore",
            "target": "train_mean_std_inverse_prediction",
        }

    def _validate_parameters(self):
        for name in ("rank", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "l2"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"{name} is outside the supported range")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(self.random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer")

    def _check_frame(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("X must be a DataFrame with unique columns")
        if hasattr(self, "columns_") and (len(X.columns) != len(self.columns_)
                                          or set(X.columns) != set(self.columns_)):
            raise ValueError("Prediction columns differ from fitted columns")

    def _numeric_array(self, X):
        return X.loc[:, self.numeric_columns_].to_numpy(dtype=np.float64, na_value=np.nan)

    def _fit_preprocessing(self, X):
        self.columns_ = list(X.columns)
        self.numeric_columns_ = [column for column in X.columns
                                 if pd.api.types.is_numeric_dtype(X[column].dtype)]
        self.categorical_columns_ = [column for column in X.columns
                                     if column not in self.numeric_columns_]
        numeric = self._numeric_array(X)
        finite = np.isfinite(numeric)
        self.numeric_medians_ = np.array([
            np.median(numeric[finite[:, i], i]) if finite[:, i].any() else 0.
            for i in range(numeric.shape[1])
        ])
        imputed = np.where(finite, numeric, self.numeric_medians_)
        self.numeric_mean_ = imputed.mean(axis=0)
        self.numeric_scale_ = imputed.std(axis=0)
        self.numeric_scale_[self.numeric_scale_ == 0] = 1.
        self.category_maps_ = []
        for column in self.categorical_columns_:
            keys = sorted({_category_key(value) for value in X[column].array})
            self.category_maps_.append({key: index for index, key in enumerate(keys)})

    def _transform(self, X):
        """Apply frozen preprocessing; numeric values precede missing flags, then OHE."""
        self._check_frame(X)
        numeric = self._numeric_array(X)
        finite = np.isfinite(numeric)
        values = np.where(finite, numeric, self.numeric_medians_)
        values = (values - self.numeric_mean_) / self.numeric_scale_
        blocks = [sparse.csr_matrix(values), sparse.csr_matrix(~finite, dtype=np.float64)]
        for column, mapping in zip(self.categorical_columns_, self.category_maps_):
            codes = np.fromiter((mapping.get(_category_key(value), -1) for value in X[column].array),
                                dtype=np.int64, count=len(X))
            rows = np.flatnonzero(codes >= 0)
            blocks.append(sparse.csr_matrix(
                (np.ones(len(rows)), (rows, codes[rows])), shape=(len(X), len(mapping))
            ))
        transformed = sparse.hstack(blocks, format="csr", dtype=np.float64)
        if not np.isfinite(transformed.data).all():
            raise ValueError("Preprocessing produced nonfinite features")
        return transformed

    def fit(self, X, y):
        self._is_fitted = False
        self._validate_parameters()
        # Refitting the same learner may legitimately use another column schema.
        if hasattr(self, "columns_"):
            del self.columns_
        self._check_frame(X)
        target = np.asarray(y, dtype=np.float64)
        if len(X) == 0 or target.shape != (len(X),) or not np.isfinite(target).all():
            raise ValueError("y must contain one finite number per nonempty training row")
        self._fit_preprocessing(X)
        encoded = self._transform(X)
        self.target_mean_ = float(target.mean())
        self.target_scale_ = float(target.std())
        if self.target_scale_ == 0:
            self.target_scale_ = 1.
        normalized = (target - self.target_mean_) / self.target_scale_
        if not np.isfinite(normalized).all() or not np.isfinite(self.target_scale_):
            raise ValueError("Target statistics are outside the finite numeric range")
        rng = np.random.default_rng(self.random_state)
        self.intercept_ = np.array(0., dtype=np.float64)
        self.coef_ = np.zeros(encoded.shape[1], dtype=np.float64)
        self.factors_ = rng.normal(0., .01, size=(encoded.shape[1], self.rank))
        parameters = [self.intercept_, self.coef_, self.factors_]
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        self.audit_ = {
            "training_rows": len(X), "encoded_features": encoded.shape[1],
            "encoded_nnz": encoded.nnz, "matrix_format": "csr",
            "numeric_columns": [str(column) for column in self.numeric_columns_],
            "categorical_columns": [str(column) for column in self.categorical_columns_],
            "numeric_medians": self.numeric_medians_.tolist(),
            "numeric_mean": self.numeric_mean_.tolist(),
            "numeric_scale": self.numeric_scale_.tolist(),
            "categorical_counts": [len(mapping) for mapping in self.category_maps_],
            "target_mean": self.target_mean_, "target_scale": self.target_scale_,
            "epochs_completed": 0, "updates": 0, "epoch_minibatch_loss": [],
            "constant_target": bool(np.all(target == target[0])),
        }
        if self.audit_["constant_target"]:
            self.factors_.fill(0.)
            self._is_fitted = True
            return self
        step = 0
        for epoch in range(self.epochs):
            indices = rng.permutation(len(X))
            epoch_loss = 0.
            for start in range(0, len(X), self.batch_size):
                selected = indices[start:start + self.batch_size]
                loss, d_bias, d_linear, d_factors = fm_loss_gradient(
                    encoded[selected], normalized[selected], float(self.intercept_),
                    self.coef_, self.factors_, self.l2,
                )
                if not np.isfinite(loss):
                    raise ValueError("FM optimization produced a nonfinite loss")
                gradients = [d_bias, d_linear, d_factors]
                step += 1
                # Bias correction follows the global minibatch count, not epochs.
                for parameter, moment, variance, gradient in zip(
                        parameters, first_moments, second_moments, gradients):
                    moment *= .9
                    moment += .1 * gradient
                    variance *= .999
                    variance += .001 * np.square(gradient)
                    parameter -= (self.learning_rate * (moment / (1. - .9 ** step))
                                  / (np.sqrt(variance / (1. - .999 ** step)) + 1e-8))
                epoch_loss += loss * len(selected)
            self.audit_["epochs_completed"] = epoch + 1
            self.audit_["updates"] = step
            self.audit_["epoch_minibatch_loss"].append(float(epoch_loss / len(X)))
        if not all(np.isfinite(parameter).all() for parameter in parameters):
            raise ValueError("FM optimization produced nonfinite parameters")
        self._is_fitted = True
        return self

    def predict(self, X):
        if not getattr(self, "_is_fitted", False):
            raise ValueError("FactorizationRegressor must be fitted before prediction")
        self._check_frame(X)
        prediction = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), self.batch_size):
            stop = min(len(X), start + self.batch_size)
            encoded = self._transform(X.iloc[start:stop])
            normalized = _fm_prediction(encoded, float(self.intercept_), self.coef_, self.factors_)
            prediction[start:stop] = self.target_mean_ + self.target_scale_ * normalized
        if not np.isfinite(prediction).all():
            raise ValueError("FM prediction produced nonfinite values")
        return prediction
