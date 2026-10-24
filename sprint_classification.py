"""SPEC011 H72 exact soft-F1 teacher residual and H73 dense ECOC.

No dataset paths, cached labels, validation selection or hidden model fits.
Teacher columns must follow ``np.unique(y)``; caller owns their provenance.
``fit_hook(model, X, y, configuration, **fit_kwargs)`` registers each ECOC head.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.special import logsumexp
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits


def soft_macro_f1_loss_and_gradient(logits, labels, epsilon=1e-8):
    """Return .5 mean CE + .5(1-soft macro-F1), exact full-batch dL/dZ.

    Absent classes contribute zero F1 with the same fixed denominator K.
    This objective is coupled across rows; it is not an additive XGB loss.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim != 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError('Expected finite nonempty N by K logits')
    n, k = z.shape
    if y.shape != (n,) or y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= k):
        raise ValueError('Labels must be integer class positions')
    logp = z - logsumexp(z, axis=1, keepdims=True)
    p = np.exp(logp)
    truth = np.eye(k)[y]
    true_positive = np.sum(truth * p, axis=0)
    denominator = truth.sum(axis=0) + p.sum(axis=0) + epsilon
    soft_f1 = np.mean(2 * true_positive / denominator)
    derivative_p = -(2 / k) * (truth * denominator - true_positive) / denominator**2
    derivative_z = p * (derivative_p - np.sum(p * derivative_p, axis=1, keepdims=True))
    loss = .5 * -np.mean(logp[np.arange(n), y]) + .5 * (1 - soft_f1)
    gradient = .5 * (p - truth) / n + .5 * derivative_z
    return float(loss), gradient


class TrainOnlyEncoder:
    """Fit-only median/log1p/standardization plus unknown-safe categorical OHE."""
    def fit(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or not len(X):
            raise ValueError('Expected nonempty DataFrame with unique columns')
        self.columns_ = list(X.columns)
        self.numeric_ = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_ = [c for c in X if c not in self.numeric_]
        numeric = X[self.numeric_].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        self.medians_ = numeric.median().fillna(0.)
        self.nonnegative_ = [c for c in self.numeric_ if numeric[c].dropna().ge(0).all()]
        values = self._numeric(X)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.
        self.ohe_ = None
        if self.categorical_:
            self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
            self.ohe_.fit(self._categorical(X))
        self.audit_ = {'fit_rows': len(X), 'numeric': self.numeric_, 'log1p': self.nonnegative_,
                       'categorical': self.categorical_, 'clip_standardized': 8,
                       'statistics_source': 'training_only'}
        return self

    def _numeric(self, X):
        values = X[self.numeric_].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        values = values.fillna(self.medians_)
        for c in self.nonnegative_:
            values[c] = np.log1p(values[c].clip(lower=0))
        return values.to_numpy(dtype=np.float32)

    def _categorical(self, X):
        # Tagged values preserve missingness without collisions with user strings.
        return X[self.categorical_].astype('string').apply(lambda col: ('V:' + col).fillna('M:'))

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or any(c not in X for c in self.columns_):
            raise ValueError('Missing fitted feature columns')
        numeric = np.clip((self._numeric(X) - self.mean_) / self.scale_, -8, 8)
        if self.ohe_ is None:
            return numeric
        return np.column_stack([numeric, self.ohe_.transform(self._categorical(X))]).astype(np.float32)


def _teacher(probabilities, rows, classes, required=False):
    if probabilities is None:
        if required:
            raise ValueError('Query teacher probabilities are required after teacher-residual fit')
        return np.full((rows, classes), 1 / classes)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (rows, classes) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
        raise ValueError('Invalid teacher probabilities or sorted class alignment')
    return p / p.sum(axis=1, keepdims=True)


class TeacherResidualF1Classifier:
    """Zero-initialized residual MLP preserving an explicitly supplied teacher."""
    def __init__(self, hidden_size=128, epochs=40, learning_rate=.003, l2=1e-4,
                 random_state=42, threads=2, max_seconds=900):
        self.hidden_size, self.epochs = int(hidden_size), int(epochs)
        self.learning_rate, self.l2 = float(learning_rate), float(l2)
        self.random_state, self.threads, self.max_seconds = random_state, int(threads), max_seconds
        self.configuration = {'hypothesis': 'HYP-072', 'family': 'teacher_residual_numpy_mlp',
            'hidden_size': self.hidden_size, 'epochs': self.epochs, 'learning_rate': self.learning_rate,
            'l2': self.l2, 'objective': '0.5meanCE+0.5(1-softmacroF1)',
            'full_batch': True, 'last_layer_initialization': 'zero',
            'preprocessing': 'fitmedian_log1pnonnegative_standardize_clip8_OHE',
            'teacher_columns': 'sorted_training_classes', 'random_state': random_state,
            'threads': self.threads}

    def fit(self, X, y, teacher_probabilities=None):
        if self.hidden_size < 1 or self.epochs < 0 or self.l2 < 0 or self.learning_rate <= 0:
            raise ValueError('Invalid MLP configuration')
        target = np.asarray(y)
        self.classes_, encoded = np.unique(target, return_inverse=True)
        if target.shape != (len(X),) or not len(target):
            raise ValueError('Training target shape mismatch')
        self.encoder_ = TrainOnlyEncoder().fit(X)
        self.requires_teacher_ = teacher_probabilities is not None
        p = _teacher(teacher_probabilities, len(X), len(self.classes_))
        teacher_logits = np.log(np.clip(p, 1e-12, 1))
        values = np.column_stack([self.encoder_.transform(X), np.log(np.clip(p, 1e-6, 1))]).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        self.w1_ = (rng.normal(size=(values.shape[1], self.hidden_size)) * np.sqrt(2 / values.shape[1])).astype(np.float32)
        self.b1_ = np.zeros(self.hidden_size, dtype=np.float32)
        self.w2_ = np.zeros((self.hidden_size, len(self.classes_)), dtype=np.float32)
        self.b2_ = np.zeros(len(self.classes_), dtype=np.float32)
        parameters = [self.w1_, self.b1_, self.w2_, self.b2_]
        moment = [np.zeros_like(p) for p in parameters]
        variance = [np.zeros_like(p) for p in parameters]
        history = []
        started = time.monotonic()
        with threadpool_limits(limits=self.threads):
            for epoch in range(self.epochs):
                if time.monotonic() - started >= self.max_seconds:
                    raise TimeoutError('H72 full frozen epoch budget exceeded; partial fit is not a score')
                preactivation = values @ self.w1_ + self.b1_
                hidden = np.maximum(preactivation, 0)
                logits = teacher_logits + hidden @ self.w2_ + self.b2_
                loss, dz = soft_macro_f1_loss_and_gradient(logits, encoded)
                loss += .5 * self.l2 * (np.sum(self.w1_**2) + np.sum(self.w2_**2))
                history.append(float(loss))
                dz = dz.astype(np.float32)
                dhidden = (dz @ self.w2_.T) * (preactivation > 0)
                gradients = [values.T @ dhidden + self.l2 * self.w1_, dhidden.sum(axis=0),
                             hidden.T @ dz + self.l2 * self.w2_, dz.sum(axis=0)]
                for parameter, grad, m, v in zip(parameters, gradients, moment, variance):
                    m *= .9
                    m += .1 * grad
                    v *= .999
                    v += .001 * grad**2
                    parameter -= self.learning_rate * (m / (1 - .9**(epoch + 1))) / (np.sqrt(v / (1 - .999**(epoch + 1))) + 1e-8)
        self.audit_ = {'fit_rows': len(X), 'epochs_completed': self.epochs, 'loss_history': history,
            'training_losses_are_independent_oof_scores': False,
            'teacher_source': 'caller_supplied_provenance' if self.requires_teacher_ else 'uniform',
            'teacher_columns': self.classes_.tolist(), 'encoder': self.encoder_.audit_,
            'configuration': self.configuration, 'elapsed_seconds': time.monotonic() - started}
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        p = _teacher(teacher_probabilities, len(X), len(self.classes_), required=self.requires_teacher_)
        values = np.column_stack([self.encoder_.transform(X), np.log(np.clip(p, 1e-6, 1))]).astype(np.float32)
        with threadpool_limits(limits=self.threads):
            hidden = np.maximum(values @ self.w1_ + self.b1_, 0)
            logits = np.log(np.clip(p, 1e-12, 1)) + hidden @ self.w2_ + self.b2_
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def predict(self, X, teacher_probabilities=None):
        return self.classes_[np.argmax(self.predict_proba(X, teacher_probabilities), axis=1)]


def hadamard_code(classes):
    if classes < 1:
        raise ValueError('Need at least one class')
    order = 1 << max(1, int(classes - 1).bit_length())
    if order > 32:
        raise ValueError('Dense ECOC is limited to 32 classes / 31 bits')
    return (hadamard(order)[:classes, 1:] > 0).astype(np.int8)


def decode_ecoc(bit_probabilities, code):
    p, code = np.asarray(bit_probabilities, dtype=np.float64), np.asarray(code)
    if p.ndim != 2 or code.ndim != 2 or p.shape[1] != code.shape[1] or not np.isfinite(p).all():
        raise ValueError('ECOC probability/code shape mismatch')
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # Rows are class codewords; their Bernoulli likelihoods define the decoder.
    logits = (np.log(p) @ code.T + np.log1p(-p) @ (1 - code).T) / code.shape[1]
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


class ECOCClassifier:
    def __init__(self, n_estimators=300, learning_rate=.05, num_leaves=31,
                 min_child_samples=20, reg_lambda=1, threads=2, fit_hook=None,
                 max_seconds=900):
        self.fit_hook, self.max_seconds = fit_hook, max_seconds
        self.parameters = dict(n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            reg_lambda=reg_lambda, n_jobs=threads, verbosity=-1, random_state=42)
        self.configuration = {'hypothesis': 'HYP-073', 'family': 'hadamard_ecoc_lightgbm',
            'parameters': self.parameters, 'decoder': 'mean_Bernoulli_CE_clip1e-6',
            'class_weight': 'N/(K*nclass)_mean_normalized',
            'preprocessing': 'fitmedian_log1pnonnegative_standardize_clip8_OHE'}

    def __getstate__(self):
        # Fit registration belongs to the runner, not to the portable predictor.
        state = self.__dict__.copy()
        state['fit_hook'] = None
        return state

    def fit(self, X, y, teacher_probabilities=None):
        from lightgbm import LGBMClassifier
        target = np.asarray(y)
        self.classes_, encoded, counts = np.unique(target, return_inverse=True, return_counts=True)
        if target.shape != (len(X),) or not len(target):
            raise ValueError('Training target shape mismatch')
        self.encoder_ = TrainOnlyEncoder().fit(X)
        values = self.encoder_.transform(X)
        self.code_ = hadamard_code(len(self.classes_))
        weights = len(target) / (len(counts) * counts[encoded])
        weights /= weights.mean()
        self.heads_, head_audit = [], []
        started = time.monotonic()
        for head in range(self.code_.shape[1]):
            if time.monotonic() - started >= self.max_seconds:
                raise TimeoutError('H73 frozen all-head budget exceeded; partial code is not a score')
            binary = self.code_[encoded, head]
            config = dict(self.configuration, head=head, code_column=self.code_[:, head].tolist(),
                          classes=self.classes_.tolist())
            unique = np.unique(binary)
            if len(unique) == 1:
                self.heads_.append(float(unique[0]))
                head_audit.append(dict(config, constant=float(unique[0]), fit=False))
                continue
            model = LGBMClassifier(**self.parameters)
            if self.fit_hook is not None:
                model = self.fit_hook(model, values, binary, config, sample_weight=weights)
            else:
                # Offline parent must provide a hook to register every weighted head.
                import os
                if os.environ.get('COMBO_RUN_DIR'):
                    raise RuntimeError('Offline ECOC requires per-head fit_hook with sample_weight support')
                model.fit(values, binary, sample_weight=weights)
            self.heads_.append(model)
            head_audit.append(dict(config, fit=True))
        self.audit_ = {'fit_rows': len(X), 'classes': self.classes_.tolist(),
            'code': self.code_.tolist(), 'heads': head_audit, 'encoder': self.encoder_.audit_,
            'elapsed_seconds': time.monotonic() - started, 'configuration': self.configuration}
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        values = self.encoder_.transform(X)
        parts = []
        for head in self.heads_:
            parts.append(np.full(len(X), head) if isinstance(head, float) else head.predict_proba(values)[:, 1])
        return decode_ecoc(np.column_stack(parts), self.code_)

    def predict(self, X, teacher_probabilities=None):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
