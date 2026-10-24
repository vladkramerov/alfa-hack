"""H77: full-batch Lovasz macro-IoU ranks in teacher-residual tree boosting.

No dataset access, teacher training, validation search or query-cohort adaptation.
The caller must audit supplied teacher probabilities and register this one fit.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

from sprint_classification import TrainOnlyEncoder


def lovasz_softmax_loss_and_gradient(logits, labels, ce_weight=.5):
    """Exact subgradient of w*meanCE+(1-w)*mean-present Lovasz-Softmax.

    For each class, sort absolute probability errors descending and dot them
    with successive increases in Jaccard error. Stable ties select a valid
    subgradient; finite differences require strictly ordered errors. Absent
    target classes are excluded from the Lovasz mean, as in classes='present'.
    Returns the derivative of the *mean loss*, before boosting's N scaling.
    """
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if (z.ndim != 2 or not z.shape[0] or z.shape[1] < 2 or not np.isfinite(z).all()
            or y.shape != (len(z),) or y.dtype.kind not in 'iu'
            or np.any(y < 0) or np.any(y >= z.shape[1])
            or not np.isfinite(ce_weight) or not 0 <= ce_weight <= 1):
        raise ValueError('Expected finite N by K logits and integer class-position labels')
    logp = z - logsumexp(z, axis=1, keepdims=True)
    p = np.exp(logp)
    truth = np.eye(z.shape[1])[y]
    active = np.unique(y)
    derivative_p = np.zeros_like(p)
    rank_loss = 0.
    for c in active:
        foreground = truth[:, c]
        errors = np.abs(foreground - p[:, c])
        order = np.argsort(-errors, kind='stable')
        fg_rank = foreground[order]
        remaining_true = foreground.sum() - np.cumsum(fg_rank)
        expanded_union = foreground.sum() + np.cumsum(1 - fg_rank)
        cumulative_error = 1 - remaining_true / expanded_union
        increments = np.diff(np.r_[0., cumulative_error])
        rank_loss += float(errors[order] @ increments) / len(active)
        derivative_p[order, c] = (1 - 2 * fg_rank) * increments / len(active)
    rank_gradient = p * (derivative_p - np.sum(derivative_p * p, axis=1, keepdims=True))
    ce_loss = -np.mean(logp[np.arange(len(y)), y])
    gradient = ce_weight * (p - truth) / len(y) + (1 - ce_weight) * rank_gradient
    return float(ce_weight * ce_loss + (1 - ce_weight) * rank_loss), gradient


class LovaszTeacherBoost:
    """One LightGBM learner with supplied teacher logits as a fixed offset.

    The diagonal curvature is max(.5*p*(1-p), .001) at the default CE weight.
    It is a positive numerical surrogate, NOT the exact Lovasz Hessian or an
    upper bound. No claim of convexity in logits or guaranteed descent is made.
    """
    def __init__(self, iterations=200, learning_rate=.05, num_leaves=63,
                 min_child_samples=30, reg_lambda=5., ce_weight=.5,
                 threads=2, max_seconds=900):
        if (not isinstance(iterations, (int, np.integer)) or iterations < 0
                or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 4
                or not np.isfinite(ce_weight) or not 0 <= ce_weight <= 1
                or not np.isfinite(max_seconds) or max_seconds <= 0
                or not np.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError('Invalid frozen Lovasz boosting configuration')
        self.iterations, self.ce_weight, self.max_seconds = iterations, ce_weight, max_seconds
        self.parameters = dict(n_estimators=iterations, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            reg_lambda=reg_lambda, n_jobs=threads, verbosity=-1, random_state=42,
            deterministic=True, force_col_wise=True, boost_from_average=False, metric='None')

    def configuration(self):
        return {'hypothesis': 'HYP-077', 'family': 'teacher_residual_lovasz_lightgbm',
                'parameters': deepcopy(self.parameters), 'ce_weight': self.ce_weight,
                'objective': 'w*meanCE+(1-w)*mean_present_LovaszSoftmax',
                'gradient': 'exact_fullbatch_sort_subgradient_times_N',
                'curvature': 'positive_surrogate_max(w*p*(1-p),0.001)_not_exact',
                'teacher_baseline': 'caller_supplied_log_normalized_p_clip1e-12',
                'inference': 'softmax(query_logteacher+tree_raw_scores)',
                'preprocessing': 'fitmedian_log1pnonnegative_standardize_clip8_OHE',
                'query_statistics': False, 'class_weights': None,
                'selection': 'fixed_all_rounds_no_validation_earlystop'}

    def _teacher(self, probabilities, rows):
        if probabilities is None:
            raise ValueError('Explicit teacher probabilities are required')
        if isinstance(probabilities, pd.DataFrame) and list(probabilities.columns) != list(self.classes_):
            raise ValueError('Teacher columns must match sorted training classes')
        p = np.asarray(probabilities, dtype=np.float64)
        if p.shape != (rows, len(self.classes_)) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError('Invalid teacher shape or probabilities')
        largest = p.max(axis=1, keepdims=True)
        if np.any(largest <= 0):
            raise ValueError('Teacher rows need positive probability mass')
        p = p / largest
        return p / p.sum(axis=1, keepdims=True)

    def _objective(self, labels, logits):
        y = np.asarray(labels, dtype=np.int64)
        z = np.asarray(logits, dtype=np.float64)
        loss, gradient = lovasz_softmax_loss_and_gradient(z, y, self.ce_weight)
        self.loss_history_.append(loss)
        p = softmax(z, axis=1)
        hessian_surrogate = np.maximum(self.ce_weight * p * (1 - p), .001)
        return len(y) * gradient, hessian_surrogate

    def fit(self, X, y, teacher_probabilities=None):
        from lightgbm import LGBMClassifier
        target = np.asarray(y)
        if (not isinstance(X, pd.DataFrame) or not len(X) or not X.columns.is_unique
                or target.shape != (len(X),) or pd.isna(target).any()):
            raise ValueError('Expected nonempty unique-column DataFrame and complete target')
        self.classes_, encoded = np.unique(target, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('Need at least two classes')
        p = self._teacher(teacher_probabilities, len(X))
        self.columns_ = list(X.columns)
        self.encoder_ = TrainOnlyEncoder().fit(X)
        values = self.encoder_.transform(X)
        self.model_, self.loss_history_ = None, []
        started = time.monotonic()
        if self.iterations:
            def budget_callback(env):
                if time.monotonic() - started >= self.max_seconds:
                    raise TimeoutError('H77 all-round fit budget exceeded; partial model is not a score')
            self.model_ = LGBMClassifier(**self.parameters, objective=self._objective,
                                        num_class=len(self.classes_))
            self.model_.fit(values, encoded, init_score=np.log(np.clip(p, 1e-12, 1)),
                            callbacks=[budget_callback])
        self.audit_ = {'fit_rows': len(X), 'classes': self.classes_.tolist(),
            'learner_fits': int(self.model_ is not None), 'removed_rows': 0,
            'teacher_source': 'caller_supplied_provenance', 'teacher_columns': self.classes_.tolist(),
            'loss_history': list(self.loss_history_), 'encoder': self.encoder_.audit_,
            'curvature_is_exact': False, 'training_loss_is_independent_score': False,
            'elapsed_seconds': time.monotonic() - started, 'configuration': self.configuration()}
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        if not hasattr(self, 'encoder_'):
            raise ValueError('Fit LovaszTeacherBoost before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query DataFrame must match fitted columns in order')
        p = self._teacher(teacher_probabilities, len(X))
        if self.model_ is None:
            return p
        if not len(X):
            return np.empty((0, len(self.classes_)))
        residual = np.asarray(self.model_.predict(self.encoder_.transform(X), raw_score=True))
        return softmax(residual + np.log(np.clip(p, 1e-12, 1)), axis=1)

    def predict(self, X, teacher_probabilities=None):
        return self.classes_[self.predict_proba(X, teacher_probabilities).argmax(axis=1)]
