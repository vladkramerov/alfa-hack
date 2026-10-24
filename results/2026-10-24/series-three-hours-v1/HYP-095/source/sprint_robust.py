"""H95 generalized cross entropy with train-only class balance and honest OOF.

Exact first derivatives are used; curvature is a positive CE surrogate. No
noise transition matrix, row deletion, teacher fit or query adaptation occurs.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

from sprint_ambitious import LovaszTeacherBoost


def generalized_ce_loss_and_gradient(logits, labels, q=.7, weights=None):
    """Weighted mean (1-p_true**q)/q and exact logits gradient.

    q=0 is the continuous cross-entropy limit. Supplied row weights enter the
    mean directly; the classifier normalizes its inverse-sqrt class weights to
    mean one separately. No clipped p_true changes the differentiable loss.
    """
    z, y = np.asarray(logits, dtype=float), np.asarray(labels)
    if (z.ndim != 2 or not len(z) or z.shape[1] < 2 or not np.isfinite(z).all()
            or y.shape != (len(z),) or y.dtype.kind not in 'iu'
            or np.any(y < 0) or np.any(y >= z.shape[1])
            or not np.isfinite(q) or not 0 <= q <= 1):
        raise ValueError('Expected finite N by K logits, integer labels and q in [0,1]')
    a = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if a.shape != (len(y),) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Expected finite nonnegative aligned row weights')
    logp = z - logsumexp(z, axis=1, keepdims=True)
    logpt = logp[np.arange(len(y)), y]
    if q == 0:
        row_loss, modulation = -logpt, np.ones(len(y))
    else:
        row_loss, modulation = -np.expm1(q * logpt) / q, np.exp(q * logpt)
    residual = np.exp(logp)
    residual[np.arange(len(y)), y] -= 1
    gradient = (a * modulation)[:, None] * residual / len(y)
    return float(np.mean(a * row_loss)), gradient


class RobustTeacherBoost(LovaszTeacherBoost):
    """Fixed q=.7 GCE with dynamic low-label-probability gradient suppression."""
    def __init__(self, iterations=300, learning_rate=.05, num_leaves=63,
                 min_child_samples=30, reg_lambda=5., threads=2, max_seconds=900):
        super().__init__(iterations=iterations, learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, reg_lambda=reg_lambda, ce_weight=1.,
            threads=threads, max_seconds=max_seconds)

    def configuration(self):
        result = deepcopy(super().configuration())
        result.pop('ce_weight', None)
        result.update(hypothesis='HYP-095', family='honest_oof_generalized_ce_lightgbm',
            objective='mean(a_class*(1-p_true**0.7)/0.7)', q=.7,
            class_weights='inverse_sqrt_training_class_counts_normalized_row_mean_one',
            gradient='a_class*p_true**0.7*(p-onehot)_exact_times_N',
            curvature='positive_surrogate_max(a_class*p*(1-p),.001)_not_exact_GCE',
            training_teacher='reused_H88_nested_grouped_oof_raw_xgboost',
            query_teacher='original_held_out_raw_xgboost', new_teacher_fits=0,
            H48_meta_state_used=False, teacher_provenance='caller_verifies_reused_honest_teacher_cache',
            noise_assumption='not_proven_classconditional_noise', row_deletion=False)
        return result

    def fit(self, X, y, teacher_probabilities=None):
        target = np.asarray(y)
        if target.shape != (len(X),) or not len(target) or pd.isna(target).any():
            raise ValueError('Complete nonempty labels matching X required')
        if teacher_probabilities is None:
            raise ValueError('Explicit provenance-checked honest OOF teacher required')
        classes, encoded, counts = np.unique(target, return_inverse=True, return_counts=True)
        self.class_weights_ = 1 / np.sqrt(counts.astype(float))
        self.class_weights_ /= np.mean(self.class_weights_[encoded])
        result = super().fit(X, target, teacher_probabilities)
        if not np.array_equal(classes, self.classes_):
            raise ValueError('Training class order unexpectedly changed')
        self.audit_.update(class_counts=counts.tolist(), class_weights=self.class_weights_.tolist(),
            class_weight_classes=self.classes_.tolist(),
            sample_weight_mean=float(np.mean(self.class_weights_[encoded])),
            sample_weight_min=float(np.min(self.class_weights_[encoded])),
            sample_weight_max=float(np.max(self.class_weights_[encoded])),
            q=.7, expected_training_teacher='reused_H88_grouped_oof',
            teacher_oof_provenance_is_callers_responsibility=True)
        return result

    def _objective(self, labels, logits):
        y, z = np.asarray(labels, dtype=np.int64), np.asarray(logits, dtype=float)
        row_weight = self.class_weights_[y]
        loss, gradient = generalized_ce_loss_and_gradient(z, y, q=.7, weights=row_weight)
        self.loss_history_.append(loss)
        p = softmax(z, axis=1)
        return len(y) * gradient, np.maximum(row_weight[:, None] * p * (1 - p), .001)
