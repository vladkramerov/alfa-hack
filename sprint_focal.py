"""H50 unweighted gamma-two multiclass focal boosting on explicit teacher logits.

Only the exact first derivative is used. The CE diagonal curvature is a
positive numerical surrogate, not an exact focal Hessian or upper bound.
"""
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp, softmax

from sprint_ambitious import LovaszTeacherBoost


def focal_loss_and_gradient(logits, labels, gamma=2.):
    """Return mean(-(1-p_y)^gamma*log(p_y)) and its exact logits gradient.

    Differentiating loss in p_y and applying dp_y/dz_j gives multiplier
    (1-p_y)^(gamma-1)*[(1-p_y)-gamma*p_y*log(p_y)] on (p-onehot).
    gamma=0 is handled as exact cross entropy, including p_y rounded to one.
    No class weights or detached focal multipliers alter the objective.
    """
    z, y = np.asarray(logits, dtype=float), np.asarray(labels)
    if (z.ndim != 2 or not len(z) or z.shape[1] < 2 or not np.isfinite(z).all()
            or y.shape != (len(z),) or y.dtype.kind not in 'iu'
            or np.any(y < 0) or np.any(y >= z.shape[1])
            or not np.isfinite(gamma) or gamma < 0):
        raise ValueError('Expected finite N by K logits, integer labels and nonnegative gamma')
    logp = z - logsumexp(z, axis=1, keepdims=True)
    p = np.exp(logp)
    logpt = logp[np.arange(len(y)), y]
    pt = np.exp(logpt)
    complement = -np.expm1(logpt)
    if gamma == 0:
        multiplier = np.ones(len(y))
        focal_weight = multiplier
    else:
        focal_weight = complement ** gamma
        before_power = np.zeros(len(y))
        np.power(complement, gamma - 1, out=before_power, where=complement > 0)
        multiplier = before_power * (complement - gamma * pt * logpt)
    residual = p.copy()
    residual[np.arange(len(y)), y] -= 1
    gradient = multiplier[:, None] * residual / len(y)
    return float(np.mean(-focal_weight * logpt)), gradient


class FocalTeacherBoost(LovaszTeacherBoost):
    """Frozen gamma2 focal loss; inherited preprocessing and teacher offsets."""
    def __init__(self, iterations=200, learning_rate=.05, num_leaves=63,
                 min_child_samples=30, reg_lambda=5., threads=2, max_seconds=900):
        super().__init__(iterations=iterations, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples, reg_lambda=reg_lambda,
            ce_weight=1., threads=threads, max_seconds=max_seconds)

    def configuration(self):
        config = deepcopy(super().configuration())
        config.pop('ce_weight', None)
        config.update(hypothesis='HYP-050', family='teacher_residual_focal_lightgbm',
            objective='mean(-(1-p_true)^2*log(p_true))', gamma=2., class_weights=None,
            gradient='exact_multiclass_focal_derivative_times_N',
            curvature='positive_CE_diagonal_max(p*(1-p),0.001)_not_exact_focal',
            ce_mixture_weight=0., teacher_refit=False)
        return config

    def _objective(self, labels, logits):
        y, z = np.asarray(labels, dtype=np.int64), np.asarray(logits, dtype=float)
        loss, gradient = focal_loss_and_gradient(z, y, gamma=2.)
        self.loss_history_.append(loss)
        p = softmax(z, axis=1)
        return len(y) * gradient, np.maximum(p * (1 - p), .001)
