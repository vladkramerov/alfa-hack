"""H75 train-only affine posterior decisions for expected macro-F1.

This fixed-path heuristic is inspired by cost-sensitive confusion optimization,
not a claim of global convergence. Teacher provenance/calibration is the caller's
responsibility. No file I/O, learner imports or query-cohort adaptation occurs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_macro_f1_gradient(class_prior, predicted_mass, true_positive, epsilon=1e-8):
    """Expected macro-F1 and exact partials with respect to t and q.

    With K classes, J = mean_k 2t_k / (pi_k + q_k + epsilon).
    Multiplying both returned gradients by K leaves the affine decision intact.
    """
    prior, q, t = (np.asarray(v, dtype=np.float64) for v in
                   (class_prior, predicted_mass, true_positive))
    if (prior.ndim != 1 or not len(prior) or q.shape != prior.shape or t.shape != prior.shape
            or any(not np.isfinite(v).all() or np.any(v < 0) for v in (prior, q, t))
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Expected finite nonnegative class moment vectors and positive epsilon')
    denominator = prior + q + epsilon
    return (float(np.mean(2 * t / denominator)),
            2 / (len(prior) * denominator),
            -2 * t / (len(prior) * denominator**2))


class MacroF1Decision:
    """Freeze scores a_k*p_k+b_k learned only from supplied training rows.

    DataFrame probability columns are aligned by explicit class labels; arrays
    must already follow np.unique(training_y). predict_proba returns softmax
    scores preserving the decision, and does not claim posterior calibration.
    """
    def __init__(self, iterations=25, damping=.5, epsilon=1e-8):
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.epsilon = float(epsilon)
        if (self.iterations != iterations or self.iterations < 0
                or not 0 < self.damping <= 1 or not np.isfinite(self.epsilon) or self.epsilon <= 0):
            raise ValueError('Invalid frozen affine-decision configuration')

    def configuration(self):
        return {'hypothesis': 'HYP-075', 'family': 'affine_macro_f1_decision',
                'iterations': self.iterations, 'damping': self.damping, 'epsilon': self.epsilon,
                'objective': 'mean(2*t/(training_prior+predicted_mass+epsilon))',
                'selection': 'best_expected_training_objective_on_fixed_path',
                'initialization': 'gain1_offset0_teacher_argmax',
                'prior_correction': 'none', 'teacher_columns': 'sorted_training_classes',
                'query_statistics': False, 'output': 'softmax_affine_scores_not_calibrated'}

    def _teacher(self, probabilities, rows):
        if probabilities is None:
            raise ValueError('Explicit teacher probabilities are required')
        if isinstance(probabilities, pd.DataFrame):
            if (not probabilities.columns.is_unique
                    or len(probabilities.columns) != len(self.classes_)
                    or set(probabilities.columns) != set(self.classes_)):
                raise ValueError('Teacher columns must match sorted training classes')
            probabilities = probabilities.loc[:, self.classes_]
        p = np.asarray(probabilities, dtype=np.float64)
        if (p.shape != (rows, len(self.classes_)) or not np.isfinite(p).all()
                or np.any(p < 0) or np.any(p.sum(axis=1) <= 0)
                or not np.isfinite(p.sum(axis=1)).all()):
            raise ValueError('Expected finite nonnegative N by K teacher probabilities')
        return p / p.sum(axis=1, keepdims=True)

    def _moments(self, p, gain, offset):
        selected = np.argmax(p * gain + offset, axis=1)
        q = np.bincount(selected, minlength=len(self.classes_)) / len(p)
        t = np.bincount(selected, weights=p[np.arange(len(p)), selected],
                        minlength=len(self.classes_)) / len(p)
        return q, t

    def fit(self, X, y, teacher_probabilities=None):
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X) or not len(target) or pd.isna(target).any():
            raise ValueError('Expected nonempty complete training labels matching X')
        self.classes_, counts = np.unique(target, return_counts=True)
        self.class_prior_ = counts.astype(np.float64) / len(target)
        p = self._teacher(teacher_probabilities, len(X))
        gain, offset = np.ones(len(self.classes_)), np.zeros(len(self.classes_))
        q, t = self._moments(p, gain, offset)
        score, dt, dq = expected_macro_f1_gradient(self.class_prior_, q, t, self.epsilon)
        best_score, best_iteration = score, 0
        best_gain, best_offset = gain.copy(), offset.copy()
        history = [score]
        for iteration in range(1, self.iterations + 1):
            # Drop common 1/K factor; this does not change the cost-sensitive argmax.
            gain = (1 - self.damping) * gain + self.damping * dt * len(self.classes_)
            offset = (1 - self.damping) * offset + self.damping * dq * len(self.classes_)
            q, t = self._moments(p, gain, offset)
            score, dt, dq = expected_macro_f1_gradient(self.class_prior_, q, t, self.epsilon)
            history.append(score)
            if score > best_score:
                best_score, best_iteration = score, iteration
                best_gain, best_offset = gain.copy(), offset.copy()
        self.gain_, self.offset_ = best_gain, best_offset
        self.audit_ = {'fit_rows': len(target), 'statistics_source': 'training_only',
                       'class_prior': self.class_prior_.tolist(),
                       'expected_objective_history': history,
                       'initial_expected_macro_f1': history[0],
                       'selected_expected_macro_f1': best_score,
                       'selected_iteration': best_iteration,
                       'gain': best_gain.tolist(), 'offset': best_offset.tolist(),
                       'prior_correction': 'none', 'output_is_calibrated_posterior': False,
                       'teacher_calibration_assumed_not_verified': True}
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        if not hasattr(self, 'gain_'):
            raise ValueError('MacroF1Decision must be fitted before prediction')
        p = self._teacher(teacher_probabilities, len(X))
        if not len(p):
            return np.empty_like(p)
        scores = p * self.gain_ + self.offset_
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X, teacher_probabilities=None):
        return self.classes_[np.argmax(self.predict_proba(X, teacher_probabilities), axis=1)]
