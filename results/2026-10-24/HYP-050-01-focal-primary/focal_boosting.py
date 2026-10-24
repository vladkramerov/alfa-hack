"""First actual H50: balanced multiclass focal XGBoost, gamma2, fixed600 trees.

The gradient is exact. The strictly positive diagonal curvature is a training
surrogate, not an exact Hessian or a claimed Hessian upper bound.
"""
from __future__ import annotations
import json
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder


def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(1, keepdims=True)


def _validate(margins, labels, alpha):
    values, labels, alpha = np.asarray(margins, dtype=float), np.asarray(labels), np.asarray(alpha, dtype=float)
    if (values.ndim != 2 or labels.shape != (len(values),) or values.shape[1] != len(alpha)
            or not np.isfinite(values).all() or not np.isfinite(alpha).all() or np.any(alpha <= 0)
            or not np.isfinite(labels).all() or np.any(labels != labels.astype(int))
            or np.any(labels < 0) or np.any(labels >= len(alpha))):
        raise ValueError("Finite margins, valid integer labels, positive class alpha required")
    return values, labels.astype(int), alpha


def focal_loss(margins, labels, alpha):
    values, labels, alpha = _validate(margins, labels, alpha)
    shifted = values - values.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    logpt = logp[np.arange(len(labels)), labels]
    pt = np.exp(logpt)
    return float(np.sum(-alpha[labels] * (1 - pt) ** 2 * logpt))


def focal_derivatives(margins, labels, alpha):
    values, labels, alpha = _validate(margins, labels, alpha)
    shifted = values - values.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    p = np.exp(logp)
    logpt = logp[np.arange(len(labels)), labels]
    pt = p[np.arange(len(labels)), labels]
    A = (1 - pt) ** 2 - 2 * pt * (1 - pt) * logpt
    factor = alpha[labels] * A
    gradient = p.copy()
    gradient[np.arange(len(labels)), labels] -= 1
    gradient *= factor[:, None]
    curvature = np.maximum(2 * factor[:, None] * p * (1 - p), 1e-6)
    if not np.isfinite(gradient).all() or not np.isfinite(curvature).all():
        raise FloatingPointError("Nonfinite focal derivatives")
    return gradient, curvature


class FocalObjective:
    """Picklable sklearn XGBoost objective with deadline checked each round."""
    def __init__(self, alpha, deadline=None):
        self.alpha = np.asarray(alpha, dtype=float)
        self.deadline = deadline
        self.calls = 0

    def __call__(self, labels, margins):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("H50 total 900-second model budget exhausted")
        self.calls += 1
        if np.asarray(margins).ndim == 1 and len(self.alpha) == 2:
            # sklearn uses binary:logistic for two classes. Its sole margin
            # represents z1-z0, so return the corresponding positive-class
            # derivative while retaining the same focal loss definition.
            expanded = np.column_stack([np.zeros(len(margins)), margins])
            gradient, curvature = focal_derivatives(expanded, labels, self.alpha)
            return gradient[:, 1], curvature[:, 1]
        return focal_derivatives(margins, labels, self.alpha)


class FocalBoostingClassifier:
    def fit(self, X, y, *, deadline=None, estimators=600):
        from automl_models import Encoder
        from xgboost import XGBClassifier
        if not isinstance(estimators, int) or not 1 <= estimators <= 600:
            raise ValueError("At most600 estimators allowed; public H50 is fixed600")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("H50 total model budget exhausted before fit")
        self.labels_ = LabelEncoder().fit(y)
        self.classes_ = self.labels_.classes_
        target = self.labels_.transform(y)
        if len(self.classes_) < 2 or len(X) != len(target):
            raise ValueError("At least two aligned training classes required")
        counts = np.bincount(target, minlength=len(self.classes_))
        alpha = 1 / np.sqrt(counts.astype(float))
        alpha *= len(target) / float(counts @ alpha)
        self.objective_ = FocalObjective(alpha, deadline=deadline)
        self.encoder_ = Encoder(native=False, scale=False)
        values = self.encoder_.fit_transform(X, target)
        self.model_ = XGBClassifier(n_estimators=estimators, max_depth=6, learning_rate=.05,
            reg_lambda=1., tree_method="hist", device="cpu", n_jobs=4,
            objective=self.objective_, random_state=42, verbosity=0)
        self.model_.fit(values, target, verbose=False)
        config = json.loads(self.model_.get_booster().save_config())
        objective_name = config["learner"]["objective"]["name"]
        expected = "multi:softprob" if len(self.classes_) > 2 else "binary:logistic"
        if objective_name != expected:
            raise ValueError(f"Unexpected XGBoost prediction objective: {objective_name}")
        self.audit_ = {"mechanism":"class_balanced_multiclass_focal_boosting_first_H50_test",
            "fit_rows":len(X),"class_counts":counts.tolist(),"classes":self.classes_.tolist(),
            "alpha":alpha.tolist(),"mean_sample_alpha":float(alpha[target].mean()),
            "gamma":2,"gradient":"exact alpha*A*(p-y)",
            "curvature":"positive diagonal surrogate max(2*alpha*A*p*(1-p),1e-6), not exact Hessian",
            "objective_calls":self.objective_.calls,"prediction_objective":objective_name,
            "estimators":estimators,"max_depth":6,"learning_rate":.05,"reg_lambda":1.,
            "tree_method":"hist","device":"cpu","threads":4,"seed":42,
            "preprocessing":"legacy Encoder median numeric/OHE categorical, no scaling, fit only",
            "early_stopping":False,"model_fit_count":1}
        return self

    def predict_proba(self, X):
        p = np.asarray(self.model_.predict_proba(self.encoder_.transform(X)), dtype=float)
        if (p.shape != (len(X), len(self.classes_)) or not np.isfinite(p).all()
                or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5)):
            raise FloatingPointError("Invalid focal classifier probabilities")
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
