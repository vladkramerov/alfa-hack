"""Train-local meta-learning mechanisms registered as H76--H78 in SPEC012.

No datasets, global coefficients, filesystem writes, or implicit base-model fits.
All learned linear coefficients are exported in raw log-probability space, so
automl_class_combo.predict_meta_state can also evaluate these state dictionaries.
The caller owns experiment registration, fit accounting, resource limits and
raw-feature-group generation. Meta cross-fitting is conditional on supplied
base OOF inputs; it does not claim to be a fully nested base-model evaluation.
"""
from __future__ import annotations

import hashlib
import warnings
import numpy as np

COMPONENT_ORDER = ("raw", "smote", "frequency")
CLIP = 1e-6


def _classes(classes):
    values = np.asarray(classes)
    if values.ndim != 1 or not len(values):
        raise ValueError("Nonempty one-dimensional classes required")
    if len(set(values.tolist())) != len(values):
        raise ValueError("Class labels must be unique")
    return values


def _targets(y, classes):
    classes = _classes(classes)
    values = np.asarray(y)
    if values.ndim != 1 or not len(values):
        raise ValueError("Nonempty one-dimensional targets required")
    lookup = {label: i for i, label in enumerate(classes.tolist())}
    try:
        indices = np.asarray([lookup[label] for label in values.tolist()], dtype=int)
    except (KeyError, TypeError) as error:
        raise ValueError("Targets contain a class outside declared classes") from error
    return values, indices


def align_probabilities(probabilities, source_classes, classes):
    """Align a branch's class columns, allowing absent but not unknown classes."""
    source, classes = _classes(source_classes), _classes(classes)
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(source):
        raise ValueError("Probability column count differs from source classes")
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values > 1 + 1e-8)
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=1e-6)):
        raise ValueError("Finite nonnegative normalized probabilities required")
    lookup = {label: i for i, label in enumerate(classes.tolist())}
    if any(label not in lookup for label in source.tolist()):
        raise ValueError("Branch contains an unknown class")
    result = np.zeros((len(values), len(classes)), dtype=float)
    for i, label in enumerate(source.tolist()):
        result[:, lookup[label]] = values[:, i]
    if len(values):
        result /= result.sum(axis=1, keepdims=True)
    return result


def _parts(probabilities, classes):
    """Accept three aligned arrays, keyed branches, or branch payloads with classes."""
    classes = _classes(classes)
    if isinstance(probabilities, dict):
        if set(probabilities) != set(COMPONENT_ORDER):
            raise ValueError("Exactly raw, smote and frequency branches required")
        pieces = [probabilities[name] for name in COMPONENT_ORDER]
    else:
        pieces = list(probabilities)
        if len(pieces) != len(COMPONENT_ORDER):
            raise ValueError("Exactly three probability branches required")
    result = []
    for piece in pieces:
        source = classes
        if isinstance(piece, dict):
            source = piece.get("classes", classes)
            piece = piece["probabilities"]
        values = align_probabilities(piece, source, classes)
        if result and len(values) != len(result[0]):
            raise ValueError("Probability branches have different row counts")
        result.append(values)
    return result


def _features(probabilities, classes):
    return np.concatenate([np.log(np.clip(p, CLIP, 1))
                           for p in _parts(probabilities, classes)], axis=1)


def _training_hash(values, indices):
    digest = hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes())
    digest.update(np.ascontiguousarray(indices, dtype='<i8').tobytes())
    return digest.hexdigest()


def _state(coef, intercept, classes, audit):
    return {"coef": np.asarray(coef, dtype=float).tolist(),
            "intercept": np.asarray(intercept, dtype=float).tolist(),
            "classes": _classes(classes).tolist(),
            "component_order": list(COMPONENT_ORDER), "clip": CLIP,
            "audit": audit}


def _linear_values(values, state, classes):
    classes = _classes(classes)
    if not np.array_equal(state.get("classes"), classes):
        raise ValueError("Prediction class order differs from fitted state")
    if state.get("component_order") != list(COMPONENT_ORDER) or state.get("clip") != CLIP:
        raise ValueError("Unsupported meta feature protocol")
    coefficient = np.asarray(state["coef"], dtype=float)
    intercept = np.asarray(state["intercept"], dtype=float)
    rows = 1 if len(classes) <= 2 else len(classes)
    if (coefficient.shape != (rows, values.shape[1]) or intercept.shape != (rows,)
            or not np.isfinite(coefficient).all() or not np.isfinite(intercept).all()):
        raise ValueError("Invalid fitted meta coefficient shape or values")
    logits = values @ coefficient.T + intercept
    if len(classes) == 2:
        logits = np.column_stack([np.zeros(len(logits)), logits[:, 0]])
    return logits


def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=1, keepdims=True)


def predict_state(probabilities, state, classes):
    """Evaluate any H75/H76/H78 raw-log-space state, without refitting."""
    return _softmax(_linear_values(_features(probabilities, classes), state, classes))


def _fit_logistic_state(probabilities, y, classes, normalized):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classes = _classes(classes)
    y, indices = _targets(y, classes)
    values = _features(probabilities, classes)
    if len(values) != len(y):
        raise ValueError("Target and probability row counts differ")
    counts = np.bincount(indices, minlength=len(classes))
    if np.any(counts == 0):
        raise ValueError("Every declared class needs training examples")
    raw_weight = 1 / np.sqrt(counts.astype(float))
    factor = len(y) / float(counts @ raw_weight) if normalized else 1.
    class_weights = raw_weight * factor
    sample_weight = class_weights[indices]
    audit = {"mechanism": "normalized_converged_meta" if normalized else "legacy_H75_meta",
             "fit_rows": len(y), "class_counts": counts.tolist(),
             "class_weights": class_weights.tolist(),
             "mean_sample_weight": float(sample_weight.mean()),
             "scaling": "StandardScaler_fit_only" if normalized else "none",
             "C": .1, "max_iter": 3000 if normalized else 1000,
             "tol": 1e-5 if normalized else 1e-4, "solver": "lbfgs",
             "fit_count": 0 if len(classes) == 1 else 1,
             "training_sha256": _training_hash(values, indices)}
    if len(classes) == 1:
        audit.update(converged=True, n_iter=[0], warnings=[])
        return _state(np.zeros((1, values.shape[1])), np.zeros(1), classes, audit)
    scaler = StandardScaler().fit(values) if normalized else None
    transformed = scaler.transform(values) if scaler is not None else values
    model = LogisticRegression(C=.1, solver="lbfgs", max_iter=audit["max_iter"],
                               tol=audit["tol"])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        # Integer labels preserve the declared class order, including numeric or
        # non-lexicographic labels, independently of sklearn's sorting behavior.
        model.fit(transformed, indices, sample_weight=sample_weight)
    if not np.array_equal(model.classes_, np.arange(len(classes))):
        raise ValueError("Fitted classifier lost a declared class")
    coef, intercept = model.coef_.copy(), model.intercept_.copy()
    if scaler is not None:
        coef = coef / scaler.scale_[None, :]
        intercept = intercept - coef @ scaler.mean_
        audit.update(scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
    audit.update(n_iter=model.n_iter_.tolist(),
                 warnings=[str(item.message) for item in captured],
                 converged=not any(issubclass(item.category, ConvergenceWarning)
                                   for item in captured))
    return _state(coef, intercept, classes, audit)


def fit_normalized_state(probabilities, y, classes):
    """H76: scale log features and normalize inverse-sqrt weights to mean one."""
    return _fit_logistic_state(probabilities, y, classes, True)


def fit_legacy_state(probabilities, y, classes):
    """H75 control meta head only; never fits or regenerates base components."""
    return _fit_logistic_state(probabilities, y, classes, False)


def focal_mcc_loss(probabilities, one_hot, gamma=1.5, epsilon=1e-12):
    """Mean OvR adaptation of the binary focal differentiable MCC formula.

    The input tensors retain gradients. W=1 is fixed in H78. This is our
    multiclass adaptation of Seber's binary objective, not an exact replication.
    """
    import torch
    p, y = probabilities, one_hot
    if p.ndim != 2 or y.shape != p.shape or p.shape[1] < 2:
        raise ValueError("Matching multiclass probability and one-hot tensors required")
    pt = p * y + (1 - p) * (1 - y)
    correct = torch.clamp(pt, 0., 1.).pow(gamma)
    incorrect = torch.clamp(1 - pt, 0., 1.).pow(gamma)
    tp, fn = (correct * y).sum(0), (incorrect * y).sum(0)
    fp, tn = (incorrect * (1 - y)).sum(0), (correct * (1 - y)).sum(0)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).clamp_min(epsilon).sqrt()
    return -((tp * tn - fp * fn) / denominator).mean()


def fit_mcc_state(probabilities, y, classes, initial_state):
    """H78: exactly 100 Adam steps after a train-local H76 warm start."""
    import torch
    from sklearn.preprocessing import StandardScaler

    classes = _classes(classes)
    y, indices = _targets(y, classes)
    values = _features(probabilities, classes)
    if len(values) != len(y) or np.any(np.bincount(indices, minlength=len(classes)) == 0):
        raise ValueError("MCC fitting requires aligned rows and every declared class")
    # Validate the warm-start protocol/classes/shapes before using coefficients.
    _linear_values(values[:1], initial_state, classes)
    if initial_state.get("audit", {}).get("mechanism") != "normalized_converged_meta":
        raise ValueError("MCC warm start must be a current H76 normalized meta head")
    if initial_state.get("audit", {}).get("training_sha256") != _training_hash(values, indices):
        raise ValueError("MCC warm start must be fitted on exactly current training rows")
    if len(classes) == 1:
        state = dict(initial_state)
        state["audit"] = {"mechanism": "focal_mcc_meta", "fit_count": 0,
                          "steps": 0, "reason": "single_class"}
        return state
    scaler = StandardScaler().fit(values)
    features = torch.tensor(scaler.transform(values), dtype=torch.float64)
    encoded = torch.as_tensor(indices, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(encoded, len(classes)).to(torch.float64)
    raw_coef = np.asarray(initial_state["coef"], dtype=float)
    raw_intercept = np.asarray(initial_state["intercept"], dtype=float)
    weight = torch.tensor(raw_coef * scaler.scale_[None, :],
                          dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(raw_intercept + raw_coef @ scaler.mean_,
                        dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=.003)
    losses = []
    for step in range(100):
        optimizer.zero_grad()
        logits = features @ weight.T + bias
        if len(classes) == 2:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        p = torch.softmax(logits, dim=1)
        loss = focal_mcc_loss(p, one_hot, gamma=1.5) + .001 * weight.square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite focal MCC objective")
        loss.backward()
        if any(param.grad is None or not bool(torch.isfinite(param.grad).all())
               for param in (weight, bias)):
            raise FloatingPointError("Nonfinite focal MCC derivative")
        optimizer.step()
        losses.append(float(loss.detach()))
    coefficient = weight.detach().numpy() / scaler.scale_[None, :]
    intercept = bias.detach().numpy() - coefficient @ scaler.mean_
    audit = {"mechanism": "mean_ovr_focal_mcc_own_adaptation", "fit_rows": len(y),
             "fit_count": 1, "steps": 100, "gamma": 1.5, "W": 1,
             "optimizer": "Adam", "learning_rate": .003, "l2_mean_weight": .001,
             "scaling": "StandardScaler_fit_only", "warm_start": "current_train_H76",
             "losses": losses, "scaler_mean": scaler.mean_.tolist(),
             "scaler_scale": scaler.scale_.tolist(),
             "training_sha256": _training_hash(values, indices)}
    return _state(coefficient, intercept, classes, audit)


def _validated_folds(folds, n_rows):
    result, seen = [], np.zeros(n_rows, dtype=int)
    for number, (fit, valid) in enumerate(folds):
        fit, valid = np.asarray(fit), np.asarray(valid)
        if (fit.ndim != 1 or valid.ndim != 1 or not len(fit) or not len(valid)
                or fit.dtype.kind not in "iu" or valid.dtype.kind not in "iu"
                or np.any(fit < 0) or np.any(valid < 0)
                or np.any(fit >= n_rows) or np.any(valid >= n_rows)
                or len(np.unique(fit)) != len(fit) or len(np.unique(valid)) != len(valid)
                or np.intersect1d(fit, valid).size):
            raise ValueError("Meta folds require unique, disjoint, in-range integer positions")
        seen[valid] += 1
        result.append((fit.astype(int), valid.astype(int)))
    if len(result) < 2 or not np.all(seen == 1):
        raise ValueError("Meta folds must validate each training row exactly once")
    return result


def crossfit_meta(probabilities, y, classes, folds, method="normalized", initial_states=None):
    """Return (OOF probabilities, audit) from supplied disjoint grouped folds.

    Folds must be produced from raw-feature group IDs by the caller; this method
    verifies index disjointness and exact OOF coverage, not raw group semantics.
    """
    from sklearn.metrics import f1_score
    classes = _classes(classes)
    y, _ = _targets(y, classes)
    parts = _parts(probabilities, classes)
    if len(parts[0]) != len(y):
        raise ValueError("Target and probability row counts differ")
    if method not in {"legacy", "normalized", "mcc"}:
        raise ValueError("Unknown meta method")
    checked = _validated_folds(folds, len(y))
    if initial_states is not None and (method != "mcc" or len(initial_states) != len(checked)):
        raise ValueError("Warm-start states require one normalized state per MCC fold")
    predictions = np.zeros((len(y), len(classes)))
    audit = {"method": method, "folds": [], "fit_count": 0, "states": [],
             "selection_source": "training_meta_crossfit_on_base_grouped_oof",
             "fully_nested_base_evaluation": False}
    for number, (fit, valid) in enumerate(checked):
        fit_parts, valid_parts = [p[fit] for p in parts], [p[valid] for p in parts]
        reused = method == "mcc" and initial_states is not None
        if reused:
            state = initial_states[number]
            if state.get("audit", {}).get("mechanism") != "normalized_converged_meta":
                raise ValueError("MCC fold warm start must be H76 normalized meta")
        elif method == "legacy":
            state = fit_legacy_state(fit_parts, y[fit], classes)
        else:
            state = fit_normalized_state(fit_parts, y[fit], classes)
        normalized_audit = state["audit"]
        fit_count = 0 if reused else state["audit"]["fit_count"]
        if method == "mcc":
            state = fit_mcc_state(fit_parts, y[fit], classes, state)
            fit_count += state["audit"]["fit_count"]
        predictions[valid] = predict_state(valid_parts, state, classes)
        audit["states"].append(state)
        audit["folds"].append({"fold": number, "fit_positions": fit.tolist(),
                                "valid_positions": valid.tolist(),
                                "fit_count": fit_count,
                                "reused_normalized_state": reused,
                                "initial_meta_audit": normalized_audit,
                                "final_meta_audit": state["audit"]})
        audit["fit_count"] += fit_count
    audit["macro_f1"] = float(f1_score(y, classes[predictions.argmax(1)],
                                        labels=classes, average="macro", zero_division=0))
    return predictions, audit


def _confusion(indices, predictions, count):
    return np.bincount(indices * count + predictions,
                       minlength=count * count).reshape(count, count).astype(float)


def _macro_from_confusion(confusion):
    denominator = confusion.sum(0) + confusion.sum(1)
    return float(np.divide(2 * np.diag(confusion), denominator,
                           out=np.zeros(len(confusion)), where=denominator > 0).mean())


def affine_utilities(probabilities, rule, classes=None):
    classes = _classes(rule["classes"] if classes is None else classes)
    if not np.array_equal(rule["classes"], classes):
        raise ValueError("Affine class order differs from fitted classes")
    p = align_probabilities(probabilities, classes, classes)
    a, b = np.asarray(rule["a"], dtype=float), np.asarray(rule["b"], dtype=float)
    if (a.shape != (len(classes),) or b.shape != a.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a <= 0) or np.any(b < 0)):
        raise ValueError("Invalid affine gain or false-positive penalty")
    return p * a[None, :] - b[None, :]


def predict_affine(probabilities, rule, classes=None):
    """Softmax utilities preserve the decision; values are NOT calibrated posteriors."""
    return _softmax(affine_utilities(probabilities, rule, classes))


def predict_affine_labels(probabilities, rule, classes=None):
    labels = _classes(rule["classes"] if classes is None else classes)
    return labels[affine_utilities(probabilities, rule, labels).argmax(1)]


def fit_affine_rule(y, crossfit_probabilities, classes):
    """H77: 50 confusion-gradient updates, best train rule including identity."""
    classes = _classes(classes)
    y, indices = _targets(y, classes)
    p = align_probabilities(crossfit_probabilities, classes, classes)
    if len(p) != len(y):
        raise ValueError("Target and cross-fitted probability row counts differ")
    count = len(classes)
    initial = _confusion(indices, p.argmax(1), count)
    best_score = _macro_from_confusion(initial)
    best = {"classes": classes.tolist(), "a": np.ones(count).tolist(),
            "b": np.zeros(count).tolist(), "chosen_iteration": -1}
    confusion = initial.copy()
    trajectory = [{"iteration": -1, "macro_f1": best_score}]
    for iteration in range(50):
        # One unit of prior mass spread over the diagonal keeps every class's
        # gain finite while retaining fixed true marginals across updates.
        smooth = confusion + np.eye(count) / count
        denominator = smooth.sum(0) + smooth.sum(1)
        a = 2 / denominator
        b = 2 * np.diag(smooth) / denominator ** 2
        candidate = p * a[None, :] - b[None, :]
        observed = _confusion(indices, candidate.argmax(1), count)
        score = _macro_from_confusion(observed)
        trajectory.append({"iteration": iteration, "macro_f1": score})
        if score > best_score + 1e-12:
            best_score = score
            best.update(a=a.tolist(), b=b.tolist(), chosen_iteration=iteration)
        gamma = 2 / (iteration + 2)
        confusion = (1 - gamma) * confusion + gamma * observed
    best["audit"] = {"mechanism": "affine_macro_f1_confusion_gradient_adaptation",
                     "selection_source": "training_meta_crossfit_only", "steps": 50,
                     "pseudocount_total": 1., "pseudocount_allocation": "uniform_diagonal",
                     "step_size": "2/(iteration+2)", "macro_f1": best_score,
                     "identity_macro_f1": trajectory[0]["macro_f1"],
                     "trajectory": trajectory, "calibrated_posteriors": False}
    return best
