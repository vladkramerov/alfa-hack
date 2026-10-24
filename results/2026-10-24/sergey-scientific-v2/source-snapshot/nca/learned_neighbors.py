"""H82 bounded CPU adaptation of ModernNCA; no pretrained or external inputs."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.metrics import f1_score


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("H82 total 900-second model budget exhausted")


def raw_groups(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)


class FitPreprocessor:
    """Numeric medians/normal quantiles and capped category OHE, fit rows only."""
    def fit(self, X):
        if not isinstance(X, pd.DataFrame) or not len(X) or not X.columns.is_unique:
            raise ValueError("Nonempty unique-column DataFrame required")
        self.columns_ = list(X.columns)
        self.numeric_ = list(X.select_dtypes(include="number").columns)
        self.categorical_ = [c for c in X if c not in self.numeric_]
        self.medians_ = X[self.numeric_].replace([np.inf, -np.inf], np.nan).median().fillna(0)
        self.quantile_ = None
        if self.numeric_:
            values = self._numeric(X)
            self.quantile_ = QuantileTransformer(n_quantiles=min(1000, len(X)),
                output_distribution="normal", subsample=None, random_state=42).fit(values)
        self.encoder_ = None
        if self.categorical_:
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", max_categories=64,
                                          sparse_output=False, dtype=np.float32).fit(self._categorical(X))
        self.audit_ = {"fit_rows": len(X), "numeric": self.numeric_,
            "categorical": self.categorical_, "numeric_medians": self.medians_.to_dict(),
            "quantiles": min(1000, len(X)), "max_categories": 64,
            "fit_only": True, "output_distribution": "normal"}
        self.features_ = self.transform(X).shape[1]
        if not self.features_:
            raise ValueError("No usable input features")
        self.audit_["features"] = self.features_
        return self

    def _numeric(self, X):
        return X[self.numeric_].replace([np.inf, -np.inf], np.nan).fillna(self.medians_).to_numpy(float)

    def _categorical(self, X):
        return X[self.categorical_].astype("string").fillna("__MISSING__").astype(str)

    def transform(self, X):
        if list(X.columns) != self.columns_:
            raise ValueError("Input columns differ from fitted schema")
        parts = []
        if self.quantile_ is not None:
            parts.append(self.quantile_.transform(self._numeric(X)).astype(np.float32))
        if self.encoder_ is not None:
            parts.append(self.encoder_.transform(self._categorical(X)))
        result = np.concatenate(parts, axis=1) if parts else np.empty((len(X), 0), dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite transformed features")
        return np.ascontiguousarray(result, dtype=np.float32)


class MetricEmbedding(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Linear(128, 32))
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(.1)), dtype=torch.float32))

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(.02, 2.)

    def forward(self, X):
        return torch.nn.functional.normalize(self.mapping(X), p=2, dim=1, eps=1e-12)


def soft_neighbors(query, candidates, candidate_targets, temperature, *,
                   query_groups=None, candidate_groups=None):
    """Differentiable all-candidate vote with exclusion of every same raw group."""
    distance = (query.square().sum(1, keepdim=True)
                + candidates.square().sum(1)[None, :] - 2 * query @ candidates.T).clamp_min(0)
    logits = -distance / temperature
    allowed = torch.ones_like(logits, dtype=torch.bool)
    if query_groups is not None or candidate_groups is not None:
        if query_groups is None or candidate_groups is None:
            raise ValueError("Both query and candidate group vectors are required")
        allowed = query_groups[:, None] != candidate_groups[None, :]
    valid = allowed.any(1)
    # All-masked rows become zero distributions and are skipped by training;
    # using a finite sentinel avoids NaNs in their softmax derivatives.
    weights = torch.softmax(logits.masked_fill(~allowed, -1e9), dim=1) * allowed
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
    return weights @ candidate_targets, valid


class LearnedNeighborClassifier:
    def __init__(self):
        self.model_ = None

    def _prepare(self, X, y):
        self.classes_, encoded = np.unique(np.asarray(y), return_inverse=True)
        if len(encoded) != len(X) or not len(X) or len(self.classes_) < 2:
            raise ValueError("At least two training classes required")
        self.preprocessor_ = FitPreprocessor().fit(X)
        self.values_ = torch.from_numpy(self.preprocessor_.transform(X))
        self.targets_ = torch.from_numpy(encoded.astype(np.int64))
        self.one_hot_ = torch.nn.functional.one_hot(self.targets_, len(self.classes_)).float()
        # Preserve bit identities through a signed int view supported by torch.
        self.groups_ = torch.from_numpy(raw_groups(X).view(np.int64).copy())
        counts = np.bincount(encoded, minlength=len(self.classes_))
        weight = 1 / np.sqrt(counts.astype(float))
        weight *= len(encoded) / float(counts @ weight)
        self.class_weights_ = torch.tensor(weight, dtype=torch.float32)
        torch.manual_seed(42)
        self.model_ = MetricEmbedding(self.values_.shape[1])
        self.audit_ = {"mechanism": "ModernNCA_inspired_own_CPU_adaptation", "fit_rows": len(X),
            "architecture": [self.values_.shape[1], 128, 32], "embedding_l2_normalized": True,
            "temperature_initial": .1, "temperature_bounds": [.02, 2.],
            "candidate_count": min(4096, len(X)), "batch_size": 256,
            "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0003,
            "normalized_class_weights": weight.tolist(), "group_mask": "all_identical_raw_X",
            "preprocessing": self.preprocessor_.audit_, "training_trace": [], "skipped_all_masked_rows": 0}

    def _memory(self, deadline):
        self.model_.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(self.values_), 4096):
                check_deadline(deadline)
                parts.append(self.model_(self.values_[start:start + 4096]))
        return torch.cat(parts)

    def fit(self, X, y, *, validation=None, epochs=20, patience=4, deadline=None, progress=None):
        """With validation select epochs; without it run exactly frozen epochs.

        Validation labels are only used after each completed training epoch for
        early stopping. They never enter preprocessing, metric gradients, or
        candidate memory. Epoch choice itself is intentionally a function of
        the training task's inner validation labels, never outer labels.
        """
        check_deadline(deadline)
        if not 1 <= epochs <= 20 or patience != 4:
            raise ValueError("Registered H82 limits require epochs1..20 and patience4")
        self._prepare(X, y)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=.001, weight_decay=.0003)
        rng = np.random.RandomState(42)
        best_score, best_epoch, stagnant = -np.inf, 1, 0
        for epoch in range(1, epochs + 1):
            self.model_.train()
            permutation = rng.permutation(len(self.targets_))
            total_loss, seen = 0., 0
            for offset in range(0, len(permutation), 256):
                check_deadline(deadline)
                query_positions = permutation[offset:offset + 256]
                candidate_positions = rng.choice(len(self.targets_),
                    size=min(4096, len(self.targets_)), replace=False)
                qi, ci = torch.from_numpy(query_positions), torch.from_numpy(candidate_positions)
                optimizer.zero_grad()
                query = self.model_(self.values_[qi])
                candidates = self.model_(self.values_[ci])
                p, valid = soft_neighbors(query, candidates, self.one_hot_[ci], self.model_.temperature,
                    query_groups=self.groups_[qi], candidate_groups=self.groups_[ci])
                self.audit_["skipped_all_masked_rows"] += int((~valid).sum())
                if not bool(valid.any()):
                    continue
                truth = self.targets_[qi][valid]
                weights = self.class_weights_[truth]
                probability = p[valid].gather(1, truth[:, None])[:, 0].clamp_min(1e-8)
                loss = (-probability.log() * weights).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite learned-neighbor objective")
                loss.backward()
                if any(param.grad is not None and not bool(torch.isfinite(param.grad).all())
                       for param in self.model_.parameters()):
                    raise FloatingPointError("Nonfinite learned-neighbor gradient")
                optimizer.step()
                total_loss += float(loss.detach()) * int(valid.sum())
                seen += int(valid.sum())
            if not seen:
                raise ValueError("No cross-group candidate neighbors available")
            record = {"epoch": epoch, "loss": total_loss / seen,
                      "temperature": float(self.model_.temperature.detach())}
            if validation is not None:
                self.memory_ = self._memory(deadline)
                p = self.predict_proba(validation[0], deadline=deadline)
                predicted = self.classes_[p.argmax(1)]
                score = float(f1_score(validation[1], predicted, labels=self.classes_,
                                        average="macro", zero_division=0))
                record["inner_macro_f1"] = score
                if score > best_score + 1e-12:
                    best_score, best_epoch, stagnant = score, epoch, 0
                else:
                    stagnant += 1
            self.audit_["training_trace"].append(record)
            if progress is not None:
                progress(dict(record))
            if validation is not None and stagnant >= patience:
                break
        self.audit_.update(epochs_run=epoch, selected_epoch=best_epoch if validation is not None else epochs,
            inner_score=None if validation is None else best_score,
            final_temperature=float(self.model_.temperature.detach()),
            selection="inner_only_macro_f1" if validation is not None else "frozen_epoch_refit",
            parameter_count=sum(p.numel() for p in self.model_.parameters()))
        self.memory_ = self._memory(deadline)
        self.model_.eval()
        return self

    def predict_proba(self, X, *, deadline=None, query_chunk=256, memory_chunk=8192):
        """Exact soft vote over ALL current training memory, bounded allocation."""
        if self.model_ is None or not hasattr(self, "memory_"):
            raise ValueError("A fitted metric and its training memory are required")
        if query_chunk < 1 or memory_chunk < 1:
            raise ValueError("Positive inference block sizes required")
        transformed = torch.from_numpy(self.preprocessor_.transform(X))
        output = np.empty((len(X), len(self.classes_)), dtype=np.float64)
        self.model_.eval()
        with torch.no_grad():
            temperature = self.model_.temperature
            for start in range(0, len(X), query_chunk):
                check_deadline(deadline)
                q = self.model_(transformed[start:start + query_chunk])
                q2 = q.square().sum(1, keepdim=True)
                maximum = torch.full((len(q), 1), -torch.inf)
                denominator = torch.zeros((len(q), 1))
                numerator = torch.zeros((len(q), len(self.classes_)))
                for offset in range(0, len(self.memory_), memory_chunk):
                    check_deadline(deadline)
                    candidate = self.memory_[offset:offset + memory_chunk]
                    distance = (q2 + candidate.square().sum(1)[None, :] - 2 * q @ candidate.T).clamp_min(0)
                    logits = -distance / temperature
                    new_maximum = torch.maximum(maximum, logits.max(1, keepdim=True).values)
                    old_scale = torch.exp(maximum - new_maximum)
                    weight = torch.exp(logits - new_maximum)
                    numerator = numerator * old_scale + weight @ self.one_hot_[offset:offset + memory_chunk]
                    denominator = denominator * old_scale + weight.sum(1, keepdim=True)
                    maximum = new_maximum
                prediction = (numerator / denominator.clamp_min(1e-12)).numpy().astype(float)
                prediction /= prediction.sum(1, keepdims=True)
                output[start:start + len(q)] = prediction
        if not np.isfinite(output).all() or np.any(output < 0):
            raise FloatingPointError("Nonfinite learned-neighbor probabilities")
        return output

    def predict(self, X, **kwargs):
        return self.classes_[self.predict_proba(X, **kwargs).argmax(1)]
