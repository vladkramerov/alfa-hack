"""H74 contextual CatBoost correction with caller-audited teacher probabilities.

No teacher model is trained or loaded here. ndarray teacher columns must follow
sorted ``classes_``; a DataFrame additionally checks its explicit class columns.
"""
from copy import deepcopy
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import softmax


class ContextualTeacherCorrector:
    """Learn tree logits on top of explicit teacher logits, using raw contexts."""

    def __init__(self, iterations=400, depth=5, learning_rate=.05,
                 l2_leaf_reg=10, threads=2):
        if not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not 1 <= threads <= 4:
            raise ValueError('threads must be between 1 and 4')
        self.parameters = dict(iterations=iterations, depth=depth,
            learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
            thread_count=threads, loss_function='MultiClass', random_seed=42,
            verbose=False, allow_writing_files=False, boost_from_average=False)

    def configuration(self):
        return {'hypothesis': 'HYP-074', 'family': 'contextual_catboost_teacher_residual',
                'parameters': deepcopy(self.parameters),
                'features': 'raw_X_plus_K_teacher_logp',
                'teacher': 'caller_supplied_sorted_classes_normalize_clip1e-6',
                'training_baseline': 'log_teacher_probability',
                'inference': 'softmax(tree_RawFormulaVal_plus_query_log_teacher)',
                'class_weight': 'inverse_sqrt_class_count_mean_one',
                'preprocessing': 'numeric_inf_to_nan_categorical_tagged_strings',
                'teacher_provenance': 'caller_responsibility_no_teacher_fit_or_cache_loading'}

    def _teacher(self, probabilities, rows):
        if probabilities is None:
            raise ValueError('Explicit teacher probabilities are required')
        if isinstance(probabilities, pd.DataFrame):
            if list(probabilities.columns) != list(self.classes_):
                raise ValueError('Teacher columns must match sorted training classes')
        p = np.asarray(probabilities, dtype=np.float64)
        if (p.shape != (rows, len(self.classes_)) or not np.isfinite(p).all()
                or np.any(p < 0)):
            raise ValueError('Invalid teacher probabilities or class count')
        # Scale before summing to avoid overflow on unnormalized nonnegative input.
        largest = p.max(axis=1, keepdims=True)
        if np.any(largest <= 0):
            raise ValueError('Every teacher probability row needs positive mass')
        p = p / largest
        p /= p.sum(axis=1, keepdims=True)
        return np.log(np.clip(p, 1e-6, 1.))

    def _features(self, X, logp):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('X must have exactly the fitted DataFrame columns in order')
        frame = pd.DataFrame(index=X.index)
        for c in self.numeric_:
            frame[c] = pd.to_numeric(X[c], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
        for c in self.categorical_:
            frame[c] = X[c].map(lambda value: 'missing:' if pd.isna(value) else 'value:' + str(value))
        frame = frame.loc[:, self.columns_]
        for k in range(len(self.classes_)):
            frame[f'__teacher_logp_{k}'] = logp[:, k]
        return frame

    def fit(self, X, y, teacher_probabilities):
        if (not isinstance(X, pd.DataFrame) or not len(X)
                or not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns)):
            raise ValueError('X must be a nonempty DataFrame with unique string columns')
        if any(c.startswith('__teacher_logp_') for c in X.columns):
            raise ValueError('Raw features use a reserved teacher feature name')
        target = np.asarray(y)
        if target.shape != (len(X),) or pd.isna(target).any():
            raise ValueError('Training labels must be present for every row')
        self.classes_, encoded, counts = np.unique(target, return_inverse=True, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError('At least two training classes must be present')
        self.columns_ = list(X.columns)
        self.numeric_ = [c for c in self.columns_ if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_ = [c for c in self.columns_ if c not in self.numeric_]
        logp = self._teacher(teacher_probabilities, len(X))
        frame = self._features(X, logp)
        weights = 1 / np.sqrt(counts[encoded].astype(float))
        weights /= weights.mean()
        pool = Pool(frame, label=encoded, cat_features=self.categorical_,
                    baseline=logp, weight=weights)
        self.model_ = CatBoostClassifier(**self.parameters, classes_count=len(self.classes_))
        started = time.monotonic()
        self.model_.fit(pool)
        self.audit_ = {'fit_rows': len(X), 'classes': self.classes_.tolist(),
            'teacher_columns': self.classes_.tolist(), 'class_counts': counts.tolist(),
            'sample_weight_mean': float(weights.mean()),
            'sample_weight_min': float(weights.min()), 'sample_weight_max': float(weights.max()),
            'raw_columns': self.columns_.copy(), 'categorical_columns': self.categorical_.copy(),
            'numeric_columns': self.numeric_.copy(), 'removed_rows': 0,
            'teacher_source': 'caller_supplied_provenance',
            'elapsed_seconds': time.monotonic() - started,
            'configuration': self.configuration()}
        return self

    def predict_proba(self, X, teacher_probabilities):
        if not hasattr(self, 'model_'):
            raise ValueError('Fit the corrector before predicting')
        logp = self._teacher(teacher_probabilities, len(X))
        frame = self._features(X, logp)
        # CatBoost does not persist a training Pool baseline inside its trees.
        pool = Pool(frame, cat_features=self.categorical_)
        residual = np.asarray(self.model_.predict(pool, prediction_type='RawFormulaVal'))
        return softmax(residual + logp, axis=1)

    def predict(self, X, teacher_probabilities):
        return self.classes_[self.predict_proba(X, teacher_probabilities).argmax(axis=1)]
