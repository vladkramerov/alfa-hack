"""H104: DART training dropout for the unchanged exact soft-F1 objective."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
import xgboost as xgb

from sprint_vector import VectorLeafTeacherBoost


class DropoutTeacherBoost(VectorLeafTeacherBoost):
    """Scalar multiclass trees with training-time dropout, fixed honest teacher.

    Inherits the exact .5CE+.5(1-soft macro-F1) gradient and positive surrogate
    curvature. Dropout removes old trees during updates and renormalizes their
    weights. Inference explicitly disables dropout and evaluates all rounds.
    """
    def __init__(self, iterations=300, threads=2, max_seconds=900):
        super().__init__(iterations=iterations, threads=threads, max_seconds=max_seconds)
        self.parameters.update(booster='dart', multi_strategy='one_output_per_tree',
            rate_drop=.1, skip_drop=.5, sample_type='uniform', normalize_type='tree', one_drop=0)

    def configuration(self):
        config = super().configuration()
        config.update(hypothesis='HYP-104', family='teacher_residual_dart_soft_macro_f1',
            tree_leaf_vector_size=1,
            dropout='training_only_uniform_tree_dropout_and_tree_weight_renormalization',
            prediction_dropout=False,
            inference='softmax(all_rounds_xgb_output_margin_with_query_base_margin_once_trainingFalse)')
        return config

    def fit(self, X, y, teacher_probabilities=None):
        super().fit(X, y, teacher_probabilities)
        self.audit_.update(tree_leaf_vector_size=1, prediction_dropout=False,
                           configuration=self.configuration())
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        if not hasattr(self, 'encoder_'):
            raise ValueError('Fit DropoutTeacherBoost before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query DataFrame must match fitted columns in order')
        teacher = self._teacher(teacher_probabilities, len(X))
        if self.model_ is None or not len(X):
            return teacher
        matrix = xgb.DMatrix(self.encoder_.transform(X),
            base_margin=np.log(np.clip(teacher, 1e-12, 1)), nthread=self.threads)
        # XGBoost already adds DMatrix.base_margin. DART dropout is training-only.
        logits = self.model_.predict(matrix, output_margin=True, strict_shape=True,
            training=False, iteration_range=(0, self.model_.num_boosted_rounds()))
        return softmax(np.asarray(logits, dtype=np.float64), axis=1)
