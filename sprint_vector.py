"""H101: shared vector-leaf XGBoost with exact full-batch soft-F1 gradient.

The supplied teacher must be honest grouped out-of-fold training probabilities.
This module fits one new model, never a teacher, and uses no validation labels.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
import xgboost as xgb

from sprint_ambitious import LovaszTeacherBoost
from sprint_classification import TrainOnlyEncoder, soft_macro_f1_loss_and_gradient


class VectorLeafTeacherBoost(LovaszTeacherBoost):
    """One K-dimensional leaf per shared tree, rather than K scalar trees.

    Objective = .5 mean cross-entropy + .5(1 - soft macro-F1). Its exact
    full-batch gradient is multiplied by N for boosting's summed objective.
    max(.5*p*(1-p), .001) is positive surrogate diagonal curvature, neither
    an exact Hessian nor a guaranteed upper bound. Nonconvex descent is not
    guaranteed. XGBoost's prediction already includes the query base margin.
    """
    def __init__(self, iterations=300, threads=2, max_seconds=900):
        if (not isinstance(iterations, (int, np.integer)) or iterations < 0
                or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 4
                or not np.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('Invalid vector-leaf boosting configuration')
        self.iterations, self.threads, self.max_seconds = int(iterations), int(threads), max_seconds
        self.parameters = {'booster': 'gbtree', 'multi_strategy': 'multi_output_tree',
            'tree_method': 'hist', 'device': 'cpu', 'objective': 'multi:softprob',
            'max_depth': 6, 'eta': .05, 'lambda': 5., 'alpha': 0.,
            'min_child_weight': 1., 'gamma': 0., 'max_bin': 256,
            'subsample': 1., 'colsample_bytree': 1., 'colsample_bylevel': 1.,
            'colsample_bynode': 1., 'nthread': self.threads, 'seed': 42,
            'verbosity': 0, 'disable_default_eval_metric': 1}

    def configuration(self):
        return {'hypothesis': 'HYP-101', 'family': 'shared_vector_leaf_xgboost_soft_macro_f1',
            'iterations': self.iterations, 'parameters': deepcopy(self.parameters),
            'xgboost_version': xgb.__version__,
            'objective': '0.5meanCE+0.5(1-softmacroF1)',
            'gradient': 'exact_fullbatch_logits_gradient_times_N',
            'curvature': 'positive_surrogate_max(0.5*p*(1-p),0.001)_not_exact',
            'teacher_baseline': 'caller_supplied_log_normalized_p_clip1e-12',
            'teacher_provenance_required': 'H88_reused_true_grouped_RAW_OOF_training_teacher',
            'inference': 'softmax(xgb_output_margin_including_query_base_margin_once)',
            'preprocessing': 'fitmedian_log1pnonnegative_standardize_clip8_OHE',
            'selection': 'one_fixed_configuration_all_rounds_no_validation_earlystop',
            'class_weights': None, 'query_statistics': False}

    def _objective(self, logits, training_matrix):
        labels = training_matrix.get_label()
        if not np.isfinite(labels).all() or np.any(labels != np.floor(labels)):
            raise ValueError('XGBoost labels must be integer class positions')
        y = labels.astype(np.int64)
        z = np.asarray(logits, dtype=np.float64)
        loss, gradient = soft_macro_f1_loss_and_gradient(z, y)
        self.loss_history_.append(loss)
        p = softmax(z, axis=1)
        return len(y) * gradient, np.maximum(.5 * p * (1-p), .001)

    def fit(self, X, y, teacher_probabilities=None):
        target = np.asarray(y)
        if (not isinstance(X, pd.DataFrame) or not len(X) or not X.columns.is_unique
                or target.shape != (len(X),) or pd.isna(target).any()):
            raise ValueError('Expected nonempty unique-column DataFrame and complete target')
        self.classes_, encoded = np.unique(target, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('Need at least two classes')
        teacher = self._teacher(teacher_probabilities, len(X))
        self.parameters['num_class'] = len(self.classes_)
        self.columns_ = list(X.columns)
        started = time.monotonic()
        self.encoder_ = TrainOnlyEncoder().fit(X)
        self.model_, self.loss_history_ = None, []
        if self.iterations:
            matrix = xgb.DMatrix(self.encoder_.transform(X), label=encoded,
                base_margin=np.log(np.clip(teacher, 1e-12, 1)), nthread=self.threads)
            max_seconds = self.max_seconds

            class BudgetCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    if time.monotonic() - started >= max_seconds:
                        raise TimeoutError('H101 frozen-round budget exceeded; partial model is not a score')
                    return False

            self.model_ = xgb.train(self.parameters, matrix, num_boost_round=self.iterations,
                obj=self._objective, callbacks=[BudgetCallback()])
        self.audit_ = {'fit_rows': len(X), 'classes': self.classes_.tolist(),
            'learner_fits': int(self.model_ is not None), 'removed_rows': 0,
            'rounds_completed': 0 if self.model_ is None else self.model_.num_boosted_rounds(),
            'tree_leaf_vector_size': len(self.classes_),
            'teacher_source': 'caller_supplied_true_grouped_RAW_OOF_required',
            'teacher_columns': self.classes_.tolist(), 'loss_history': list(self.loss_history_),
            'encoder': self.encoder_.audit_, 'curvature_is_exact': False,
            'training_loss_is_independent_score': False,
            'base_margin_internal_dtype': 'float32_in_XGBoost',
            'elapsed_seconds': time.monotonic()-started, 'configuration': self.configuration()}
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        if not hasattr(self, 'encoder_'):
            raise ValueError('Fit VectorLeafTeacherBoost before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query DataFrame must match fitted columns in order')
        teacher = self._teacher(teacher_probabilities, len(X))
        if self.model_ is None or not len(X):
            return teacher
        matrix = xgb.DMatrix(self.encoder_.transform(X),
            base_margin=np.log(np.clip(teacher, 1e-12, 1)), nthread=self.threads)
        # output_margin=True already includes DMatrix.base_margin; no extra teacher sum.
        logits = np.asarray(self.model_.predict(matrix, output_margin=True, strict_shape=True),
                            dtype=np.float64)
        return softmax(logits, axis=1)
