"""Fixed H84 train-only hierarchical signature counts and teacher pooling.

This is an empirical backoff posterior estimator, not an independence-based
Bayesian combination. No teacher is fitted, calibrated or stored. Only the nine
explicit network fields participate; identifiers and other inputs are ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORIES = ('proto', 'service', 'state')
TTL = ('sttl', 'dttl')
PACKETS = ('spkts', 'dpkts')
BYTES = ('sbytes', 'dbytes')
SIGNATURE_FIELDS = CATEGORIES + TTL + PACKETS + BYTES
LEVEL_WIDTHS = (3, 5, 7, 9)


def _frame(X):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError('X must be a DataFrame with unique columns')
    if any(column not in X for column in SIGNATURE_FIELDS):
        raise ValueError('All nine signature fields are required')


def _categorical(value):
    return ('missing',) if pd.isna(value) else ('value', str(value))


def _numeric(value, ttl=False):
    if not np.isfinite(value):
        return ('missing',)
    if value < 0:
        return ('invalid_negative',)
    return ('bin', int(np.floor(value / 32. if ttl else np.log2(1. + value))))


def _keys(X):
    _frame(X)
    columns = [[_categorical(value) for value in X[column]] for column in CATEGORIES]
    for column in TTL + PACKETS + BYTES:
        numeric = pd.to_numeric(X[column], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
        columns.append([_numeric(value, ttl=column in TTL) for value in numeric])
    return [tuple(row) for row in zip(*columns)]


def _teacher(probabilities, rows, classes):
    if probabilities is None:
        raise ValueError('Explicit teacher probabilities are required')
    if isinstance(probabilities, pd.DataFrame):
        if (not probabilities.columns.is_unique
                or list(probabilities.columns) != classes.tolist()):
            raise ValueError('Teacher columns must exactly follow sorted training classes')
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.shape != (rows, len(classes)) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1 + 1e-7)
            or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=1e-6)):
        raise ValueError('Teacher must contain finite normalized N by K probabilities')
    # Preserve the unchanged fallback exactly, including harmless roundoff.
    return p


class SignaturePosteriorClassifier:
    """Most-specific supported signature posterior, with a fixed teacher weight."""

    def configuration(self):
        return {'hypothesis': 'HYP-084', 'strategy': 'hierarchical_signature_posterior_pooling',
            'version': 1, 'signature_fields': list(SIGNATURE_FIELDS),
            'levels': [list(SIGNATURE_FIELDS[:width]) for width in LEVEL_WIDTHS],
            'categorical': 'exact_tagged_tuple', 'ttl_bins': 'floor(value/32)',
            'packet_and_byte_bins': 'floor(log2(1+value))',
            'numeric_missing': 'nonfinite_or_nonnumeric_distinct_from_zero',
            'numeric_negative': 'distinct_invalid_token',
            'selection': 'most_specific_training_support_at_least_8',
            'minimum_support': 8, 'prior': 'unweighted_empirical_training_class_prior',
            'prior_strength': 20., 'posterior': '(counts+20*pi)/(n+20)',
            'alpha': '.5*n/(n+50)', 'maximum_alpha': .5,
            'pooling': '(1-alpha)*teacher+alpha*signature_posterior',
            'unseen_or_unsupported': 'unchanged_teacher',
            'teacher_fit_usage': 'validation_only_no_statistics_or_weight_selection',
            'class_order': 'sorted_lexical_training_classes',
            'query_statistics': False, 'id_as_feature': False, 'row_deletion': False}

    def fit(self, X, y, teacher_probabilities=None):
        _frame(X)
        target = np.asarray(y)
        if (target.ndim != 1 or len(target) != len(X) or not len(target)
                or pd.isna(target).any()
                or not all(isinstance(label, (str, np.str_)) for label in target)):
            raise ValueError('Aligned nonempty complete lexical string labels required')
        target = target.astype(str)
        classes, encoded, counts = np.unique(target, return_inverse=True, return_counts=True)
        _teacher(teacher_probabilities, len(X), classes)
        keys = _keys(X)
        tables = [{} for _ in LEVEL_WIDTHS]
        for key, label in zip(keys, encoded):
            for table, width in zip(tables, LEVEL_WIDTHS):
                signature = key[:width]
                if signature not in table:
                    table[signature] = np.zeros(len(classes), dtype=np.int64)
                table[signature][label] += 1
        self.classes_, self.tables_ = classes, tables
        self.class_prior_ = counts.astype(float) / len(target)
        self.audit_ = {'configuration': self.configuration(), 'fit_rows': len(target),
            'removed_rows': 0, 'classes': classes.tolist(), 'class_counts': counts.tolist(),
            'class_prior': self.class_prior_.tolist(), 'level_table_sizes': [len(table) for table in tables],
            'statistics_source': 'training_only', 'teacher_training_values_used': False,
            'teacher_validated_only': True, 'teacher_provenance': 'caller_responsibility',
            'query_statistics': False, 'id_as_feature': False, 'row_deletion': False}
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        if not hasattr(self, 'tables_'):
            raise ValueError('Fit the signature classifier before predicting')
        _frame(X)
        teacher = _teacher(teacher_probabilities, len(X), self.classes_)
        result = teacher.copy()
        for row, key in enumerate(_keys(X)):
            for table, width in zip(reversed(self.tables_), reversed(LEVEL_WIDTHS)):
                counts = table.get(key[:width])
                if counts is None:
                    continue
                support = int(counts.sum())
                if support < 8:
                    continue
                posterior = (counts + 20. * self.class_prior_) / (support + 20.)
                alpha = .5 * support / (support + 50.)
                result[row] = (1. - alpha) * teacher[row] + alpha * posterior
                break
        return result

    def predict(self, X, teacher_probabilities=None):
        return self.classes_[self.predict_proba(X, teacher_probabilities).argmax(axis=1)]
