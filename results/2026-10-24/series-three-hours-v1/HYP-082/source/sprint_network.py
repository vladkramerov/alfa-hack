"""H82 fixed, rowwise network features before the unchanged H74 learner."""
from copy import deepcopy

import numpy as np
import pandas as pd

from sprint_contextual import ContextualTeacherCorrector


class NetworkSemanticFeatures:
    """Deterministic assumptions about TTL and directional flow measurements.

    TTL initial buckets are a modelling assumption, not recovered ground truth.
    A raw zero TTL is kept, with missing derived path estimates. Schema is the
    only fitted state; no class, cohort, quantile or frequency statistics exist.
    """
    DROP = ('stcpb', 'dtcpb')
    PREFIX = 'network__'

    def configuration(self):
        return {'version': 1, 'drop_columns': list(self.DROP),
                'ttl_columns': ['sttl', 'dttl'], 'ttl_initial_buckets': [32, 64, 128, 255],
                'ttl_initial_rule': 'least_bucket_greater_or_equal_to_observed',
                'ttl_valid_range': [1, 255], 'ttl_invalid_and_zero_derived': 'missing_raw_retained',
                'hop_fraction': '(initial-observed)/initial',
                'directional_pairs': [['spkts', 'dpkts'], ['sbytes', 'dbytes'], ['sload', 'dload']],
                'directional_formula': '(source-destination)/(abs(source)+abs(destination)+1)',
                'loss_fractions': ['sloss/(spkts+1)', 'dloss/(dpkts+1)'],
                'rtt_residual': 'tcprtt-(synack+ackdat)',
                'numeric_infinity': 'missing', 'zero_indicators': False,
                'scaling': 'none', 'learned_statistics': False, 'query_cohort_statistics': False}

    @staticmethod
    def _check_frame(X):
        if (not isinstance(X, pd.DataFrame) or not X.columns.is_unique
                or not all(isinstance(c, str) for c in X)):
            raise ValueError('Expected a DataFrame with unique string columns')

    def fit(self, X, y=None):
        self._check_frame(X)
        if not len(X) or any(c.startswith(self.PREFIX) for c in X):
            raise ValueError('Need nonempty input without reserved network feature names')
        self.columns_ = list(X.columns)
        self.output_columns_ = list(self.transform(X).columns)
        self.audit_ = {'fit_rows': len(X), 'raw_columns': self.columns_.copy(),
                       'output_columns': self.output_columns_.copy(),
                       'dropped_columns': [c for c in self.DROP if c in X],
                       'generated_columns': [c for c in self.output_columns_ if c.startswith(self.PREFIX)],
                       'removed_rows': 0, 'learned_statistics': False,
                       'configuration': self.configuration()}
        return self

    def transform(self, X):
        if not hasattr(self, 'columns_'):
            raise ValueError('Fit network feature schema before transform')
        self._check_frame(X)
        if list(X.columns) != self.columns_:
            raise ValueError('Network input schema differs from fitted raw columns')
        out = X.drop(columns=[c for c in self.DROP if c in X]).copy()
        for column in out:
            if pd.api.types.is_numeric_dtype(out[column]):
                out[column] = out[column].replace([np.inf, -np.inf], np.nan)

        def numeric(column):
            return pd.to_numeric(X[column], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)

        for column in ('sttl', 'dttl'):
            if column not in X:
                continue
            observed = numeric(column)
            valid = observed.between(1, 255).to_numpy()
            initial = np.full(len(X), np.nan)
            buckets = np.array([32., 64., 128., 255.])
            initial[valid] = buckets[np.searchsorted(buckets, observed.to_numpy()[valid], side='left')]
            out[self.PREFIX + column + '_initial'] = initial
            out[self.PREFIX + column + '_hop_fraction'] = (initial - observed.to_numpy()) / initial
        for a, b, name in (('spkts', 'dpkts', 'packet'), ('sbytes', 'dbytes', 'byte'), ('sload', 'dload', 'load')):
            if a not in X or b not in X:
                continue
            source, destination = numeric(a), numeric(b)
            # Algebraically equivalent scaling avoids overflow in abs(a)+abs(b).
            scale = np.maximum(np.maximum(np.abs(source), np.abs(destination)), 1.)
            out[self.PREFIX + name + '_imbalance'] = ((source / scale - destination / scale)
                / (np.abs(source) / scale + np.abs(destination) / scale + 1 / scale))
        for loss, packets, direction in (('sloss', 'spkts', 'source'), ('dloss', 'dpkts', 'destination')):
            if loss in X and packets in X:
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    value = numeric(loss) / (numeric(packets) + 1)
                out[self.PREFIX + direction + '_loss_fraction'] = value.replace([np.inf, -np.inf], np.nan)
        if all(c in X for c in ('tcprtt', 'synack', 'ackdat')):
            with np.errstate(over='ignore', invalid='ignore'):
                residual = numeric('tcprtt') - (numeric('synack') + numeric('ackdat'))
            out[self.PREFIX + 'rtt_residual'] = residual.replace([np.inf, -np.inf], np.nan)
        return out


class NetworkSemanticTeacherCorrector:
    """Fixed semantic representation followed by unmodified CatBoost400/depth5."""
    def __init__(self, threads=2):
        if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 4:
            raise ValueError('Need one to four CPU threads')
        self.threads = int(threads)

    def configuration(self):
        learner = ContextualTeacherCorrector(threads=self.threads).configuration()
        return {'hypothesis': 'HYP-082', 'family': 'network_semantic_contextual_catboost',
                'version': 1, 'representation': NetworkSemanticFeatures().configuration(),
                'learner': learner, 'teacher_source': 'caller_supplied_provenance',
                'query_statistics': False, 'row_deletion': False,
                'parameters_selected_from_validation': False}

    def fit(self, X, y, teacher_probabilities=None):
        self.features_ = NetworkSemanticFeatures().fit(X)
        frame = self.features_.transform(X)
        self.model_ = ContextualTeacherCorrector(threads=self.threads).fit(frame, y, teacher_probabilities)
        self.classes_ = self.model_.classes_.copy()
        self.audit_ = {'fit_rows': len(X), 'fit_original_rows': len(X),
                       'fit_retained_rows': len(X), 'removed_rows': 0,
                       'raw_columns': list(X.columns), 'classes': self.classes_.tolist(),
                       'features': deepcopy(self.features_.audit_),
                       'learner': deepcopy(self.model_.audit_), 'configuration': self.configuration()}
        return self

    def predict_proba(self, X, teacher_probabilities=None):
        if not hasattr(self, 'model_'):
            raise ValueError('Fit NetworkSemanticTeacherCorrector before prediction')
        return self.model_.predict_proba(self.features_.transform(X), teacher_probabilities)

    def predict(self, X, teacher_probabilities=None):
        return self.classes_[self.predict_proba(X, teacher_probabilities).argmax(axis=1)]
