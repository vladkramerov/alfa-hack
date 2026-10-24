"""Portable frozen H70/H83/H85 pool; one shared freshly fitted H69 baseline."""
from copy import deepcopy

import numpy as np
import pandas as pd

from automl_combo_memory import fit_once
from automl_temporal_memory import TemporalProfileModel
from sprint_ratio_transport import OriginalLossTransport
from sprint_regression import TemporalEntityLinker
from sprint_uncertainty import ConditionalLognormalRegressor


COMPONENT_ORDER = ('H70', 'H83', 'H85')


def canonical_pool(candidate):
    """Validate explicit frozen weights; never infer or fit an ensemble state."""
    if not isinstance(candidate, dict):
        raise ValueError('Pool candidate must be a mapping')
    c = deepcopy(candidate)
    allowed = {'strategy', 'family', 'parameters', 'pool_weights', 'hypothesis', 'name', 'display_name',
               'preprocessing', 'scale', 'target_transform', 'max_outlier_fraction', 'feature_code'}
    if set(c) - allowed or c.get('strategy') != 'temporal_transport_pool':
        raise ValueError('Only the frozen temporal transport pool protocol is supported')
    if c.get('family', 'extra_trees') != 'extra_trees' or c.get('parameters', {}) != {}:
        raise ValueError('Pool requires the fixed default ExtraTrees H69 base')
    if (c.get('preprocessing') not in (None, 'temporal-transport-pool-v1')
            or c.get('target_transform') not in (None, 'none')
            or c.get('scale', False) != False or c.get('max_outlier_fraction', 0) != 0):
        raise ValueError('Pool owns every component transform and preserves all rows')
    code = c.get('feature_code', '')
    if not isinstance(code, str) or code.strip():
        raise ValueError('Generated features are forbidden for the frozen pool')
    try:
        raw = np.asarray(c['pool_weights'])
        if raw.shape != (3,) or raw.dtype.kind not in 'fiu':
            raise ValueError('Need three numeric frozen pool weights')
        weights = raw.astype(float)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Explicit frozen three-component pool weights are required') from exc
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-10)):
        raise ValueError('Frozen pool weights must form a finite nonnegative simplex')
    c.update(family='extra_trees', parameters={}, pool_weights=weights.tolist(),
             preprocessing='temporal-transport-pool-v1', target_transform='none',
             scale=False, max_outlier_fraction=0)
    return c


class _WithSharedBaseline:
    """Expose journal's two-argument fit while passing an already fitted H69."""
    def __init__(self, correction, baseline):
        self.correction, self.baseline = correction, baseline

    def fit(self, X, y):
        return self.correction.fit(X, y, base_predictor=self.baseline)


class TemporalTransportPool:
    """Refit fixed components on supplied training only; weights remain frozen."""

    def __init__(self, candidate, threads=4):
        if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 4:
            raise ValueError('Pool training requires one to four CPU threads')
        self.candidate, self.threads = canonical_pool(candidate), int(threads)
        self.weights_ = np.asarray(self.candidate['pool_weights'], float).copy()

    @staticmethod
    def _base_configuration():
        return {'strategy': 'temporal_profile_memory', 'parameters': {},
                'features': [29, 30, 31], 'group_column': 'building_class',
                'minimum_expert_rows': 500, 'shrinkage': 500.,
                'target_transform': 'log1p', 'profile_memory_version': 1}

    def _new_components(self):
        return [TemporalEntityLinker(alpha=.95), OriginalLossTransport(),
                ConditionalLognormalRegressor(threads=min(2, self.threads))]

    def configuration(self):
        return {'strategy': 'temporal_transport_pool', 'version': 1,
                'pool_weights': self.weights_.tolist(), 'weight_training': 'none_explicit_frozen_candidate',
                'component_order': list(COMPONENT_ORDER), 'base': self._base_configuration(),
                'components': [component.configuration() for component in self._new_components()],
                'shared_H69_for_components': ['H70', 'H83'], 'H85_independent': True,
                'source_cache_required': False, 'target_transform': 'component_owned',
                'query_statistics': False, 'row_deletion': False}

    def fit(self, X, y):
        target = np.asarray(y, dtype=float)
        if (not isinstance(X, pd.DataFrame) or not len(X) or not X.columns.is_unique
                or not all(isinstance(column, str) for column in X)
                or target.shape != (len(X),) or not np.isfinite(target).all() or np.any(target < 0)):
            raise ValueError('Expected nonempty raw DataFrame and finite nonnegative aligned targets')
        self.columns_ = list(X.columns)
        config = self.configuration()
        self.base_ = None
        if np.any(self.weights_[:2] > 0):
            self.base_ = fit_once(TemporalProfileModel({'parameters': {}}, self.threads),
                                  X, target, config['base'])
        self.components_ = [None, None, None]
        for i, component in enumerate(self._new_components()):
            if self.weights_[i] == 0:
                continue
            learner = _WithSharedBaseline(component, self.base_) if i < 2 else component
            stage_config = {'strategy': component.configuration()['strategy'], 'version': 1,
                            'component': config['components'][i]}
            if i < 2:
                stage_config['base'] = config['base']
            self.components_[i] = fit_once(learner, X, target, stage_config)
        self.keep_mask_ = np.ones(len(X), dtype=bool)
        self.removed_positions_ = np.empty(0, dtype=int)
        self.audit_ = {'strategy': 'temporal_transport_pool', 'fit_original_rows': len(X),
            'fit_retained_rows': len(X), 'row_deletion': False, 'validation_target_used': False,
            'pool_weights': self.weights_.tolist(), 'weight_fit_count': 0,
            'base_fit_count': int(self.base_ is not None), 'source_cache_required': False,
            'base': deepcopy(self.base_.audit_) if self.base_ is not None else None,
            'components': {name: deepcopy(model.audit_) for name, model in zip(COMPONENT_ORDER, self.components_)
                           if model is not None}, 'configuration': config}
        return self

    def predict_components(self, X):
        """Return raw component outputs in H70/H83/H85 order; skipped columns=0."""
        if not hasattr(self, 'components_'):
            raise ValueError('Fit TemporalTransportPool before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query columns must match the fitted raw feature schema')
        matrix = np.zeros((len(X), 3), dtype=float)
        for i, component in enumerate(self.components_):
            if component is None:
                continue
            prediction = np.asarray(component.predict(X), dtype=float)
            if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
                raise ValueError('Invalid pool component predictions')
            matrix[:, i] = prediction
        return matrix

    def predict(self, X):
        prediction = self.predict_components(X) @ self.weights_
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite frozen pool prediction')
        self.last_prediction_audit_ = {'query_rows': len(X),
            'active_components': [name for name, model in zip(COMPONENT_ORDER, self.components_) if model is not None],
            'pool_weights': self.weights_.tolist(), 'weight_fit_count': 0, 'query_statistics': False}
        return prediction
