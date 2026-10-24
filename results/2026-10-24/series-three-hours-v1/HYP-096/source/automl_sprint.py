"""Portable SPEC011 adapters; no research cache paths or validation fitting.

Regression refits the original H69 pipeline before fitting a frozen correction.
Classification only accepts explicit frozen H48 meta and H75 decision states.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.special import softmax

from automl_class_combo import ClassCombination, predict_meta_state
from automl_combo_memory import fit_once
from automl_temporal_memory import TemporalProfileModel
from sprint_panel import PanelTransitionRegressor
from sprint_regression import TemporalEntityLinker


REGRESSION_STRATEGIES = {'entity_linked_history', 'panel_transition_history'}
DECISION_STRATEGY = 'affine_macro_f1_stacking'
DECISION_STRATEGIES = {DECISION_STRATEGY, 'proper_macro_f1_stacking'}


def canonical_sprint(candidate):
    c=deepcopy(candidate);strategy=c['strategy'];reg=strategy in REGRESSION_STRATEGIES
    if not reg and strategy not in DECISION_STRATEGIES:raise ValueError('Unknown sprint strategy')
    active={'strategy','family','parameters','hypothesis','name','display_name','preprocessing','scale','target_transform','max_outlier_fraction','feature_code'}
    if not reg:active.update({'meta_state','decision_state'})
    if set(c)-active or c.get('feature_code','').strip() or c.get('parameters',{})!={}:raise ValueError('Only frozen sprint protocols are supported')
    if c.get('target_transform') not in [None,'log1p' if reg else 'none']:raise ValueError('Frozen sprint target transform mismatch')
    c.update(parameters={},preprocessing=strategy.replace('_','-')+'-v1',scale=False,max_outlier_fraction=0,target_transform='log1p' if reg else 'none')
    (SprintRegression if reg else SprintDecision)(c)
    return c


def _threads(threads):
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 4:
        raise ValueError('Sprint training requires one to four CPU threads')
    return int(threads)


def _frame_target(X, y, regression=False):
    target = np.asarray(y, dtype=float if regression else None)
    if (not isinstance(X, pd.DataFrame) or not len(X) or not X.columns.is_unique
            or target.shape != (len(X),) or pd.isna(target).any()):
        raise ValueError('Expected nonempty unique-column DataFrame and complete aligned target')
    if regression and (not np.isfinite(target).all() or np.any(target < 0)):
        raise ValueError('Regression requires finite nonnegative consumption')
    return target


class _CorrectionWithBaseline:
    """Two-argument fit protocol for the central read-before-fit journal."""
    def __init__(self, correction, baseline):
        self.correction, self.baseline = correction, baseline

    def fit(self, X, y):
        self.correction.fit(X, y, base_predictor=self.baseline)
        return self.correction


class SprintRegression:
    """Fixed H70 or staged H76 history correction with freshly fitted H69 base."""
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = deepcopy(candidate), _threads(threads)
        self.strategy_ = self.candidate.get('strategy')
        if self.strategy_ not in REGRESSION_STRATEGIES:
            raise ValueError('Unknown sprint regression strategy')
        if self.candidate.get('family', 'extra_trees') != 'extra_trees':
            raise ValueError('Sprint regression requires the original ExtraTrees base')
        if self.candidate.get('parameters', {}) != {}:
            raise ValueError('Sprint regression uses only frozen default base parameters')
        allowed = {'strategy', 'family', 'parameters', 'hypothesis', 'name', 'display_name',
                   'preprocessing', 'scale', 'target_transform', 'max_outlier_fraction', 'feature_code'}
        if set(self.candidate) - allowed or self.candidate.get('feature_code', '').strip():
            raise ValueError('Sprint regression correction and features are fixed')
        if self.candidate.get('scale', False) or self.candidate.get('max_outlier_fraction', 0) != 0:
            raise ValueError('Sprint regression has fixed preprocessing and no row removal')
        if self.candidate.get('target_transform') not in (None, 'log1p'):
            raise ValueError('Sprint regression base uses its measured log1p target protocol')

    def configuration(self):
        correction = (TemporalEntityLinker(alpha=.95) if self.strategy_ == 'entity_linked_history'
                      else PanelTransitionRegressor(threads=self.threads))
        return {'strategy': self.strategy_, 'version': 1,
                'base': {'strategy': 'temporal_profile_memory', 'parameters': {},
                         'features': [29, 30, 31], 'group_column': 'building_class',
                         'minimum_expert_rows': 500, 'shrinkage': 500.,
                         'target_transform': 'log1p', 'profile_memory_version': 1},
                'correction': correction.configuration(),
                'base_data': 'same_raw_features_and_dtypes_as_original_H69',
                'source_cache_required': False}

    def fit(self, X, y):
        target = _frame_target(X, y, regression=True)
        self.columns_ = list(X.columns)
        config = self.configuration()
        self.base_ = fit_once(TemporalProfileModel({'parameters': {}}, self.threads),
                              X, target, config['base'])
        correction = (TemporalEntityLinker(alpha=.95) if self.strategy_ == 'entity_linked_history'
                      else PanelTransitionRegressor(threads=self.threads))
        self.correction_ = fit_once(_CorrectionWithBaseline(correction, self.base_), X, target,
                                   {'strategy': self.strategy_, 'version': 1,
                                    'correction': config['correction'], 'base': config['base']})
        self.keep_mask_ = np.ones(len(X), dtype=bool)
        self.removed_positions_ = np.empty(0, dtype=int)
        self.audit_ = {'strategy': self.strategy_, 'fit_original_rows': len(X),
                       'fit_retained_rows': len(X), 'row_deletion': False,
                       'validation_target_used': False, 'base': deepcopy(self.base_.audit_),
                       'correction': deepcopy(self.correction_.audit_),
                       'configuration': config, 'base_refitted_on_supplied_training_only': True}
        return self

    def predict(self, X):
        if not hasattr(self, 'correction_'):
            raise ValueError('Fit SprintRegression before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query columns must match the fitted raw feature schema')
        result = np.asarray(self.correction_.predict(X), dtype=float)
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError('Invalid sprint regression predictions')
        return result


class SprintDecision:
    """Frozen affine H75 decisions over a refitted frozen-state H48 classifier.

    Refuses generic unfrozen tasks rather than learning a corrector from
    in-sample teacher predictions. No decision coefficients are fit here.
    """
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = deepcopy(candidate), _threads(threads)
        self.strategy_ = self.candidate.get('strategy')
        if self.strategy_ not in DECISION_STRATEGIES:
            raise ValueError('Unknown sprint decision strategy')
        state, meta = self.candidate.get('decision_state'), self.candidate.get('meta_state')
        if not isinstance(state, dict) or not isinstance(meta, dict):
            raise ValueError('Explicit frozen decision_state and nonnull meta_state are required')
        try:
            self.classes_ = np.asarray(state['classes'])
            self.gain_ = np.asarray(state['gain'], dtype=float)
            self.offset_ = np.asarray(state['offset'], dtype=float)
            valid_classes = (self.classes_.ndim == 1 and len(self.classes_) > 0
                and not pd.isna(self.classes_).any()
                and np.array_equal(np.unique(self.classes_), self.classes_))
            if self.classes_.dtype.kind in 'fiu':
                valid_classes = valid_classes and np.isfinite(self.classes_).all()
            if (not valid_classes or self.gain_.shape != self.classes_.shape
                    or self.offset_.shape != self.classes_.shape
                    or not np.isfinite(self.gain_).all() or np.any(self.gain_ <= 0)
                    or not np.isfinite(self.offset_).all()):
                raise ValueError('Invalid frozen affine decision coefficients or class order')
            uniform = np.full((1, len(self.classes_)), 1 / len(self.classes_))
            predict_meta_state([uniform] * 3, meta, self.classes_)
            self.theta_ = None
            if self.strategy_ == 'proper_macro_f1_stacking':
                self.theta_ = np.asarray(state.get('theta'), dtype=float)
                if (self.theta_.shape != (len(self.classes_) + 1,)
                        or not np.isfinite(self.theta_).all() or abs(self.theta_[0]) > 2.3
                        or np.any(np.abs(self.theta_[1:]) > 5)):
                    raise ValueError('Proper decisions require frozen bounded temperature and bias theta')
        except (KeyError, TypeError) as exc:
            raise ValueError('Invalid frozen decision/meta state') from exc
        if self.candidate.get('family', 'xgboost') != 'xgboost':
            raise ValueError('Sprint decisions require the H48 XGBoost component protocol')

    def configuration(self):
        return {'strategy': self.strategy_, 'version': 1,
                'teacher': 'probability_stacking_frozen_meta_state',
                'decision': 'softmax(gain*p+offset)', 'decision_training': 'none_frozen_candidate',
                'calibration': ('softmax(logclip1e-6(p)*exp(-theta0)+theta1:)'
                                if self.theta_ is not None else 'none'),
                'meta_state': deepcopy(self.candidate['meta_state']),
                'decision_state': deepcopy(self.candidate['decision_state']),
                'parameters': deepcopy(self.candidate.get('parameters', {})),
                'query_statistics': False, 'output_is_calibrated_posterior': False}

    def fit(self, X, y):
        target = _frame_target(X, y)
        if not np.array_equal(np.unique(target), self.classes_):
            raise ValueError('Training classes must exactly match frozen decision/meta class order')
        self.columns_ = list(X.columns)
        base_candidate = deepcopy(self.candidate)
        base_candidate.pop('decision_state', None)
        base_candidate.update(strategy='probability_stacking', preprocessing='probability-stacking-v1')
        self.base_ = fit_once(ClassCombination(base_candidate, self.threads), X, target,
                              {'strategy': 'probability_stacking', 'version': 1,
                               'parameters': deepcopy(base_candidate.get('parameters', {})),
                               'meta_state': deepcopy(base_candidate['meta_state'])})
        if not np.array_equal(self.base_.classes_, self.classes_):
            raise ValueError('Refitted teacher class columns mismatch frozen decisions')
        self.learned_state_ = {'meta_state': deepcopy(self.candidate['meta_state']),
                               'decision_state': deepcopy(self.candidate['decision_state'])}
        self.keep_mask_ = np.ones(len(X), dtype=bool)
        self.removed_positions_ = np.empty(0, dtype=int)
        self.audit_ = {'strategy': self.strategy_, 'fit_original_rows': len(X),
                       'fit_retained_rows': len(X), 'decision_fit_count': 0,
                       'state_source': 'frozen_candidate', 'query_statistics': False,
                       'row_deletion': False, 'base': deepcopy(self.base_.audit_),
                       'configuration': self.configuration(), **deepcopy(self.learned_state_)}
        return self

    def predict_proba(self, X):
        if not hasattr(self, 'base_'):
            raise ValueError('Fit SprintDecision before prediction')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns_:
            raise ValueError('Query columns must match the fitted raw feature schema')
        p = np.asarray(self.base_.predict_proba(X), dtype=float)
        if (p.shape != (len(X), len(self.classes_)) or not np.isfinite(p).all()
                or np.any(p < 0) or not np.isfinite(p.sum(axis=1)).all()
                or np.any(p.sum(axis=1) <= 0)):
            raise ValueError('Refitted teacher returned invalid class probabilities')
        p = p / p.sum(axis=1, keepdims=True)
        if self.theta_ is not None:
            p = softmax(np.log(np.clip(p, 1e-6, 1)) * np.exp(-self.theta_[0]) + self.theta_[1:], axis=1)
        return softmax(self.gain_ * p + self.offset_, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
