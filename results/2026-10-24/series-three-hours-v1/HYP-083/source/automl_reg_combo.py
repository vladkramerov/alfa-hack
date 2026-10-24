"""Portable train-only regression combinations for SPEC-008.

Every component owns its feature statistics and encoder. Model fitting is
registered centrally, including expert and gate fits.
"""
import numpy as np
import pandas as pd

from automl_feature_sets import make_features


def _fit_cached_estimator(model, X, y, configuration):
    """Cache each estimator immediately; semantic identity is shared with guard."""
    import os
    from pathlib import Path
    from automl_combo_memory import fit_once, fit_fingerprint
    directory = os.environ.get('COMBO_RUN_DIR')
    if not directory:
        return fit_once(model, X, y, configuration)
    import joblib
    path = Path(directory) / 'component-models' / (fit_fingerprint(X, y, configuration) + '.joblib')
    if path.exists():
        return joblib.load(path)
    fitted = fit_once(model, X, y, configuration)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.joblib')
    joblib.dump(fitted, temporary, compress=3)
    temporary.replace(path)
    return fitted


class FeatureComposition:
    def __init__(self, numbers=()):
        self.numbers = tuple(numbers)

    def fit(self, X, y=None):
        self.transforms_ = []
        frame = X
        for number in self.numbers:
            transform = make_features(number).fit(frame)
            frame = transform.transform(frame)
            self.transforms_.append(transform)
        self.audit_ = {'numbers': list(self.numbers), 'fit_rows': len(X), 'target_used': False}
        return self

    def transform(self, X):
        frame = X
        for transform in self.transforms_:
            frame = transform.transform(frame)
        return frame.copy()


class RegressionComponent:
    def __init__(self, features=(), scale=False, target_transform='log1p', parameters=None, threads=4):
        self.features, self.scale, self.target_transform = tuple(features), scale, target_transform
        self.parameters, self.threads = dict(parameters or {}), threads

    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesRegressor
        from automl_models import Encoder
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all() or (self.target_transform == 'log1p' and np.any(y <= -1)):
            raise ValueError('Invalid regression target domain')
        self.feature_ = FeatureComposition(self.features).fit(X)
        frame = self.feature_.transform(X)
        self.encoder_ = Encoder(native=False, scale=self.scale)
        values = self.encoder_.fit_transform(frame, y)
        parameters = dict(n_estimators=200, max_depth=24, min_samples_leaf=2, max_features=.8)
        parameters.update(self.parameters)
        self.model_ = ExtraTreesRegressor(**parameters, random_state=42, n_jobs=self.threads)
        configuration = {'component': 'regression-extra-trees', 'model_class': type(self.model_).__module__ + '.' + type(self.model_).__name__, 'features': list(self.features),
                         'scale': self.scale, 'target_transform': self.target_transform,
                         'parameters': self.model_.get_params(deep=False), 'columns': list(frame.columns)}
        target = np.log1p(y) if self.target_transform == 'log1p' else y
        self.model_ = _fit_cached_estimator(self.model_, values, target, configuration)
        self.audit_ = {**configuration, 'fit_retained_rows': len(X)}
        return self

    def predict(self, X):
        values = self.encoder_.transform(self.feature_.transform(X))
        result = np.asarray(self.model_.predict(values), dtype=float)
        if self.target_transform == 'log1p':
            result = np.expm1(result)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite component prediction')
        return result


def _rows(model, X, strategy):
    model.keep_mask_ = np.ones(len(X), dtype=bool)
    model.removed_positions_ = np.empty(0, dtype=int)
    model.audit_ = {'strategy': strategy, 'fit_original_rows': len(X), 'fit_retained_rows': len(X),
                    'row_deletion': False, 'validation_target_used': False}


class FeatureGroupExperts:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y, global_model=None):
        y = np.asarray(y)
        features = self.candidate.get('feature_numbers', [])
        parameters = self.candidate.get('parameters', {})
        self.group_column_ = self.candidate.get('group_column', 'building_class')
        minimum = int(self.candidate.get('minimum_expert_rows', 500))
        shrinkage = float(self.candidate.get('shrinkage', 500.))
        if minimum < 500 or not np.isfinite(shrinkage) or shrinkage <= 0:
            raise ValueError('Invalid group expert minimum/shrinkage')
        if global_model is not None and features:
            raise ValueError('Raw legacy global reuse is incompatible with new features')
        self.global_ = global_model if global_model is not None else RegressionComponent(features, False, parameters=parameters, threads=self.threads).fit(X, y)
        self.experts_, self.weights_ = {}, {}
        groups = sorted(X[self.group_column_].dropna().unique(), key=str) if self.group_column_ in X else []
        records = []
        if len(groups) >= 2:
            for group in groups:
                positions = np.flatnonzero(X[self.group_column_].eq(group).fillna(False).to_numpy(bool))
                weight = len(positions) / (len(positions) + shrinkage)
                trained = len(positions) >= minimum
                records.append({'group': str(group), 'fit_rows': len(positions), 'weight': weight, 'fitted': trained})
                if trained:
                    self.experts_[group] = RegressionComponent(features, False, parameters=parameters, threads=self.threads).fit(X.iloc[positions], y[positions])
                    self.weights_[group] = weight
        _rows(self, X, 'reg_feature_experts')
        self.audit_.update(feature_numbers=list(features), experts=records, group_column=self.group_column_,
                           minimum_expert_rows=minimum, shrinkage=shrinkage, global_reused=global_model is not None,
                           encoder={'scale': False, 'native': False}, global_fit_rows=len(X))
        return self

    def predict(self, X):
        prediction = np.asarray(self.global_.predict(X), dtype=float).copy()
        if self.group_column_ in X:
            for group, expert in self.experts_.items():
                positions = np.flatnonzero(X[self.group_column_].eq(group).fillna(False).to_numpy(bool))
                if len(positions):
                    weight = self.weights_[group]
                    prediction[positions] = weight * expert.predict(X.iloc[positions]) + (1 - weight) * prediction[positions]
        return prediction


def temporal_folds(X):
    years = pd.to_numeric(X['Year_Factor'], errors='raise').to_numpy()
    values = sorted(np.unique(years))
    if len(values) < 3:
        raise ValueError('Temporal nested validation requires at least three years')
    return [(np.flatnonzero(years < year), np.flatnonzero(years == year)) for year in values[-2:]]


def validate_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-8):
        raise ValueError('Blend weights must be four nonnegative finite values summing to one')
    return weights


def choose_convex_weights(y, predictions, fold_ids):
    from scipy.optimize import minimize
    y, predictions, fold_ids = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float), np.asarray(fold_ids)
    mask = fold_ids >= 0
    if predictions.shape != (len(y), 4) or not mask.any() or not np.isfinite(predictions[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Invalid convex-blend OOF data')
    folds = sorted(np.unique(fold_ids[mask]))
    # Constant response scaling preserves the convex optimum while preventing
    # SLSQP's absolute tolerances from depending on the target's physical unit.
    scale = max(float(np.max(np.abs(y[mask]))), float(np.max(np.abs(predictions[mask])))) or 1.
    subsets = [(predictions[fold_ids == fold] / scale, y[fold_ids == fold] / scale) for fold in folds]
    def objective(weights):
        return float(np.mean([np.mean((matrix @ weights - target) ** 2) for matrix, target in subsets]))
    def gradient(weights):
        return np.mean([2 * matrix.T @ (matrix @ weights - target) / len(target) for matrix, target in subsets], axis=0)
    result = minimize(objective, np.full(4, .25), method='SLSQP', bounds=[(0., 1.)] * 4,
                      jac=gradient,
                      constraints={'type': 'eq', 'fun': lambda weights: weights.sum() - 1.},
                      options={'maxiter': 1000, 'ftol': 1e-12})
    if not result.success:
        raise ValueError('Convex OOF optimization failed: ' + result.message)
    weights = np.clip(result.x, 0., 1.)
    weights /= weights.sum()
    validate_weights(weights)
    return weights, {'selection_source': 'training_temporal_oof', 'objective': 'mean_fold_mse',
                     'fold_weighting': 'equal', 'inner_mse': objective(weights) * scale ** 2, 'optimizer': 'SLSQP',
                     'objective_scale': scale,
                     'optimizer_success': True, 'folds': list(map(int, folds)), 'weights': weights.tolist()}


def regression_components(parameters=None, threads=4):
    return [RegressionComponent(parameters=parameters, threads=threads),
            FeatureGroupExperts({'feature_numbers': [], 'parameters': parameters or {}}, threads),
            RegressionComponent([30], True, parameters=parameters, threads=threads),
            RegressionComponent([31], True, parameters=parameters, threads=threads)]


class ConvexRegressionBlend:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y):
        y = np.asarray(y)
        supplied = self.candidate.get('blend_weights')
        parameters = self.candidate.get('parameters', {})
        if supplied is None:
            predictions, fold_ids = np.full((len(X), 4), np.nan), np.full(len(X), -1)
            for number, (fit, valid) in enumerate(temporal_folds(X)):
                models = regression_components(parameters, self.threads)
                # Share the exact raw global model, rather than training it twice.
                models[0].fit(X.iloc[fit], y[fit])
                models[1].fit(X.iloc[fit], y[fit], global_model=models[0])
                for model in models[2:]:
                    model.fit(X.iloc[fit], y[fit])
                predictions[valid] = np.column_stack([model.predict(X.iloc[valid]) for model in models])
                fold_ids[valid] = number
            weights, selection = choose_convex_weights(y, predictions, fold_ids)
        else:
            weights = validate_weights(supplied)
            selection = {'selection_source': 'frozen_candidate', 'weights': weights.tolist()}
        self.weights_ = weights
        self.models_ = regression_components(parameters, self.threads)
        self.models_[0].fit(X, y)
        self.models_[1].fit(X, y, global_model=self.models_[0])
        for model in self.models_[2:]:
            model.fit(X, y)
        _rows(self, X, 'reg_convex_blend')
        self.audit_.update(blend_weights=weights.tolist(), selection=selection, component_order=['raw', 'experts', 'season', 'relative'])
        return self

    def predict(self, X):
        return np.column_stack([model.predict(X) for model in self.models_]) @ self.weights_


class TargetTailMixture:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = dict(candidate), threads

    def fit(self, X, y):
        from xgboost import XGBClassifier
        from automl_models import Encoder
        y = np.asarray(y, dtype=float)
        quantile = float(self.candidate.get('tail_quantile', .8))
        if not 0 < quantile < 1 or not np.isfinite(y).all():
            raise ValueError('Invalid target-tail quantile or targets')
        self.threshold_ = float(np.quantile(y, quantile))
        high = y > self.threshold_
        if not high.any() or high.all():
            raise ValueError('Target-tail mixture requires two nonempty training regimes')
        self.gate_encoder_ = Encoder(native=False, scale=False)
        values = self.gate_encoder_.fit_transform(X, high.astype(int))
        gate_parameters = dict(n_estimators=600, max_depth=6, learning_rate=.05, reg_lambda=1., tree_method='hist', device='cpu')
        gate_parameters.update(self.candidate.get('gate_parameters', {}))
        gate = XGBClassifier(**gate_parameters, random_state=42, n_jobs=self.threads)
        self.gate_ = _fit_cached_estimator(gate, values, high.astype(int), {'component': 'target-tail-gate',
                                                              'model_class': type(gate).__module__ + '.' + type(gate).__name__, 'parameters': gate.get_params(deep=False),
                                                              'tail_quantile': quantile, 'threshold': self.threshold_, 'columns': list(X.columns)})
        parameters = self.candidate.get('parameters', {})
        self.low_ = RegressionComponent(parameters=parameters, threads=self.threads).fit(X.iloc[np.flatnonzero(~high)], y[~high])
        self.high_ = RegressionComponent(target_transform='none', parameters=parameters, threads=self.threads).fit(X.iloc[np.flatnonzero(high)], y[high])
        _rows(self, X, 'target_tail_mixture')
        self.audit_.update(tail_quantile=quantile, threshold=self.threshold_, threshold_source='training_target_quantile',
                           low_rows=int((~high).sum()), high_rows=int(high.sum()), gate_parameters=gate_parameters,
                           low_target_transform='log1p', high_target_transform='none')
        return self

    def tail_probability(self, X):
        return np.asarray(self.gate_.predict_proba(self.gate_encoder_.transform(X)))[:, 1]

    def predict(self, X):
        probability = self.tail_probability(X)
        return probability * self.high_.predict(X) + (1 - probability) * self.low_.predict(X)


def build_reg_combo(candidate, threads=4):
    candidate = dict(candidate)
    classes = {'reg_feature_experts': FeatureGroupExperts, 'reg_convex_blend': ConvexRegressionBlend,
               'target_tail_mixture': TargetTailMixture}
    if candidate.get('strategy') not in classes:
        raise ValueError('Unknown regression combination strategy')
    for field, allowed in [('parameters', {'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'criterion'}),
                           ('gate_parameters', {'n_estimators', 'max_depth', 'learning_rate', 'reg_lambda', 'reg_alpha', 'subsample', 'colsample_bytree', 'tree_method', 'device'})]:
        parameters = dict(candidate.get(field, {}))
        for key in ('random_state', 'random_seed', 'seed', 'n_jobs', 'thread_count', 'verbose'):
            parameters.pop(key, None)
        if set(parameters) - allowed:
            raise ValueError('Unsupported regression combination parameter')
        for key, upper in (('n_estimators', 1200), ('max_depth', 32)):
            if key in parameters and (not isinstance(parameters[key], int) or not 1 <= parameters[key] <= upper):
                raise ValueError('Parameter exceeds worker resource bounds: ' + key)
        if parameters.get('device', 'cpu') != 'cpu':
            raise ValueError('Only CPU training is permitted')
        candidate[field] = parameters
    return classes[candidate['strategy']](candidate, threads)
