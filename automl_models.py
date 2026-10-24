"""Bounded tabular model candidates used by the isolated AutoResearch worker."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FAMILIES = {'catboost', 'lightgbm', 'xgboost', 'extra_trees', 'random_forest', 'linear', 'blend'}
_PARAM_LIMITS = {
    'iterations': (50, 1600), 'depth': (2, 10), 'learning_rate': (.005, .3),
    'l2_leaf_reg': (.01, 100), 'n_estimators': (50, 1200), 'max_depth': (2, 40),
    'min_samples_leaf': (1, 40), 'min_samples_split': (2, 80),
    'max_features': (.1, 1.0), 'reg_lambda': (0, 100), 'reg_alpha': (0, 100),
    'subsample': (.5, 1.0), 'colsample_bytree': (.3, 1.0), 'num_leaves': (4, 128),
    'min_child_samples': (2, 100), 'C': (.001, 1000), 'alpha': (.0001, 1000),
}
_FAMILY_PARAMS = {
    'catboost': {'iterations', 'depth', 'learning_rate', 'l2_leaf_reg'},
    'lightgbm': {'n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 'min_child_samples', 'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha'},
    'xgboost': {'n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha'},
    'extra_trees': {'n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_features'},
    'random_forest': {'n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_features'},
    'linear': {'C', 'alpha'},
}
_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')


def _class_key(value):
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__ + ':' + repr(value)


def _bounded_number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return min(high, max(low, value))


def _feature_ops(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 24:
        raise ValueError('feature_operations must be a list of at most 24 operations')
    result = []
    allowed = {'add', 'sub', 'mul', 'div', 'log1p', 'abs'}
    names = set()
    for item in value:
        if not isinstance(item, dict) or item.get('op') not in allowed:
            raise ValueError('Unsupported feature operation')
        op, name = item['op'], item.get('name')
        left = item.get('left')
        right = item.get('right')
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name.startswith('__'):
            raise ValueError('Invalid generated feature name')
        if name in names:
            raise ValueError('Generated feature names must be unique')
        names.add(name)
        if not isinstance(left, str) or not left or len(left) > 128 or '\x00' in left:
            raise ValueError('Feature operands must name existing columns')
        if op in {'add', 'sub', 'mul', 'div'} and (not isinstance(right, str) or not right or len(right) > 128 or '\x00' in right):
            raise ValueError(f'{op} requires a column name in right')
        if op in {'abs', 'log1p'} and right is not None:
            raise ValueError(f'{op} does not accept right')
        result.append({'op': op, 'name': name, 'left': left, **({'right': right} if right is not None else {})})
    return result


def canonical_candidate(candidate):
    """Validate and normalize a candidate, rejecting direct executable code."""
    if not isinstance(candidate, dict):
        raise ValueError('candidate must be an object')
    if 'feature_code' in candidate:
        raise ValueError('feature_code is not supported; use feature_operations')
    family = candidate.get('family')
    if family not in FAMILIES:
        raise ValueError('Unsupported model family')
    result = {'family': family}
    stopping = candidate.get('early_stopping', False)
    if not isinstance(stopping, bool) or stopping and family not in {'catboost', 'lightgbm'}:
        raise ValueError('early_stopping supports CatBoost/LightGBM only')
    if stopping:
        result['early_stopping'] = True
    if candidate.get('panel_time_column') is not None:
        column = candidate['panel_time_column']
        if not isinstance(column, str) or not column or len(column) > 128 or family == 'blend':
            raise ValueError('Invalid panel_time_column')
        result['panel_time_column'] = column
    strategy = candidate.get('panel_strategy', 'features')
    if strategy not in {'features', 'hybrid'}:
        raise ValueError('Unsupported panel_strategy')
    if strategy == 'hybrid':
        if family != 'extra_trees' or not result.get('panel_time_column') or candidate.get('expert_column'):
            raise ValueError('Hybrid panel requires a global ExtraTrees temporal candidate')
        result['panel_strategy'] = strategy
    view = candidate.get('numeric_view', 'none')
    if view not in {'none', 'rank', 'signed_log', 'projection', 'rank_projection'}:
        raise ValueError('Unsupported numeric_view')
    if view != 'none':
        if family == 'blend':
            raise ValueError('Numeric views belong in explicit blend members')
        result['numeric_view'] = view
    if family == 'catboost':
        encoding = candidate.get('catboost_encoding', 'onehot')
        if encoding not in {'native', 'onehot'}:
            raise ValueError('catboost_encoding must be native or onehot')
        result['catboost_encoding'] = encoding
    elif 'catboost_encoding' in candidate:
        raise ValueError('catboost_encoding applies only to CatBoost')
    if 'hypothesis' in candidate:
        result['hypothesis'] = str(candidate['hypothesis'])[:500]
    if candidate.get('target_transform', 'none') not in {'none', 'log1p'}:
        raise ValueError('Unsupported target_transform')
    result['target_transform'] = candidate.get('target_transform', 'none')
    result['feature_operations'] = _feature_ops(candidate.get('feature_operations'))
    drops = candidate.get('drop_columns', [])
    if not isinstance(drops, list) or len(drops) > 64 or any(not isinstance(x, str) or not x or len(x) > 128 or '\x00' in x for x in drops):
        raise ValueError('drop_columns must contain at most 64 column names')
    result['drop_columns'] = sorted(set(drops))
    frequency = candidate.get('frequency_columns', [])
    if not isinstance(frequency, list) or len(frequency) > 12 or any(not isinstance(x, str) or not x or len(x) > 128 or '\x00' in x for x in frequency):
        raise ValueError('frequency_columns must contain at most 12 column names')
    result['frequency_columns'] = sorted(set(frequency))
    expert_column = candidate.get('expert_column')
    if expert_column is not None:
        if not isinstance(expert_column, str) or not expert_column or len(expert_column) > 128 or '\x00' in expert_column:
            raise ValueError('expert_column must name one existing column')
        result['expert_column'] = expert_column
        result['expert_min_rows'] = int(_bounded_number(candidate.get('expert_min_rows', 300), 50, 10000, 'expert_min_rows'))
        result['max_expert_groups'] = int(_bounded_number(candidate.get('max_expert_groups', 16), 2, 40, 'max_expert_groups'))
    elif any(key in candidate for key in ('expert_min_rows', 'max_expert_groups')):
        raise ValueError('expert_min_rows and max_expert_groups require expert_column')
    if family == 'blend':
        mode = candidate.get('blend_mode', 'arithmetic')
        if mode not in {'arithmetic', 'geometric'}:
            raise ValueError('Unsupported blend_mode')
        if mode != 'arithmetic':
            result['blend_mode'] = mode
        if (result['feature_operations'] or result['drop_columns'] or result['frequency_columns']
                or result['target_transform'] != 'none' or candidate.get('class_weight_power', 0)
                or expert_column is not None):
            raise ValueError('Blend transformations and group experts belong in explicit members')
        members, weights = candidate.get('members'), candidate.get('weights')
        if not isinstance(members, list) or not 2 <= len(members) <= 3:
            raise ValueError('blend requires 2 or 3 explicit members')
        if not isinstance(weights, list) or len(weights) != len(members):
            raise ValueError('blend weights must match members')
        normalized = []
        for member in members:
            if not isinstance(member, dict) or member.get('family') == 'blend':
                raise ValueError('nested blends are not supported')
            item = canonical_candidate(member)
            # Blends consume member probabilities; member-only argmax correction has no effect.
            item['decision_power'] = 0
            normalized.append(item)
        raw = [_bounded_number(x, 0, 1, 'blend weight') for x in weights]
        total = sum(raw)
        if total <= 0:
            raise ValueError('blend weights must have positive sum')
        result.update(members=normalized, weights=[round(x / total, 12) for x in raw])
    else:
        params = candidate.get('parameters', {})
        if not isinstance(params, dict):
            raise ValueError('parameters must be an object')
        allowed = _FAMILY_PARAMS[family]
        clean = {}
        for key, value in params.items():
            if key not in allowed:
                if key in {'seed', 'random_seed', 'random_state', 'n_jobs', 'thread_count', 'threads'}:
                    continue
                raise ValueError(f'Unknown parameter for {family}: {key}')
            low, high = _PARAM_LIMITS[key]
            number = -1 if key == 'max_depth' and family == 'lightgbm' and value == -1 else _bounded_number(value, low, high, key)
            clean[key] = int(round(number)) if key in {'iterations', 'n_estimators', 'depth', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'num_leaves', 'min_child_samples'} else float(number)
        defaults = {
            'catboost': {'iterations': 500, 'depth': 6, 'learning_rate': .05, 'l2_leaf_reg': 3.0},
            'lightgbm': {'n_estimators': 500, 'max_depth': -1, 'learning_rate': .05, 'num_leaves': 31, 'min_child_samples': 20, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_lambda': 0.0, 'reg_alpha': 0.0},
            'xgboost': {'n_estimators': 400, 'max_depth': 6, 'learning_rate': .05, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_lambda': 1.0, 'reg_alpha': 0.0},
            'extra_trees': {'n_estimators': 200, 'max_depth': 24, 'min_samples_leaf': 2, 'min_samples_split': 2, 'max_features': .8},
            'random_forest': {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 2, 'max_features': .8},
            'linear': {'alpha': 1.0, 'C': 1.0},
        }[family]
        defaults.update(clean)
        result['parameters'] = dict(sorted(defaults.items()))
    for key in ('class_weight_power', 'decision_power'):
        result[key] = _bounded_number(candidate.get(key, 0), 0, 1, key)
    return result


def candidate_key(candidate):
    canonical = canonical_candidate(candidate)
    def remove_hypotheses(item):
        item.pop('hypothesis', None)
        for member in item.get('members', []):
            remove_hypotheses(member)
    remove_hypotheses(canonical)
    payload = json.dumps(canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _engineer_factory(operations):
    """Compile generated code from the fixed DSL; this is not a sandbox for Python input."""
    lines = ['def engineer(X):', '    X = X.copy()']
    for item in operations:
        op, name, left = item['op'], item['name'], item['left']
        if op in {'add', 'sub', 'mul', 'div'}:
            symbol = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[op]
            expr = f"X[{left!r}] {symbol} X[{item['right']!r}]" if op != 'div' else f"X[{left!r}] / X[{item['right']!r}].replace(0, np.nan)"
        elif op == 'abs':
            expr = f"X[{left!r}].abs()"
        else:
            expr = f"np.log1p(X[{left!r}].clip(lower=0))"
        lines.append(f'    X[{name!r}] = {expr}')
    lines.append('    return X')
    source = '\n'.join(lines) + '\n'
    # This checks the generated subset as a guard against accidental compiler changes.
    tree = ast.parse(source)
    if len(list(ast.walk(tree))) > 500:
        raise ValueError('Generated feature code is too large')
    code = compile(tree, '<feature_operations>', 'exec')
    scope = {'np': np}
    scope['__builtins__'] = {}
    exec(code, scope, scope)
    exported = ('import numpy as np\n\n' + source) if operations else source
    return scope['engineer'], exported


class _FrequencyFeatures:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        self.tables_ = {}
        for col in self.columns:
            if col not in X:
                raise ValueError(f'Unknown frequency column: {col}')
            key = X[col].astype('string').fillna('__MISSING__')
            self.tables_[col] = (key.value_counts(dropna=False) / max(len(key), 1)).to_dict()
        return self

    def transform(self, X):
        out = X.copy()
        for col, table in self.tables_.items():
            key = out[col].astype('string').fillna('__MISSING__')
            out[f'{col}__frequency'] = key.map(table).fillna(0.0).astype(float)
        return out


def _numeric_frame(X):
    return X.select_dtypes(include=[np.number, 'bool']).columns.tolist()


def _clean_infinities(X):
    numeric = _numeric_frame(X)
    if numeric:
        X = X.copy()
        X.loc[:, numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    return X


class _NumericViews:
    """Append bounded numerical views; learn distributions only from fit rows."""
    def __init__(self, mode):
        self.mode = mode

    def fit(self, X):
        self.columns_ = list(X.select_dtypes(include='number').columns)[:128]
        self.tables_ = {}
        self.names_ = []
        if self.mode in {'rank', 'rank_projection'}:
            for c in self.columns_:
                values = X[c].to_numpy(dtype=float, na_value=np.nan)
                self.tables_[c] = np.sort(values[np.isfinite(values)])
        if self.mode in {'projection', 'rank_projection'} and self.columns_:
            numeric = X[self.columns_].replace([np.inf, -np.inf], np.nan)
            self.center_ = numeric.median().fillna(0).to_numpy(dtype=float)
            scale = (numeric.quantile(.75) - numeric.quantile(.25)).to_numpy(dtype=float)
            self.scale_ = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.)
            width = min(12, len(self.columns_))
            self.rotation_ = np.random.default_rng(1729).normal(size=(len(self.columns_), width)) / np.sqrt(len(self.columns_))
        if self.mode in {'rank', 'rank_projection', 'signed_log'}:
            self.names_ += [f'__research_{self.mode}_{i}' for i in range(len(self.columns_))]
        if self.mode in {'projection', 'rank_projection'} and self.columns_:
            self.names_ += [f'__research_projection_{i}' for i in range(self.rotation_.shape[1])]
        if set(self.names_) & set(X.columns):
            raise ValueError('Numeric view name collides with input column')
        return self

    def transform(self, X):
        if self.mode == 'none' or not self.columns_:
            return X
        generated = {}
        if self.mode in {'rank', 'rank_projection', 'signed_log'}:
            for i, c in enumerate(self.columns_):
                values = X[c].to_numpy(dtype=float, na_value=np.nan)
                if self.mode == 'signed_log':
                    result = np.sign(values) * np.log1p(np.abs(values))
                else:
                    table = self.tables_[c]
                    result = (np.searchsorted(table, values, side='left') + np.searchsorted(table, values, side='right')) / (2 * max(1, len(table)))
                generated[self.names_[i]] = np.where(np.isfinite(values), result, np.nan)
        if self.mode in {'projection', 'rank_projection'}:
            matrix = X[self.columns_].to_numpy(dtype=float, na_value=np.nan)
            matrix = np.clip((matrix - self.center_) / self.scale_, -10, 10)
            matrix = np.nan_to_num(matrix, nan=0., posinf=10., neginf=-10.) @ self.rotation_
            for i in range(matrix.shape[1]):
                generated[f'__research_projection_{i}'] = matrix[:, i]
        return pd.concat([X, pd.DataFrame(generated, index=X.index)], axis=1)


class CandidateModel:
    """sklearn-like estimator; all learned preprocessing is fitted on fit(X, y)."""
    def __init__(self, kind, candidate, threads=4):
        self.kind = kind
        self.candidate = canonical_candidate(candidate)
        if kind == 'regression' and self.candidate.get('blend_mode') == 'geometric':
            raise ValueError('Geometric probability pooling requires classification')
        if kind == 'regression' and (self.candidate.get('class_weight_power', 0) or self.candidate.get('decision_power', 0)):
            raise ValueError('Class weighting and decision correction require classification')
        if kind != 'regression' and self.candidate.get('target_transform') != 'none':
            raise ValueError('Target transforms require regression')
        self.threads = max(1, min(8, int(threads)))
        self.feature_code_ = ''
        self.audit_ = {}

    def _prepare_fit(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.input_columns_ = list(X.columns)
        ops = self.candidate.get('feature_operations', [])
        seen = set(self.input_columns_)
        for operation in ops:
            if (operation['left'] not in seen or
                    operation.get('right') is not None and operation['right'] not in seen):
                raise ValueError('Feature operation references an unknown column')
            if operation['name'] in seen:
                raise ValueError('Generated feature name already exists')
            seen.add(operation['name'])
        self.engineer_, generated = _engineer_factory(ops)
        self.feature_code_ = generated if ops else ''
        out = _clean_infinities(self.engineer_(X))
        requested_drops = self.candidate.get('drop_columns', [])
        missing_drops = [c for c in requested_drops if c not in out]
        if missing_drops:
            raise ValueError('Unknown drop_columns: ' + ', '.join(missing_drops[:5]))
        drops = list(requested_drops)
        out = out.drop(columns=drops)
        self.frequency_ = _FrequencyFeatures(self.candidate.get('frequency_columns', [])).fit(out)
        out = _clean_infinities(self.frequency_.transform(out))
        self.numeric_view_ = _NumericViews(self.candidate.get('numeric_view', 'none')).fit(out)
        out = self.numeric_view_.transform(out)
        if not len(out.columns):
            raise ValueError('Candidate removed every feature')
        self.columns_ = list(out.columns)
        self.family_ = self.candidate['family']
        self.audit_ = {'fit_rows': len(X), 'input_columns': list(self.input_columns_),
                       'output_columns': list(self.columns_), 'dropped_columns': drops,
                       'frequency_columns': list(self.frequency_.tables_),
                       'categorical_columns': [], 'target_transform': self.candidate.get('target_transform', 'none'),
                       'model_family': self.family_,
                       'categorical_handling': {'catboost': self.candidate.get('catboost_encoding', 'onehot'), 'lightgbm': 'native_category',
                           'xgboost': 'onehot_max_categories_32'}.get(self.family_, 'onehot')}
        return out

    def _prepare(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.input_columns_)
        out = _clean_infinities(self.engineer_(X))
        out = out.drop(columns=[c for c in self.candidate.get('drop_columns', []) if c in out])
        out = self.frequency_.transform(out)
        out = self.numeric_view_.transform(out)
        return _clean_infinities(out.reindex(columns=self.columns_))

    def fit(self, X, y, eval_set=None):
        y_series = pd.Series(y, index=X.index if isinstance(X, pd.DataFrame) else None)
        self.stopping_ = None
        if self.candidate.get('early_stopping') and eval_set is None and len(X) >= 300:
            from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
            groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
            if self.kind == 'classification':
                fit, valid = next(StratifiedGroupKFold(5, shuffle=True, random_state=73).split(X, y_series, groups))
                missing = set(y_series.iloc[valid]) - set(y_series.iloc[fit])
                if missing:
                    move = set(groups[np.flatnonzero(y_series.isin(missing).to_numpy())])
                    valid = valid[~np.isin(groups[valid], list(move))]
                    fit = np.setdiff1d(np.arange(len(X)), valid)
            else:
                fit, valid = next(GroupShuffleSplit(test_size=.2, random_state=73).split(X, y_series, groups))
            if len(valid):
                probe = CandidateModel(self.kind, {**self.candidate, 'early_stopping': False}, self.threads)
                probe.fit(X.iloc[fit], y_series.iloc[fit], eval_set=(X.iloc[valid], y_series.iloc[valid]))
                if self.candidate['family'] == 'catboost':
                    count = probe.model_.get_best_iteration() + 1
                    parameter = 'iterations'
                else:
                    count = probe.model_.best_iteration_
                    parameter = 'n_estimators'
                count = max(50, min(self.candidate['parameters'][parameter], int(count or 50)))
                self.effective_parameters_ = {**self.candidate['parameters'], parameter: count}
                self.stopping_ = {'fit_rows': len(fit), 'valid_rows': len(valid), 'iterations': count,
                                  'split': 'train-only grouped seed73', 'probe_fit_rows': probe.audit_['fit_rows']}
        self.panel_ = None
        if self.candidate.get('panel_time_column'):
            if self.kind != 'regression':
                raise ValueError('Causal panel currently supports regression only')
            from automl_panel import CausalPanelFeatures
            self.panel_ = CausalPanelFeatures(self.candidate['panel_time_column']).fit(X, y_series)
            X = self.panel_.transform(X)
        Xf = self._prepare_fit(X)
        if self.panel_ is not None:
            self.audit_['causal_panel'] = getattr(self.panel_, 'audit_', {'key_columns': []})
        if self.stopping_ is not None:
            self.audit_['early_stopping'] = self.stopping_
        family = self.candidate['family']
        if family == 'blend':
            self.members_ = [build_pipeline(self.kind, m, self.threads).fit(X, y_series) for m in self.candidate['members']]
            generated = [member.feature_code_ for member in self.members_ if member.feature_code_]
            if generated:
                self.feature_code_ = '\n\n'.join(
                    code.replace('def engineer(X):', f'def engineer_member_{i}(X):', 1)
                    for i, code in enumerate(generated, 1))
            self.classes_ = self.members_[0].classes_ if self.kind != 'regression' else None
            if self.kind != 'regression':
                labels = [_class_key(value) for value in pd.Series(y_series).tolist()]
                counts = pd.Series(labels).value_counts().reindex(list(map(_class_key, self.classes_)), fill_value=0).to_numpy(dtype=float)
                self.priors_ = counts / counts.sum()
            self.audit_['blend_members'] = len(self.members_)
            return self
        self.is_regression_ = self.kind == 'regression'
        if not self.is_regression_:
            labels = [_class_key(value) for value in y_series.tolist()]
            ordered_labels = sorted(set(labels))
            self.label_to_index_ = {label: i for i, label in enumerate(ordered_labels)}
            first_values = {}
            for original, label in zip(y_series.tolist(), labels):
                first_values.setdefault(label, original)
            self.class_values_ = [first_values[label] for label in ordered_labels]
            label_types = {type(value.item() if isinstance(value, np.generic) else value) for value in self.class_values_}
            if len(label_types) > 1:
                raise ValueError('Mixed target label types are unsupported')
            self.classes_ = np.asarray(self.class_values_)
            y_fit = np.asarray([self.label_to_index_[label] for label in labels], dtype=int)
            counts = np.bincount(y_fit, minlength=len(self.classes_)).astype(float)
            alpha = float(self.candidate.get('class_weight_power', 0.0))
            weights = np.power(np.maximum(counts, 1) ** -1, alpha)
            sample_weight = weights[y_fit]
            sample_weight /= sample_weight.mean()
            self.priors_ = counts / counts.sum()
        else:
            y_fit = np.asarray(y_series, dtype=float)
            if self.candidate.get('target_transform') == 'log1p':
                if np.any(y_fit <= -1):
                    raise ValueError('log1p target transform requires targets greater than -1')
                y_fit = np.log1p(y_fit)
            sample_weight = None
            self.classes_ = None
        self.model_, self.preprocessor_ = self._build_estimator(Xf, y_fit)
        matrix = self.preprocessor_.fit_transform(Xf)
        self.audit_['categorical_columns'] = self.categorical_columns_
        kwargs = {'sample_weight': sample_weight} if sample_weight is not None and self.candidate.get('class_weight_power', 0) else {}
        if (self.candidate['family'] == 'catboost' and self.categorical_columns_
                and self.candidate['catboost_encoding'] == 'native'):
            kwargs['cat_features'] = self.categorical_columns_
        if eval_set is not None:
            valid_X, valid_y = eval_set
            if self.panel_ is not None:
                valid_X = self.panel_.transform(valid_X)
            valid_matrix = self.preprocessor_.transform(self._prepare(valid_X))
            if self.is_regression_:
                valid_target = np.asarray(valid_y, float)
                if self.candidate.get('target_transform') == 'log1p':
                    valid_target = np.log1p(valid_target)
            else:
                valid_target = np.asarray([self.label_to_index_[_class_key(v)] for v in valid_y])
            kwargs['eval_set'] = (valid_matrix, valid_target) if family == 'catboost' else [(valid_matrix, valid_target)]
            if family == 'catboost':
                if not self.is_regression_:
                    metric = 'TotalF1:average=Macro'
                    if self.candidate.get('class_weight_power', 0):
                        metric += ';use_weights=false'
                    self.model_.set_params(eval_metric=metric)
                kwargs.update(early_stopping_rounds=60, use_best_model=True, verbose=False)
            else:
                import lightgbm as lgb
                kwargs['callbacks'] = [lgb.early_stopping(60, verbose=False)]
                if not self.is_regression_:
                    from sklearn.metrics import f1_score
                    def macro_f1(actual, probabilities):
                        labels = (probabilities > .5).astype(int) if probabilities.ndim == 1 else probabilities.argmax(axis=1)
                        return 'macro_f1', f1_score(actual, labels, labels=np.arange(len(self.classes_)), average='macro', zero_division=0), True
                    self.model_.set_params(metric='None')
                    kwargs['eval_metric'] = macro_f1
        self.model_.fit(matrix, y_fit, **kwargs)
        return self

    def _build_estimator(self, X, y):
        family = self.candidate['family']
        params = getattr(self, 'effective_parameters_', self.candidate.get('parameters', {}))
        numeric = _numeric_frame(X)
        categorical = [c for c in X if c not in numeric]
        self.categorical_columns_ = categorical
        if family in {'catboost', 'lightgbm'}:
            frame = X.copy()
            for col in numeric:
                frame[col] = pd.to_numeric(frame[col], errors='coerce')
            for col in categorical:
                frame[col] = frame[col].astype('string').fillna('__MISSING__').astype(str)
            if family == 'catboost':
                from catboost import CatBoostClassifier, CatBoostRegressor
                model_cls = CatBoostRegressor if self.is_regression_ else CatBoostClassifier
                model = model_cls(iterations=params.get('iterations', 500), depth=params.get('depth', 6),
                    learning_rate=params.get('learning_rate', .05), l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
                    thread_count=self.threads, random_seed=42, verbose=False, allow_writing_files=False,
                    loss_function='RMSE' if self.is_regression_ else 'MultiClass')
                if self.candidate['catboost_encoding'] == 'onehot':
                    return model, _preprocessor(X, numeric, categorical)
                return model, _NativeFrame(frame, categorical)
            import lightgbm as lgb
            for col in categorical:
                cats = sorted(frame[col].unique().tolist())
                frame[col] = pd.Categorical(frame[col], categories=cats)
            model_cls = lgb.LGBMRegressor if self.is_regression_ else lgb.LGBMClassifier
            model = model_cls(n_estimators=params.get('n_estimators', 500), max_depth=params.get('max_depth', -1),
                learning_rate=params.get('learning_rate', .05), num_leaves=params.get('num_leaves', 31),
                min_child_samples=params.get('min_child_samples', 20), subsample=params.get('subsample', 1.0),
                subsample_freq=1 if params.get('subsample', 1.0) < 1.0 else 0,
                colsample_bytree=params.get('colsample_bytree', 1.0), reg_lambda=params.get('reg_lambda', 0.0),
                reg_alpha=params.get('reg_alpha', 0.0), n_jobs=self.threads, random_state=42, verbosity=-1)
            return model, _NativeFrame(frame, categorical)
        if family == 'xgboost':
            from xgboost import XGBClassifier, XGBRegressor
            pre = _preprocessor(X, numeric, categorical, max_categories=32)
            model_cls = XGBRegressor if self.is_regression_ else XGBClassifier
            model = model_cls(n_estimators=params.get('n_estimators', 400), max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', .05), subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0), reg_lambda=params.get('reg_lambda', 1.0),
                reg_alpha=params.get('reg_alpha', 0.0), n_jobs=self.threads, random_state=42,
                objective='reg:squarederror' if self.is_regression_ else ('binary:logistic' if len(self.classes_) == 2 else 'multi:softprob'),
                eval_metric='rmse' if self.is_regression_ else ('logloss' if len(self.classes_) == 2 else 'mlogloss'),
                tree_method='hist', device='cpu', verbosity=0)
            if not self.is_regression_ and len(self.classes_) > 2:
                model.set_params(num_class=len(self.classes_))
            return model, pre
        pre = _preprocessor(X, numeric, categorical)
        if family == 'extra_trees':
            cls = ExtraTreesRegressor if self.is_regression_ else ExtraTreesClassifier
            model = cls(n_estimators=params.get('n_estimators', 300), max_depth=params.get('max_depth'),
                min_samples_leaf=params.get('min_samples_leaf', 1), min_samples_split=params.get('min_samples_split', 2),
                max_features=params.get('max_features', 1.0), n_jobs=self.threads, random_state=42)
        elif family == 'random_forest':
            cls = RandomForestRegressor if self.is_regression_ else RandomForestClassifier
            model = cls(n_estimators=params.get('n_estimators', 300), max_depth=params.get('max_depth'),
                min_samples_leaf=params.get('min_samples_leaf', 1), min_samples_split=params.get('min_samples_split', 2),
                max_features=params.get('max_features', 1.0), n_jobs=self.threads, random_state=42)
        else:
            model = Ridge(alpha=params.get('alpha', 1.0)) if self.is_regression_ else LogisticRegression(
                C=params.get('C', 1.0), max_iter=1200, n_jobs=self.threads, random_state=42)
        return model, pre

    def _matrix(self, X):
        if self.panel_ is not None:
            X = self.panel_.transform(X)
        prepared = self._prepare(X)
        if self.family_ in {'catboost', 'lightgbm'}:
            return self.preprocessor_.transform(prepared)
        return self.preprocessor_.transform(prepared)

    def predict(self, X):
        if self.candidate['family'] == 'blend':
            weights = self.candidate['weights']
            preds = [m.predict(X) for m in self.members_]
            if self.kind == 'regression':
                return sum(w * np.asarray(p, dtype=float) for w, p in zip(weights, preds))
            return self.classes_[self._decision(self.predict_proba(X))]
        if self.is_regression_:
            pred = np.asarray(self.model_.predict(self._matrix(X))).reshape(-1)
            if self.candidate.get('target_transform') == 'log1p':
                pred = np.expm1(pred)
            return pred
        return self.classes_[self._decision(self.predict_proba(X))]

    def predict_proba(self, X):
        if self.kind == 'regression':
            raise AttributeError('Regression candidates do not expose predict_proba')
        if self.candidate['family'] == 'blend':
            result = np.zeros((len(X), len(self.classes_)), dtype=float)
            geometric = self.candidate.get('blend_mode') == 'geometric'
            for weight, model in zip(self.candidate['weights'], self.members_):
                probs = model.predict_proba(X)
                source = list(map(_class_key, model.classes_))
                target = list(map(_class_key, self.classes_))
                aligned = probs[:, [source.index(c) for c in target]]
                result += weight * (np.log(np.maximum(aligned, 1e-9)) if geometric else aligned)
            if geometric:
                result = np.exp(result - result.max(axis=1, keepdims=True))
        else:
            raw = np.asarray(self.model_.predict_proba(self._matrix(X)), dtype=float)
            result = np.zeros((len(X), len(self.classes_)), dtype=float)
            source = list(map(int, self.model_.classes_))
            for i, label in enumerate(source):
                result[:, label] = raw[:, i]
        sums = result.sum(axis=1, keepdims=True)
        return result / np.maximum(sums, 1e-15)

    def _decision(self, probabilities):
        power = float(self.candidate.get('decision_power', 0.0))
        if power:
            probabilities = probabilities * np.power(np.maximum(self.priors_, 1e-12), -power)[None, :]
        return np.argmax(probabilities, axis=1)


class GroupExpertModel:
    """Per-value experts with one global fallback for sparse or unseen groups.

    The mechanism is only used when a candidate explicitly names a categorical
    column. Experts are trained on fit rows only; a group is eligible when it
    contains enough rows. Classification probabilities are smoothed toward the
    global model so a class absent from one group does not receive zero mass.
    """
    def __init__(self, kind, candidate, threads=4):
        self.kind = kind
        self.candidate = canonical_candidate(candidate)
        if 'expert_column' not in self.candidate:
            raise ValueError('GroupExpertModel requires expert_column')
        self.threads = max(1, min(8, int(threads)))
        self.feature_code_ = ''
        self.audit_ = {}

    def _base_candidate(self):
        return {key: value for key, value in self.candidate.items()
                if key not in ('expert_column', 'expert_min_rows', 'max_expert_groups')}

    def _keys(self, X):
        return X[self.candidate['expert_column']].astype('string').fillna('__MISSING__')

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        column = self.candidate['expert_column']
        if column not in X:
            raise ValueError('Unknown expert_column: ' + column)
        y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
        self.global_ = CandidateModel(self.kind, self._base_candidate(), self.threads).fit(X, y_series)
        self.classes_ = self.global_.classes_
        self.priors_ = getattr(self.global_, 'priors_', None)
        keys = self._keys(X)
        counts = keys.value_counts()
        minimum = self.candidate['expert_min_rows']
        eligible = [key for key, count in counts.items() if count >= minimum]
        eligible.sort(key=lambda key: (-int(counts[key]), str(key)))
        limit = self.candidate['max_expert_groups']
        self.experts_, skipped = {}, []
        for key in eligible[:limit]:
            mask = (keys == key).to_numpy()
            local_y = y_series[mask]
            if self.kind != 'regression' and local_y.nunique() < 2:
                skipped.append(str(key))
                continue
            try:
                self.experts_[key] = CandidateModel(
                    self.kind, self._base_candidate(), self.threads).fit(X.loc[mask], local_y)
            except (ValueError, TypeError):
                skipped.append(str(key))
        if not self.experts_:
            # Keep the candidate usable: a global fallback is still a measured model.
            self.audit_['expert_fallback_only'] = True
        self.audit_.update({
            'expert_column': column,
            'expert_groups': {str(key): int(counts[key]) for key in sorted(self.experts_, key=str)},
            'expert_models': len(self.experts_),
            'expert_skipped_groups': skipped[:20],
            'expert_min_rows': minimum,
            'expert_fallback': 'global',
            'global_model': self.global_.audit_,
        })
        self.feature_code_ = self.global_.feature_code_
        return self

    def predict(self, X):
        if self.kind == 'regression':
            result = np.asarray(self.global_.predict(X), dtype=float).reshape(-1)
            keys = self._keys(X)
            for key, model in self.experts_.items():
                mask = (keys == key).to_numpy()
                if mask.any():
                    result[mask] = np.asarray(model.predict(X.loc[mask]), dtype=float).reshape(-1)
            return result
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        if self.kind == 'regression':
            raise AttributeError('Regression candidates do not expose predict_proba')
        result = np.asarray(self.global_.predict_proba(X), dtype=float).copy()
        keys = self._keys(X)
        target = {_class_key(value): index for index, value in enumerate(self.classes_)}
        for key, model in self.experts_.items():
            mask = (keys == key).to_numpy()
            if not mask.any():
                continue
            local = np.asarray(model.predict_proba(X.loc[mask]), dtype=float)
            aligned = np.zeros((int(mask.sum()), len(self.classes_)), dtype=float)
            for source, label in enumerate(model.classes_):
                if _class_key(label) in target:
                    aligned[:, target[_class_key(label)]] = local[:, source]
            result[mask] = .85 * aligned + .15 * result[mask]
        sums = result.sum(axis=1, keepdims=True)
        return result / np.maximum(sums, 1e-15)


class _NativeFrame:
    def __init__(self, frame, categorical):
        self.frame = frame
        self.categorical = categorical
        self.preserve_categorical = {c: isinstance(frame[c].dtype, pd.CategoricalDtype) for c in categorical}

    def fit_transform(self, X):
        self.columns = list(self.frame.columns)
        self.categories = {c: list(self.frame[c].dropna().astype(str).unique()) for c in self.categorical}
        return self.transform(X)

    def transform(self, X):
        out = X.copy().reindex(columns=self.columns)
        for col in self.categorical:
            values = out[col].astype('string').fillna('__MISSING__').astype(str)
            cats = self.categories[col]
            if self.preserve_categorical[col]:
                out[col] = pd.Categorical(values, categories=cats)
            else:
                out[col] = values
        for col in out.columns:
            if col not in self.categorical:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        return out


def _preprocessor(X, numeric, categorical, max_categories=None):
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=max_categories)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
    transformers = []
    if numeric:
        transformers.append(('numeric', Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler(with_mean=False))]), numeric))
    if categorical:
        transformers.append(('categorical', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='__MISSING__')), ('encode', encoder)]), categorical))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)


def build_pipeline(kind, candidate, threads=4):
    aliases = {'multiclass': 'classification', 'binary': 'classification', 'classification': 'classification', 'regression': 'regression'}
    if kind not in aliases:
        raise ValueError('kind must be classification or regression')
    normalized = aliases[kind]
    canonical = canonical_candidate(candidate)
    if canonical.get('panel_strategy') == 'hybrid':
        if normalized != 'regression':
            raise ValueError('Hybrid causal panel requires regression')
        from automl_residual import CausalHybridModel
        return CausalHybridModel(canonical, threads)
    if 'expert_column' in canonical:
        return GroupExpertModel(normalized, canonical, threads)
    return CandidateModel(normalized, canonical, threads)
