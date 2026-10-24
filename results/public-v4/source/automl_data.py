"""Task/data contract shared by the entry point and isolated workers."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split


LOWER_IS_BETTER = {'rmse', 'mae', 'mse', 'rmsle', 'mape', 'log_loss'}
CLASSIFICATION_METRICS = {'binary_f1', 'micro_f1', 'macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss'}
REGRESSION_METRICS = {'rmse', 'mae', 'mse', 'rmsle', 'mape', 'r2'}
_METRIC_PATTERNS = {
    'balanced_accuracy': r'\bbalanced[\s_-]*accuracy\b',
    'binary_f1': r'\b(?:binary[\s_-]*f1|f1[\s_-]*binary)\b',
    'micro_f1': r'\b(?:micro[\s_-]*f1|f1[\s_-]*micro|f1(?:[\s_-]*score)?)\b',
    'weighted_f1': r'\b(?:weighted[\s_-]*f1|f1[\s_-]*weighted)\b',
    'macro_f1': r'\b(?:macro[\s_-]*f1|f1[\s_-]*macro)\b',
    'roc_auc': r'\b(?:roc[\s_-]*auc|auc)\b',
    'log_loss': r'\b(?:log[\s_-]*loss|cross[\s_-]*entropy)\b',
    'rmse': r'\b(?:rmse|root[\s_-]+mean[\s_-]+squared[\s_-]+error)\b',
    'mae': r'\b(?:mae|mean[\s_-]+absolute[\s_-]+error)\b',
    'mse': r'\b(?:mse|mean[\s_-]+squared[\s_-]+error)\b',
    'rmsle': r'\b(?:rmsle|root[\s_-]+mean[\s_-]+squared[\s_-]+logarithmic[\s_-]+error)\b',
    'mape': r'\b(?:mape|mean[\s_-]+absolute[\s_-]+percentage[\s_-]+error)\b',
    'r2': r'\b(?:r2|r\^2|r²|r[\s_-]*squared|coefficient[\s_-]+of[\s_-]+determination)\b',
    'accuracy': r'\b(?:accuracy|точност[ьи])\b',
}
_DECLARATION = re.compile(r'\b(?:(?P<primary>primary|main|основн\w*)\s+)?(?:(?:evaluation|quality)\s+)?(?:metric|метрик\w*)\b', re.I)
_SECONDARY = re.compile(r'\b(?:secondary|auxiliary|example\w*|diagnostic\w*|дополнител\w*|вторичн\w*|справочн\w*|пример\w*)\b', re.I)


def _metric_mentions(text):
    matches = [(m.start(), m.end(), key) for key, pattern in _METRIC_PATTERNS.items()
               for m in re.finditer(pattern, text, re.I)]
    accepted = []
    for start, end, key in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if not any(start < old_end and end > old_start for old_start, old_end, _ in accepted):
            accepted.append((start, end, key))
    return {key for _, _, key in accepted}


def infer_metric(text, default=None):
    """Extract one declared primary metric; reject unknown or conflicting declarations."""
    lines = [re.sub(r'[*`]', '', line).strip() for line in text.splitlines()]
    declarations = []
    for i, line in enumerate(lines):
        for match in _DECLARATION.finditer(line):
            primary = bool(match.group('primary'))
            if _SECONDARY.search(line[:match.start()]) and not primary:
                continue
            fragment = line[match.end():].strip()
            heading = bool(re.match(r'^#{1,6}\s', line))
            if not (primary or heading or re.match(r'^(?:[:—–=\-]|is\b|equals\b|выбран\w*\b|является\b)', fragment, re.I)):
                continue
            fragment = _SECONDARY.split(fragment, maxsplit=1)[0].strip()
            if not fragment.strip(' :—–=-.'):
                fragment = next((following for following in lines[i + 1:] if following and not following.startswith('#')), '')
            value = re.sub(r'^[\s:—–=.,\-]*(?:(?:is|equals|the|will\s+be|выбран\w*|является)\b[\s:—–=.,\-]*)*', '', fragment, flags=re.I)
            known = any(re.match(pattern, value, re.I) for pattern in _METRIC_PATTERNS.values())
            declarations.append((2 if primary else 1, fragment, _metric_mentions(fragment) if known else set()))
    if declarations:
        rank = max(item[0] for item in declarations)
        chosen = [item for item in declarations if item[0] == rank]
        unknown = [fragment for _, fragment, names in chosen if not names]
        if unknown:
            raise ValueError('Unsupported explicit metric: ' + repr(unknown[0]))
        names = set().union(*(item[2] for item in chosen))
    else:
        names = _metric_mentions(text)
    if len(names) > 1:
        raise ValueError('Ambiguous metric: ' + ', '.join(sorted(names)))
    if names:
        return next(iter(names))
    if default not in CLASSIFICATION_METRICS | REGRESSION_METRICS:
        raise ValueError('Primary metric is missing or unsupported')
    return default


def _role_for_context(context):
    value = context.lower().replace('-', '_')
    if re.search(r'dictionary|data dictionary|словар|описани[ея]\s+признак', value):
        return 'dictionary'
    if re.search(r'sample[_ ]?submission|submission[_ ]?sample|образец\s+предсказ|шаблон\s+отправ', value):
        return 'sample_submission'
    if re.search(r'\btrain(?:ing)?\b|обучающ\w*', value):
        return 'train'
    if re.search(r'\btest\b|тестов\w*', value):
        return 'test'
    return None


def _csv_references(text):
    # Markdown links, inline code and ordinary relative/absolute paths are all accepted.
    pattern = r'''(?:\[[^\]]*\]\(\s*)?[<`"']?([^\s<>`"'()\]]+\.csv)[>`"']?'''
    refs = []
    for match in re.finditer(pattern, text, re.I):
        token = match.group(1).strip("'\"`<>.,;:)")
        line_start = text.rfind('\n', 0, match.start()) + 1
        line_end = text.find('\n', match.end())
        if line_end < 0:
            line_end = len(text)
        context = text[line_start:line_end]
        role = _role_for_context(context)
        if role is None:
            role = _role_for_context(text[max(0, match.start() - 100):match.end() + 100])
        refs.append((token.replace('\\', '/'), role))
    return refs


def resolve_paths(task_path, text):
    """Resolve role-linked files, then use a conservative structural fallback."""
    task_path = Path(task_path).resolve()
    base = task_path.parent
    keys = ('train', 'test', 'sample_submission', 'dictionary')
    files = sorted(p.resolve() for p in base.rglob('*.csv') if 'agent_runs' not in p.parts)
    by_role = {key: set() for key in keys}
    for token, role in _csv_references(text):
        if role is None:
            continue
        path = Path(token)
        path = path if path.is_absolute() else base / path
        if path.is_file():
            by_role[role].add(path.resolve())
    # Filename semantics are useful fallback hints, not public-dataset assumptions.
    for path in files:
        name = path.stem.lower().replace('-', '_')
        if re.search(r'(^|_)(sample_submission|submission_sample|sample)(_|$)', name):
            by_role['sample_submission'].add(path)
        elif re.search(r'(^|_)(dictionary|data_dictionary|schema|metadata)(_|$)', name):
            by_role['dictionary'].add(path)
        elif re.search(r'(^|_)(train|training)(_|$)', name):
            by_role['train'].add(path)
        elif re.search(r'(^|_)(test|testing)(_|$)', name):
            by_role['test'].add(path)
    # Ignore basename guesses when role-linked files already resolve uniquely.
    chosen = {}
    for key in keys:
        if key == 'dictionary':
            continue
        if len(by_role[key]) == 1:
            chosen[key] = next(iter(by_role[key]))
        elif len(by_role[key]) > 1:
            raise ValueError(f'Ambiguous {key} CSV references: {sorted(map(str, by_role[key]))}')
    if 'train' not in chosen or 'test' not in chosen or 'sample_submission' not in chosen:
        frames = {}
        for path in files:
            if path in by_role['dictionary']:
                continue
            try:
                frames[path] = pd.read_csv(path, nrows=8)
            except (OSError, UnicodeError, pd.errors.ParserError):
                continue
        if 'train' not in chosen:
            candidates = []
            for path, frame in frames.items():
                if path in chosen.values() or path in by_role['sample_submission']:
                    continue
                if any(len(other.columns) < len(frame.columns) and set(other.columns).issubset(frame.columns)
                       for other_path, other in frames.items() if other_path != path):
                    candidates.append(path)
            if len(candidates) == 1:
                chosen['train'] = candidates[0]
        if 'test' not in chosen and 'train' in chosen:
            train_cols = set(frames.get(chosen['train'], pd.read_csv(chosen['train'], nrows=8)).columns)
            candidates = [p for p, frame in frames.items() if p != chosen['train'] and set(frame.columns).issubset(train_cols)]
            if len(candidates) == 1:
                chosen['test'] = candidates[0]
        if 'sample_submission' not in chosen:
            reserved = set(chosen.values())
            candidates = [p for p, frame in frames.items() if p not in reserved and
                          (p in by_role['sample_submission'] or len(frame.columns) <= 2)]
            if len(candidates) == 1:
                chosen['sample_submission'] = candidates[0]
    missing = [k for k in ('train', 'test', 'sample_submission') if k not in chosen]
    if missing:
        raise ValueError('Could not resolve required CSV roles: ' + ', '.join(missing))
    dictionaries = by_role['dictionary']
    if not dictionaries:
        dictionaries = {p for p in files if re.search(r'(dictionary|schema|metadata)', p.stem, re.I)}
    if len(dictionaries) > 1:
        raise ValueError('Ambiguous data dictionary CSV')
    chosen['dictionary'] = next(iter(dictionaries)) if dictionaries else None
    return chosen


def _schema_roles(dictionary):
    if dictionary is None or dictionary.empty:
        return {}, {}
    name_col = next((c for c in dictionary.columns if str(c).lower() in {'feature', 'column', 'column_name', 'field', 'name', 'variable'}), None)
    role_col = next((c for c in dictionary.columns if str(c).lower() in {'role', 'type', 'usage', 'description'}), None)
    if name_col is None or role_col is None:
        return {}, {}
    roles = {}
    for _, row in dictionary.iterrows():
        feature, role = str(row[name_col]), str(row[role_col]).lower()
        if re.search(r'target|label|outcome|dependent|цель|таргет|класс', role):
            roles.setdefault('target', []).append(feature)
        if re.search(r'\bid\b|identifier|unique key|идентификатор', role):
            roles.setdefault('id', []).append(feature)
        if re.search(r'group|групп', role):
            roles.setdefault('group', []).append(feature)
    return roles, {str(row[name_col]): str(row[role_col]) for _, row in dictionary.iterrows()}


def _text_declared_column(text, role):
    patterns = {
        'target': r'\b(?:target|label|outcome|цель|таргет)\s*(?:column|field|переменн\w*)?\s*[:=]\s*[`"\']?([\w .-]+)',
        'id': r'\b(?:id|identifier|идентификатор)\s*(?:column|field|столбец)?\s*[:=]\s*[`"\']?([\w .-]+)',
        'group': r'\b(?:group|групп\w*)\s*(?:column|field|столбец)?\s*[:=]\s*[`"\']?([\w .-]+)',
        "positive_class": r"(?:positive\s+class|positive\s+label|положительн\w*\s+класс)\s*[:=]\s*[`\"']?([^\n`\"']+)",
    }
    match = re.search(patterns[role], text, re.I)
    return match.group(1).strip().rstrip(' .;,') if match else None


@dataclass
class TaskData:
    task_path: Path
    text: str
    paths: dict
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    dictionary: pd.DataFrame
    target: str
    id_column: str | None
    task_type: str
    metric: str
    classes: tuple | None = None
    positive_class: object | None = None
    group_column: str | None = None

    @property
    def X(self):
        excluded = [self.target] + ([self.id_column] if self.id_column else [])
        return self.train.drop(columns=excluded)

    @property
    def y(self):
        return self.train[self.target]

    @property
    def X_test(self):
        """Predictors aligned to sample_submission row order when an ID is available."""
        frame = self.test
        if self.id_column and self.id_column in self.sample:
            lookup = pd.Series(np.arange(len(frame)), index=frame[self.id_column].to_numpy())
            try:
                positions = lookup.loc[self.sample[self.id_column].to_numpy()].to_numpy(dtype=int)
                frame = frame.iloc[positions]
            except (KeyError, TypeError):
                raise ValueError('Sample IDs do not match test IDs')
        return frame.drop(columns=[self.id_column], errors='ignore')[self.X.columns]

    @property
    def hashes(self):
        return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in self.paths.items() if p is not None}


def _ordered_classes(values):
    unique = list(pd.unique(values))
    try:
        return tuple(sorted(unique))
    except TypeError:
        return tuple(sorted(unique, key=lambda value: (type(value).__name__, str(value))))


def load_task(path):
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8-sig')
    paths = resolve_paths(path, text)
    train = pd.read_csv(paths['train'])
    test = pd.read_csv(paths['test'])
    sample = pd.read_csv(paths['sample_submission'])
    dictionary = pd.read_csv(paths['dictionary']) if paths['dictionary'] else pd.DataFrame()
    roles, _ = _schema_roles(dictionary)
    target_declared = _text_declared_column(text, 'target')
    target_options = [c for c in roles.get('target', []) if c in train.columns and c not in test.columns]
    if target_declared:
        if target_declared not in train or target_declared in test:
            raise ValueError('Declared target column is not a training-only column: ' + target_declared)
        target = target_declared
    elif len(target_options) == 1:
        target = target_options[0]
    else:
        extra = [c for c in train.columns if c not in test.columns]
        if len(extra) != 1:
            raise ValueError('Target is ambiguous; declare it in task.md or data dictionary')
        target = extra[0]
    if train[target].isna().any():
        raise ValueError('Training target contains missing values')
    explicit_id = _text_declared_column(text, 'id')
    id_options = [c for c in roles.get('id', []) if c in train.columns and c in test.columns]
    if explicit_id:
        if explicit_id not in train or explicit_id not in test:
            raise ValueError('Declared ID column is absent from train/test: ' + explicit_id)
        identifier = explicit_id
    elif len(id_options) == 1:
        identifier = id_options[0]
    else:
        common = [c for c in sample.columns if c in test.columns and c in train.columns and c != target]
        matching = [c for c in common if sample[c].is_unique and test[c].is_unique and
                    len(sample[c]) == len(test[c]) and set(sample[c]) == set(test[c])]
        if len(id_options) > 1:
            raise ValueError('Ambiguous identifier in data dictionary')
        if len(matching) > 1:
            raise ValueError('Ambiguous identifier matching sample and test')
        identifier = matching[0] if matching else None
    if identifier:
        if identifier not in sample:
            raise ValueError('Sample submission is missing the declared ID column')
        if not train[identifier].is_unique or not test[identifier].is_unique or not sample[identifier].is_unique:
            raise ValueError('ID values must be unique in train, test, and sample')
        if len(sample) != len(test) or set(sample[identifier]) != set(test[identifier]):
            raise ValueError('Sample IDs do not match test IDs')
    elif len(sample) != len(test):
        raise ValueError('Sample/test row count mismatch')
    excluded = {target, identifier}
    train_features = [c for c in train if c not in excluded]
    test_features = [c for c in test if c not in excluded]
    if set(train_features) != set(test_features):
        raise ValueError('Train/test predictor schemas differ')
    if identifier:
        test = test.set_index(identifier, drop=False).loc[sample[identifier].to_list()].reset_index(drop=True)
    elif len(sample) != len(test):
        raise ValueError('Sample/test row count mismatch')
    metric_text = text
    metric_default = None
    metric = infer_metric(metric_text, default=metric_default)
    task_type = 'classification' if metric in CLASSIFICATION_METRICS else 'regression'
    if task_type == 'classification' and pd.api.types.is_numeric_dtype(train[target]) and re.search(r'regression|регресс', text, re.I):
        raise ValueError('Task description and primary metric disagree about task type')
    classes = _ordered_classes(train[target]) if task_type == 'classification' else None
    if metric in ('roc_auc', 'log_loss') and len(classes) > 2:
        probability_columns = [c for c in sample if c != identifier]
        if set(map(str, probability_columns)) != set(map(str, classes)):
            raise ValueError('Multiclass probability output must have one column named for each class')
    positive_text = _text_declared_column(text, 'positive_class')
    positive_class = None
    if positive_text is not None:
        matches = [value for value in classes or () if str(value) == positive_text]
        if len(matches) != 1:
            raise ValueError('Declared positive class does not match a target label')
        positive_class = matches[0]
    group_declared = _text_declared_column(text, 'group')
    group_options = [c for c in roles.get('group', []) if c in train_features]
    group_column = group_declared if group_declared in train_features else (group_options[0] if len(group_options) == 1 else None)
    return TaskData(path, text, paths, train, test, sample, dictionary, target, identifier,
                    task_type, metric, classes, positive_class, group_column)


def _time_values(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors='coerce')
        return values if values.notna().all() else None
    parsed = pd.to_datetime(series, errors='coerce', utc=True)
    if parsed.notna().all():
        return parsed.astype('int64')
    return None


def _temporal_split(bundle):
    candidates = []
    for column in bundle.X.columns:
        if not re.search(r'year|period|date|time|год|период|дата|врем', str(column), re.I):
            continue
        values = _time_values(bundle.X[column])
        test_values = _time_values(bundle.X_test[column])
        if values is None or test_values is None:
            continue
        unique = np.unique(values.to_numpy())
        numeric_period = pd.api.types.is_numeric_dtype(bundle.X[column]) and 3 <= len(unique) <= 100
        parsed_date = not pd.api.types.is_numeric_dtype(bundle.X[column]) and len(unique) >= 3
        if (numeric_period or parsed_date) and test_values.min() > values.max():
            candidates.append((column, values, unique))
    if not candidates:
        return None
    column, values, unique = candidates[0]
    last = unique[-1]
    fit = np.flatnonzero(values.to_numpy() < last)
    valid = np.flatnonzero(values.to_numpy() == last)
    if not len(fit) or not len(valid):
        return None
    description = f'temporal:{column}:train_last={last}:test_after_train'
    return fit, valid, description


def _stable_groups(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)


def _move_singletons_to_fit(fit, valid, y, groups):
    counts = pd.Series(y).value_counts(dropna=False)
    singleton = set(counts[counts < 2].index.tolist())
    if not singleton:
        return fit, valid, []
    y_values = np.asarray(y)
    valid_singletons = [index for index in valid if y_values[index] in singleton]
    if not valid_singletons:
        return fit, valid, sorted(map(str, singleton))
    singleton_groups = set(groups[valid_singletons])
    mask = np.array([groups[index] in singleton_groups for index in valid], dtype=bool)
    moved = valid[mask]
    return np.concatenate([fit, moved]), valid[~mask], sorted(map(str, singleton))


def validation_split(bundle):
    X, y = bundle.X, bundle.y
    positions = np.arange(len(y))
    group_col = getattr(bundle, 'group_column', None)
    if group_col and group_col in X:
        groups = X[group_col].astype('string').fillna('__MISSING_GROUP__').to_numpy()
        group_desc = f'group:{group_col}'
    else:
        groups = _stable_groups(X)
        group_desc = 'exact-feature-groups'
    has_duplicates = len(np.unique(groups)) < len(groups)
    temporal = _temporal_split(bundle)
    if temporal is not None:
        temporal_fit, temporal_valid, temporal_desc = temporal
        if set(groups[temporal_fit]).isdisjoint(set(groups[temporal_valid])):
            if bundle.task_type == 'regression':
                return temporal_fit, temporal_valid, temporal_desc
            temporal_fit, temporal_valid, singleton_classes = _move_singletons_to_fit(
                temporal_fit, temporal_valid, y, groups)
            if len(temporal_valid):
                if singleton_classes:
                    temporal_desc += f':singleton-classes-kept-in-fit={len(singleton_classes)}'
                return np.sort(temporal_fit), np.sort(temporal_valid), temporal_desc
    if bundle.task_type == 'regression':
        if has_duplicates or group_col:
            fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
            return fit, valid, f'group-shuffle80/20:seed42:{group_desc}'
        fit, valid = train_test_split(positions, test_size=.2, random_state=42)
        return fit, valid, 'random80/20:seed42'
    if has_duplicates or group_col:
        try:
            fit, valid = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))
            description = f'stratified-group5:first:seed42:{group_desc}'
        except ValueError:
            fit, valid = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
            description = f'group-shuffle80/20:seed42:{group_desc}:stratification-impossible'
    else:
        try:
            fit, valid = train_test_split(positions, test_size=.2, stratify=y, random_state=42)
            description = 'stratified80/20:seed42'
        except ValueError:
            fit, valid = train_test_split(positions, test_size=.2, random_state=42)
            description = 'random80/20:seed42:stratification-impossible'
    fit, valid, singleton_classes = _move_singletons_to_fit(fit, valid, y, groups)
    description += f':singleton-classes-kept-in-fit={len(singleton_classes)}' if singleton_classes else ''
    if not len(valid):
        # Use only groups that contain no singleton label, preserving the train-only constraint.
        counts = pd.Series(y).value_counts(dropna=False)
        singleton_values = counts[counts < 2].index.to_numpy()
        singleton_groups = set(groups[np.isin(np.asarray(y), singleton_values)])
        eligible_groups = np.asarray([group for group in np.unique(groups) if group not in singleton_groups])
        if len(eligible_groups) < 2:
            return positions, np.array([], dtype=int), description + ':no-validation-safe-rows;all-classes-retained-in-fit'
        group_split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        eligible_positions = np.flatnonzero(~np.isin(groups, list(singleton_groups)))
        fallback_fit, fallback_valid = next(group_split.split(eligible_positions, groups=groups[eligible_positions]))
        valid = eligible_positions[fallback_valid]
        fit = np.setdiff1d(positions, valid, assume_unique=False)
        description += ':singleton-fallback-grouped'
    if has_duplicates and not set(groups[fit]).isdisjoint(set(groups[valid])):
        raise ValueError('Validation split would separate duplicate feature rows')
    return np.sort(fit), np.sort(valid), description


def split_key(fit, valid):
    return hashlib.sha256(np.asarray(fit, dtype='<i8').tobytes() + b'|' + np.asarray(valid, dtype='<i8').tobytes()).hexdigest()


def _class_order(y, classes):
    labels = tuple(classes) if classes is not None else _ordered_classes(y)
    if set(pd.unique(np.asarray(y))) - set(labels):
        raise ValueError('classes does not contain every target label')
    return labels


def score(metric, y, prediction, classes=None, positive_class=None):
    y_array = np.asarray(y)
    pred = np.asarray(prediction)
    if metric == 'rmse': return float(np.sqrt(metrics.mean_squared_error(y_array, pred)))
    if metric == 'mse': return float(metrics.mean_squared_error(y_array, pred))
    if metric == 'mae': return float(metrics.mean_absolute_error(y_array, pred))
    if metric == 'mape': return float(metrics.mean_absolute_percentage_error(y_array, pred))
    if metric == 'rmsle':
        if np.any(y_array < 0): raise ValueError('RMSLE requires nonnegative actual targets')
        return float(np.sqrt(metrics.mean_squared_log_error(y_array, np.maximum(pred, 0))))
    if metric == 'r2': return float(metrics.r2_score(y_array, pred))
    if metric == 'accuracy': return float(metrics.accuracy_score(y_array, pred))
    if metric == 'balanced_accuracy': return float(metrics.balanced_accuracy_score(y_array, pred))
    if metric in ('binary_f1', 'micro_f1', 'macro_f1', 'weighted_f1'):
        labels = _class_order(y_array, classes)
        if metric == 'binary_f1':
            if len(labels) != 2: raise ValueError('binary_f1 requires exactly two classes')
            positive = positive_class if positive_class is not None else labels[-1]
            if positive not in labels: raise ValueError('positive_class is absent from classes')
            return float(metrics.f1_score(y_array == positive, pred == positive, zero_division=0))
        average = {'micro_f1': 'micro', 'macro_f1': 'macro', 'weighted_f1': 'weighted'}[metric]
        return float(metrics.f1_score(y_array, pred, labels=labels, average=average, zero_division=0))
    if metric in ('roc_auc', 'log_loss'):
        labels = _class_order(y_array, classes)
        if pred.ndim == 1:
            if len(labels) != 2:
                raise ValueError(f'{metric} needs a probability column for every class')
            positive = positive_class if positive_class is not None else labels[-1]
            if positive not in labels:
                raise ValueError('positive_class is absent from classes')
            binary_y = (y_array == positive).astype(int)
            if metric == 'roc_auc': return float(metrics.roc_auc_score(binary_y, pred))
            positive_index = labels.index(positive)
            probabilities = np.empty((len(pred), 2), dtype=float)
            probabilities[:, positive_index] = pred
            probabilities[:, 1 - positive_index] = 1 - pred
            pred = probabilities
        if pred.ndim != 2 or pred.shape[1] != len(labels):
            raise ValueError(f'{metric} probability columns do not match classes')
        if metric == 'roc_auc':
            if len(labels) == 2:
                positive = positive_class if positive_class is not None else labels[-1]
                if positive not in labels: raise ValueError('positive_class is absent from classes')
                binary_y = (y_array == positive).astype(int)
                return float(metrics.roc_auc_score(binary_y, pred[:, labels.index(positive)]))
            return float(np.mean([metrics.roc_auc_score(y_array == label, pred[:, index])
                                  for index, label in enumerate(labels)]))
        if not np.isfinite(pred).all() or np.any(pred < 0) or np.any(pred > 1):
            raise ValueError('Probabilities must be finite and in [0, 1]')
        row_sums = pred.sum(axis=1)
        if np.any(row_sums <= 0): raise ValueError('Probability rows must have positive mass')
        probabilities = pred / row_sums[:, None]
        label_index = {label: index for index, label in enumerate(labels)}
        target_indices = np.asarray([label_index[value] for value in y_array], dtype=int)
        chosen = probabilities[np.arange(len(y_array)), target_indices]
        return float(-np.log(np.clip(chosen, np.finfo(float).eps, 1 - np.finfo(float).eps)).mean())
    raise ValueError('Unsupported metric: ' + str(metric))


def better(metric, a, b):
    if metric not in LOWER_IS_BETTER | {'r2', 'accuracy', 'balanced_accuracy', 'roc_auc', 'binary_f1', 'micro_f1', 'macro_f1', 'weighted_f1'}:
        raise ValueError('Unsupported metric: ' + str(metric))
    if b is None: return np.isfinite(a)
    return np.isfinite(a) and (a < b if metric in LOWER_IS_BETTER else a > b)


def _output_columns(bundle):
    return [column for column in bundle.sample.columns if column != bundle.id_column]


def make_submission(bundle, prediction):
    """Predictions must already follow X_test/sample order."""
    frame = bundle.sample.copy()
    outputs = _output_columns(bundle)
    pred = np.asarray(prediction)
    if len(outputs) == 1:
        if pred.ndim == 2 and pred.shape[1] > 1 and bundle.metric in ('roc_auc', 'log_loss'):
            classes = tuple(bundle.classes or _ordered_classes(bundle.y))
            positive = bundle.positive_class if bundle.positive_class is not None else classes[-1]
            if pred.shape[1] != len(classes): raise ValueError('Probability columns do not match classes')
            pred = pred[:, classes.index(positive)]
        elif pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if pred.ndim != 1 or len(pred) != len(frame): raise ValueError('Prediction shape does not match sample rows')
        frame[outputs[0]] = pred
    elif pred.ndim == 2 and pred.shape == (len(frame), len(outputs)):
        classes = getattr(bundle, 'classes', None)
        if classes is not None and set(map(str, outputs)) == set(map(str, classes)):
            index = {str(label): i for i, label in enumerate(classes)}
            pred = pred[:, [index[str(column)] for column in outputs]]
        frame.loc[:, outputs] = pred
    else:
        raise ValueError('Prediction shape does not match sample columns')
    validate_submission(bundle, frame)
    return frame


def validate_submission(bundle, frame):
    if list(frame) != list(bundle.sample) or len(frame) != len(bundle.test) or not len(frame):
        raise ValueError('Invalid submission shape')
    if frame.isna().any().any(): raise ValueError('Missing submission value')
    if bundle.id_column:
        c = bundle.id_column
        if c not in frame or not frame[c].is_unique or not frame[c].reset_index(drop=True).equals(bundle.sample[c].reset_index(drop=True)):
            raise ValueError('Invalid ID values or order')
    outputs = _output_columns(bundle)
    if bundle.task_type == 'regression' or bundle.metric in ('roc_auc', 'log_loss') or len(outputs) > 1:
        try:
            values = frame[outputs].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Predictions must be numeric') from exc
        if not np.isfinite(values).all(): raise ValueError('Nonfinite predictions')
        if bundle.metric in ('roc_auc', 'log_loss'):
            if np.any(values < 0) or np.any(values > 1): raise ValueError('Probabilities must be in [0, 1]')
            if len(outputs) > 1 and not np.allclose(values.sum(axis=1), 1, rtol=1e-5, atol=1e-6):
                raise ValueError('Multiclass probability rows must sum to one')
    elif len(outputs) != 1 or not set(frame[outputs[0]]).issubset(set(bundle.y)):
        raise ValueError('Unknown class label')


def fallback_submission(bundle):
    if bundle.task_type == 'regression':
        prediction = np.full(len(bundle.sample), float(bundle.y.mean()))
    elif bundle.metric in ('roc_auc', 'log_loss'):
        classes = tuple(bundle.classes or _ordered_classes(bundle.y))
        priors = bundle.y.value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy(dtype=float)
        outputs = _output_columns(bundle)
        if len(outputs) > 1:
            if set(map(str, outputs)) != set(map(str, classes)):
                raise ValueError('Probability output columns must match class labels')
        if len(outputs) == 1:
            positive = bundle.positive_class if bundle.positive_class is not None else classes[-1]
            prediction = np.full(len(bundle.sample), priors[classes.index(positive)])
        else:
            prediction = np.tile(priors, (len(bundle.sample), 1))
    else:
        prediction = np.repeat(bundle.y.mode().iloc[0], len(bundle.sample))
    return make_submission(bundle, prediction)


def atomic_submission(bundle, frame, destination=None):
    validate_submission(bundle, frame)
    target = Path(destination) if destination else bundle.task_path.parent / 'submission.csv'
    temporary = target.with_name(target.name + '.tmp')
    frame.to_csv(temporary, index=False)
    validate_submission(bundle, pd.read_csv(temporary))
    os.replace(temporary, target)
