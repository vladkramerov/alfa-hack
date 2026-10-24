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


LOWER_IS_BETTER = {'rmse', 'mae', 'mse', 'rmsle', 'mape', 'smape', 'medae', 'log_loss'}
CLASSIFICATION_METRICS = {'binary_f1', 'micro_f1', 'macro_f1', 'weighted_f1', 'accuracy',
                          'balanced_accuracy', 'roc_auc', 'log_loss', 'cohen_kappa', 'mcc'}
REGRESSION_METRICS = {'rmse', 'mae', 'mse', 'rmsle', 'mape', 'smape', 'medae', 'r2', 'explained_variance'}
_METRIC_PATTERNS = {
    'balanced_accuracy': r'\bbalanced[\s_-]*accuracy\b',
    'binary_f1': r'\b(?:binary[\s_-]*f1|f1[\s_-]*binary)\b',
    'micro_f1': r'\b(?:micro[\s_-]*f1|f1[\s_-]*micro)\b',
    'weighted_f1': r'\b(?:weighted[\s_-]*f1|f1[\s_-]*weighted)\b',
    'macro_f1': r'\b(?:macro[\s_-]*f1|f1[\s_-]*macro|macro[\s_-]*f[\s_-]*measure)\b',
    'roc_auc': r'\b(?:roc[\s_-]*auc|auc[\s_-]*roc|area[\s_-]+under[\s_-]+(?:the[\s_-]+)?roc|auc)\b',
    'log_loss': r'\b(?:log[\s_-]*loss|cross[\s_-]*entropy|logarithmic[\s_-]+loss)\b',
    'rmse': r'\b(?:rmse|root[\s_-]+mean[\s_-]+squared[\s_-]+error|среднеквадратичн\w*\s+ошибк\w*)\b',
    'mae': r'\b(?:mae|mean[\s_-]+absolute[\s_-]+error|средн\w*\s+абсолютн\w*\s+ошибк\w*)\b',
    'mse': r'\b(?:mse|mean[\s_-]+squared[\s_-]+error)\b',
    'rmsle': r'\b(?:rmsle|root[\s_-]+mean[\s_-]+squared[\s_-]+logarithmic[\s_-]+error)\b',
    'mape': r'\b(?:mape|mean[\s_-]+absolute[\s_-]+percentage[\s_-]+error)\b',
    'smape': r'\b(?:smape|symmetric[\s_-]+mean[\s_-]+absolute[\s_-]+percentage[\s_-]+error)\b',
    'medae': r'\b(?:medae|median[\s_-]+absolute[\s_-]+error)\b',
    'r2': r'\b(?:r2|r\^2|r²|r[\s_-]*squared|coefficient[\s_-]+of[\s_-]+determination|коэффициент[\s_-]+детерминации)\b',
    'accuracy': r'\b(?:accuracy|точност[ьи])\b',
    'cohen_kappa': r'\b(?:cohen[\s_-]*kappa|kappa[\s_-]*score)\b',
    'mcc': r'\b(?:mcc|matthews[\s_-]+correlation)\b',
    'explained_variance': r'\bexplained[\s_-]+variance\b',
    'f1': r'\b(?:f1|f[\s_-]*score|f[\s_-]*measure)\b',
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


def _canonical_metric(name):
    # A bare "F1" means scikit-learn's default binary/micro average; explicit
    # macro/weighted/binary declarations always win during span de-duplication.
    return 'micro_f1' if name == 'f1' else name


def infer_metric(text, default=None, strict=True):
    """Extract one declared primary metric.

    ``strict=False`` is used only as a last-resort contract fallback: an
    unsupported or conflicting metric never prevents a valid submission from
    being produced. The chosen fallback is recorded in ``TaskData.warnings``.
    """
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
        if unknown and strict:
            raise ValueError('Unsupported explicit metric: ' + repr(unknown[0]))
        names = set().union(*(item[2] for item in chosen))
    else:
        names = _metric_mentions(text)
    names = {_canonical_metric(name) for name in names}
    if len(names) > 1:
        if strict:
            raise ValueError('Ambiguous metric: ' + ', '.join(sorted(names)))
        names = set()
    if names:
        return next(iter(names))
    if default not in CLASSIFICATION_METRICS | REGRESSION_METRICS:
        if strict:
            raise ValueError('Primary metric is missing or unsupported')
        return None
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
    features: tuple | None = None
    warnings: tuple = ()
    metric_source: str = 'declared'

    @property
    def X(self):
        if self.features is not None:
            return self.train.loc[:, list(self.features)]
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
        columns = list(self.features) if self.features is not None else list(self.X.columns)
        frame = frame.drop(columns=[c for c in (self.id_column, self.target) if c], errors='ignore')
        return frame.reindex(columns=columns)

    @property
    def hashes(self):
        return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in self.paths.items() if p is not None}


def _ordered_classes(values):
    unique = list(pd.unique(values))
    try:
        return tuple(sorted(unique))
    except TypeError:
        return tuple(sorted(unique, key=lambda value: (type(value).__name__, str(value))))


def _target_looks_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return True
    text = series.dropna().astype(str).str.strip()
    if not len(text):
        return False
    converted = pd.to_numeric(text.str.replace(',', '.', regex=False), errors='coerce')
    return bool(converted.notna().mean() > .98)


def _default_metric(task_type, classes):
    if task_type == 'regression':
        return 'rmse'
    if classes is not None and len(classes) == 2:
        return 'binary_f1'
    return 'macro_f1'


def _choose_target(train, test, text, roles):
    declared = _text_declared_column(text, 'target')
    options = [c for c in roles.get('target', []) if c in train.columns]
    if declared:
        if declared not in train.columns:
            raise ValueError('Declared target column is absent from train: ' + declared)
        if declared in test.columns and not test[declared].isna().all():
            raise ValueError('Declared target column contains values in test: ' + declared)
        return declared
    if len(options) == 1:
        return options[0]
    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) == 1:
        return train_only[0]
    # Target-like names are a conservative last structural hint, never a
    # requirement tied to the public datasets.
    named = [c for c in train_only if re.search(r'(^|_)(target|label|outcome|y|response|class|цель)(_|$)', str(c), re.I)]
    if len(named) == 1:
        return named[0]
    raise ValueError('Target is ambiguous; declare it in task.md or data dictionary')


def _choose_identifier(train, test, sample, text, roles, target):
    declared = _text_declared_column(text, 'id')
    options = [c for c in roles.get('id', []) if c in train.columns and c in test.columns]
    if declared:
        if declared not in train.columns or declared not in test.columns:
            raise ValueError('Declared ID column is absent from train/test: ' + declared)
        return declared
    if len(options) == 1:
        return options[0]
    if len(options) > 1:
        raise ValueError('Ambiguous identifier in data dictionary')
    common = [c for c in sample.columns if c in test.columns and c in train.columns and c != target]
    matching = [c for c in common if sample[c].is_unique and test[c].is_unique and
                len(sample[c]) == len(test[c]) and set(sample[c]) == set(test[c])]
    if len(matching) > 1:
        raise ValueError('Ambiguous identifier matching sample and test')
    return matching[0] if matching else None


def load_task(path):
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8-sig')
    paths = resolve_paths(path, text)
    train = pd.read_csv(paths['train'])
    test = pd.read_csv(paths['test'])
    sample = pd.read_csv(paths['sample_submission'])
    dictionary = pd.read_csv(paths['dictionary']) if paths['dictionary'] else pd.DataFrame()
    warnings = []
    roles, _ = _schema_roles(dictionary)
    target = _choose_target(train, test, text, roles)
    if train[target].isna().any():
        # A handful of missing targets cannot be learned; keep rows only if a
        # small minority is affected, otherwise the contract is invalid.
        missing = int(train[target].isna().sum())
        if missing > max(1, int(len(train) * .02)):
            raise ValueError('Training target contains too many missing values')
        train = train.loc[train[target].notna()].reset_index(drop=True)
        warnings.append(f'dropped {missing} rows with missing target')
    identifier = _choose_identifier(train, test, sample, text, roles, target)
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
    train_features = [c for c in train.columns if c not in excluded]
    if identifier:
        # Align future rows to the sample order before the identifier is removed
        # from the predictor frame.
        test = test.set_index(identifier, drop=False).loc[sample[identifier].to_list()].reset_index(drop=True)
    test = test.drop(columns=[target] if target in test.columns else [], errors='ignore')
    test_features = [c for c in test.columns if c not in excluded]
    features = [c for c in train_features if c in test_features]
    extra_train = [c for c in train_features if c not in test_features]
    extra_test = [c for c in test_features if c not in train_features]
    if extra_train:
        warnings.append('train-only predictors ignored: ' + ', '.join(map(str, extra_train[:10])))
    if extra_test:
        warnings.append('test-only predictors ignored: ' + ', '.join(map(str, extra_test[:10])))
    if not features:
        raise ValueError('No shared predictors between train and test')
    metric = None
    metric_source = 'declared'
    try:
        metric = infer_metric(text, strict=True)
    except ValueError as exc:
        if 'Unsupported explicit metric' in str(exc) or 'Ambiguous metric' in str(exc):
            # A declared but unsupported or conflicting primary metric is a
            # contract error, never a silent substitution.
            raise
        warnings.append(str(exc)[:300])
        metric = infer_metric(text, strict=False)
        metric_source = 'inferred'
    numeric_target = _target_looks_numeric(train[target])
    if metric is None:
        # Infer task type from the target and dataset structure, never from a
        # public target name.
        unique = int(train[target].nunique(dropna=True))
        textual = re.search(r'classification|классификац|classify', text, re.I)
        if textual:
            task_type = 'classification'
        elif re.search(r'regression|регресс', text, re.I):
            task_type = 'regression'
        elif numeric_target and (unique > max(20, int(len(train) * .01)) or not np.allclose(
                pd.to_numeric(train[target], errors='coerce').dropna() % 1, 0)):
            task_type = 'regression'
        else:
            task_type = 'classification'
        classes_preview = _ordered_classes(train[target]) if task_type == 'classification' else None
        metric = _default_metric(task_type, classes_preview)
        metric_source = 'inferred_default'
        warnings.append(f'metric inferred as {metric}')
    task_type = 'classification' if metric in CLASSIFICATION_METRICS else 'regression'
    if task_type == 'regression':
        if not numeric_target:
            # A declared regression metric on a categorical target is a contract
            # conflict; classification is the only safe executable fallback.
            warnings.append('regression metric on non-numeric target; using classification')
            numeric_target = False
        if not numeric_target:
            task_type = 'classification'
            metric = _default_metric(task_type, _ordered_classes(train[target]))
            metric_source = 'inferred_default'
    if task_type == 'regression' and not pd.api.types.is_numeric_dtype(train[target]):
        converted = pd.to_numeric(train[target].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        if converted.isna().any():
            raise ValueError('Regression target cannot be parsed as numeric')
        train[target] = converted.astype(float)
        warnings.append('converted regression target to numeric')
    classes = _ordered_classes(train[target]) if task_type == 'classification' else None
    if task_type == 'classification' and classes is not None and len(classes) < 2:
        raise ValueError('Classification target has fewer than two classes')
    if metric in ('roc_auc', 'log_loss') and len(classes) > 2:
        probability_columns = [c for c in sample if c != identifier]
        if set(map(str, probability_columns)) != set(map(str, classes)):
            raise ValueError('Multiclass probability output must have one column named for each class')
    positive_text = _text_declared_column(text, 'positive_class')
    positive_class = None
    if positive_text is not None:
        matches = [value for value in classes or () if str(value) == positive_text]
        if len(matches) != 1:
            warnings.append('declared positive class not found; using default')
        else:
            positive_class = matches[0]
    group_declared = _text_declared_column(text, 'group')
    group_options = [c for c in roles.get('group', []) if c in features]
    group_column = group_declared if group_declared in features else (group_options[0] if len(group_options) == 1 else None)
    return TaskData(path, text, paths, train, test, sample, dictionary, target, identifier,
                    task_type, metric, classes, positive_class, group_column,
                    tuple(features), tuple(warnings), metric_source)


def _time_values(series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors='coerce')
        return values if values.notna().all() else None
    parsed = pd.to_datetime(series, errors='coerce', utc=True)
    if parsed.notna().all():
        return parsed.astype('int64')
    return None


def _temporal_candidate(bundle):
    candidates = []
    for column in bundle.X.columns:
        if not re.search(r'year|period|date|time|год|период|дата|врем', str(column), re.I):
            continue
        values = _time_values(bundle.X[column])
        test_values = _time_values(bundle.X_test[column])
        if values is None or test_values is None:
            continue
        unique = np.unique(values.to_numpy())
        numeric_period = pd.api.types.is_numeric_dtype(bundle.X[column]) and 3 <= len(unique) <= 200
        parsed_date = not pd.api.types.is_numeric_dtype(bundle.X[column]) and len(unique) >= 3
        if not (numeric_period or parsed_date) or test_values.min() <= values.max():
            continue
        # Prefer a compact, ordered period column with enough rows per period.
        counts = pd.Series(values.to_numpy()).value_counts()
        small = int((counts < max(20, int(len(values) * .01))).sum())
        score = (len(unique), small, 0 if re.search(r'year|год', str(column), re.I) else 1)
        candidates.append((score, column, values, unique))
    if not candidates:
        return None
    _, column, values, unique = sorted(candidates, key=lambda item: item[0])[0]
    return column, values, unique


def _temporal_split(bundle):
    candidate = _temporal_candidate(bundle)
    if candidate is None:
        return None
    column, values, unique = candidate
    last = unique[-1]
    fit = np.flatnonzero(values.to_numpy() < last)
    valid = np.flatnonzero(values.to_numpy() == last)
    if not len(fit) or not len(valid):
        return None
    description = f'temporal:{column}:train_last={last}:test_after_train'
    return fit, valid, description


def _stable_groups(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)


def _groups_for(bundle):
    X = bundle.X
    group_col = getattr(bundle, 'group_column', None)
    if group_col and group_col in X:
        return X[group_col].astype('string').fillna('__MISSING_GROUP__').to_numpy(), f'group:{group_col}'
    return _stable_groups(X), 'exact-feature-groups'


def _disjoint_temporal_fit(fit, valid, groups):
    """Drop fit rows whose exact/group identity also occurs in the future window."""
    overlap = set(groups[valid])
    keep = np.asarray([groups[index] not in overlap for index in fit], dtype=bool)
    return fit[keep]


def _ensure_classes_in_fit(fit, valid, y, classes):
    y = np.asarray(y)
    fit_labels = set(pd.unique(y[fit])) if len(fit) else set()
    missing = [label for label in classes or () if label not in fit_labels]
    if not missing:
        return fit, valid, []
    move = np.asarray([value in set(missing) for value in y[valid]], dtype=bool)
    return np.concatenate([fit, valid[move]]), valid[~move], missing


def temporal_windows(bundle):
    """Rolling-origin windows, newest first.

    The newest window is the closest available proxy for a future test period.
    For a task whose test period is beyond all training periods this is a more
    honest selection signal than a random split, and using the newest window
    avoids selecting a model on a stale year. The previous window is retained
    as a stability check, not as a second opportunity to tune on the audit.
    """
    candidate = _temporal_candidate(bundle)
    if candidate is None:
        return []
    column, values, unique = candidate
    groups, _ = _groups_for(bundle)
    labels = np.asarray(bundle.y)
    windows = []
    for period in list(unique[-2:])[::-1]:
        fit = np.flatnonzero(values.to_numpy() < period)
        valid = np.flatnonzero(values.to_numpy() == period)
        if not len(fit) or not len(valid):
            continue
        fit = _disjoint_temporal_fit(fit, valid, groups)
        if not len(fit):
            continue
        description = f'temporal:{column}:train_last={period}:test_after_train'
        if bundle.task_type == 'classification':
            fit, valid, missing = _ensure_classes_in_fit(fit, valid, labels, bundle.classes)
            if missing:
                description += ':classes-only-in-fit=' + ','.join(map(str, missing[:8]))
            fit, valid, singleton_classes = _move_singletons_to_fit(fit, valid, bundle.y, groups)
            if singleton_classes:
                description += f':singleton-classes-kept-in-fit={len(singleton_classes)}'
        if len(valid):
            windows.append({'fit': np.sort(fit), 'valid': np.sort(valid),
                            'description': description, 'period': str(period)})
    if not windows:
        return []
    if len(windows) == 1:
        windows[0]['weight'] = 1.0
    else:
        total = .65 + .35
        windows[0]['weight'] = .65 / total
        windows[1]['weight'] = .35 / total
    return windows


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
    groups, group_desc = _groups_for(bundle)
    group_col = getattr(bundle, 'group_column', None)
    has_duplicates = len(np.unique(groups)) < len(groups)
    windows = temporal_windows(bundle)
    if windows:
        primary = windows[0]
        return primary['fit'], primary['valid'], primary['description']
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
    singleton_classes = []
    if bundle.task_type == 'classification':
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
    if metric == 'smape':
        denominator = np.abs(y_array) + np.abs(pred)
        ratio = np.divide(2 * np.abs(y_array - pred), denominator,
                          out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
        return float(np.mean(ratio) * 100)
    if metric == 'medae': return float(metrics.median_absolute_error(y_array, pred))
    if metric == 'rmsle':
        if np.any(y_array < 0): raise ValueError('RMSLE requires nonnegative actual targets')
        return float(np.sqrt(metrics.mean_squared_log_error(y_array, np.maximum(pred, 0))))
    if metric == 'r2': return float(metrics.r2_score(y_array, pred))
    if metric == 'explained_variance': return float(metrics.explained_variance_score(y_array, pred))
    if metric == 'accuracy': return float(metrics.accuracy_score(y_array, pred))
    if metric == 'balanced_accuracy': return float(metrics.balanced_accuracy_score(y_array, pred))
    if metric == 'cohen_kappa': return float(metrics.cohen_kappa_score(y_array, pred))
    if metric == 'mcc': return float(metrics.matthews_corrcoef(y_array, pred))
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
    if metric not in LOWER_IS_BETTER | {'r2', 'explained_variance', 'accuracy', 'balanced_accuracy',
                                        'roc_auc', 'binary_f1', 'micro_f1', 'macro_f1',
                                        'weighted_f1', 'cohen_kappa', 'mcc'}:
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


def _bootstrap_frames(task_path):
    """Resolve the minimum needed to write a valid file without semantic parsing."""
    task_path = Path(task_path).resolve()
    base = task_path.parent
    text = task_path.read_text(encoding='utf-8-sig', errors='replace') if task_path.is_file() else ''
    try:
        paths = resolve_paths(task_path, text)
        return paths, pd.read_csv(paths['train']), pd.read_csv(paths['test']), pd.read_csv(paths['sample_submission'])
    except Exception:
        pass
    files = sorted(p for p in base.rglob('*.csv') if 'agent_runs' not in p.parts)
    if not files:
        raise FileNotFoundError('No CSV files near task.md')
    frames = {}
    for path in files:
        try:
            frames[path] = pd.read_csv(path, nrows=200)
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
    if not frames:
        raise FileNotFoundError('No readable CSV files near task.md')
    by_name = lambda pattern: [p for p in frames if re.search(pattern, p.stem, re.I)]
    sample_candidates = by_name(r'sample[_ -]?submission|submission[_ -]?sample|sample')
    dictionary = by_name(r'dictionary|schema|metadata')
    if dictionary:
        frames.pop(dictionary[0], None)
    if not sample_candidates:
        rows = {p: len(frame) for p, frame in frames.items()}
        candidates = [p for p, frame in frames.items() if len(frame.columns) <= 3]
        if candidates:
            sample_candidates = sorted(candidates, key=lambda p: (len(frames[p].columns), -rows[p]))
    if not sample_candidates:
        raise ValueError('Cannot identify sample_submission.csv')
    sample_path = sample_candidates[0]
    sample = frames[sample_path]
    remaining = [p for p in frames if p != sample_path]
    train_path = next((p for p in remaining if re.search(r'(^|[_ -])train(ing)?([_ -]|$)', p.stem, re.I)), None)
    test_path = next((p for p in remaining if p != train_path and re.search(r'(^|[_ -])test(ing)?([_ -]|$)', p.stem, re.I)), None)
    if train_path is None:
        train_path = max(remaining, key=lambda p: (len(frames[p].columns), len(frames[p]))) if remaining else None
    if test_path is None:
        test_rows = len(sample)
        candidates = [p for p in remaining if p != train_path and len(frames[p]) == test_rows]
        test_path = candidates[0] if candidates else (remaining[0] if remaining else None)
    if train_path is None or test_path is None:
        raise ValueError('Cannot identify train/test CSVs')
    return {'train': train_path, 'test': test_path, 'sample_submission': sample_path,
            'dictionary': dictionary[0] if dictionary else None}, frames[train_path], frames[test_path], sample


def bootstrap_submission(task_path):
    """Write a structurally valid early submission with zero model training.

    This intentionally avoids metric/model assumptions: it only needs enough
    structure to fill sample_submission. Any later, stronger result replaces it
    atomically.
    """
    try:
        task_path = Path(task_path).resolve()
        paths, train, test, sample = _bootstrap_frames(task_path)
        target = None
        try:
            text = task_path.read_text(encoding='utf-8-sig', errors='replace')
            declared = _text_declared_column(text, 'target')
            if declared in train.columns:
                target = declared
        except Exception:
            pass
        if target is None:
            extra = [c for c in train.columns if c not in test.columns]
            if len(extra) == 1:
                target = extra[0]
        output_columns = list(sample.columns)
        id_column = output_columns[0] if len(output_columns) > 1 else None
        if id_column is None and len(sample.columns) == 1:
            id_column = None
        values = output_columns[1:] if id_column else output_columns
        frame = sample.copy()
        if target is not None and target in train.columns:
            target_values = train[target]
            numeric = _target_looks_numeric(target_values)
            unique = int(target_values.nunique(dropna=True))
            classification = not numeric or unique <= min(50, max(2, int(len(target_values) * .02)))
            if classification:
                labels = pd.unique(target_values.dropna())
                fill = labels[0] if len(labels) else 0
                for column in values:
                    frame[column] = fill
            else:
                numeric_values = pd.to_numeric(target_values, errors='coerce')
                fill = float(numeric_values.mean()) if numeric_values.notna().any() else 0.
                for column in values:
                    frame[column] = fill
        else:
            for column in values:
                if pd.api.types.is_numeric_dtype(sample[column]):
                    frame[column] = 0.
                else:
                    frame[column] = 'unknown'
        # Probability-like submissions: spread the fallback rather than leaving
        # a single class with all mass, which is invalid for some metrics.
        if len(values) > 1:
            for column in values:
                frame[column] = 1. / len(values)
        for column in output_columns:
            if frame[column].isna().any():
                raise ValueError('fallback contains missing values')
        destination = task_path.parent / 'submission.csv'
        temporary = destination.with_name(destination.name + '.bootstrap.tmp')
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
        return destination
    except Exception:
        return None


def emergency_submission(task_path):
    """Final safety net used by solution.py after unexpected failures."""
    path = Path(task_path).resolve().parent / 'submission.csv'
    if path.is_file():
        try:
            if len(pd.read_csv(path)) > 0:
                return path
        except Exception:
            pass
    return bootstrap_submission(task_path)
