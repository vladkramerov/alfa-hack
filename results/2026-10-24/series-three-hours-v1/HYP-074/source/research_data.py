"""Independent task/CSV boundary, validation splits, metrics and bounded diagnostics.

CSV identifiers and classification targets are read as lexical strings before any
numeric inference. Probability metrics use lexicographically sorted class order;
multiclass output columns follow the sample's lexical class names.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass
class Task:
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    sample: pd.DataFrame
    task_path: Path
    task_type: str
    metric: str
    id_column: str
    target_column: str
    hashes: dict[str, str]
    # Optional lexical source for exact duplicate groups. Subset this with X when
    # constructing an inner task; ordinary manually constructed Tasks also work.
    raw_X: pd.DataFrame | None = field(default=None, repr=False)
    probability_columns: list[str] = field(default_factory=list)
    class_order: list[str] = field(default_factory=list)


_MINIMIZE = {'rmse', 'mse', 'mae', 'rmsle', 'log_loss'}
_MAXIMIZE = {'r2', 'macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy', 'roc_auc'}
_PATTERNS = [
    ('balanced_accuracy', r'balanced[\s_-]*accuracy'),
    ('macro_f1', r'(?:macro[\s_-]*f1|f1[\s_-]*macro)'),
    ('weighted_f1', r'(?:weighted[\s_-]*f1|f1[\s_-]*weighted)'),
    ('rmsle', r'\brmsle\b'), ('rmse', r'\brmse\b'),
    ('mse', r'\bmse\b'), ('mae', r'\bmae\b'),
    ('roc_auc', r'(?:roc[\s_-]*auc|\bauc\b)'),
    ('log_loss', r'log[\s_-]*loss'), ('r2', r'\br[\s²^]*2\b|r²'),
    ('accuracy', r'\baccuracy\b'),
]


def _metric(text: str) -> str:
    clean = text.lower().replace('*', '').replace('`', '')
    # Scope to the primary declaration before considering descriptive references.
    primary = re.search(r'(?:primary[\s_-]*metric|основн\w*\s+метрик\w*)\s*[:=—-]?\s*([^\n.]+)', clean)
    if primary:
        scope = primary.group(1)
    else:
        declaration = re.search(r'(?:metric|метрик\w*)\s*(?:оценки\s*)?[:=—-]?\s*([^\n.]+)', clean)
        scope = declaration.group(1) if declaration else clean
    found = [(match.start(), name) for name, pattern in _PATTERNS
             for match in [re.search(pattern, scope)] if match]
    if not found:
        raise ValueError('task must declare a supported primary metric')
    return min(found)[1]


def _role(path: str) -> str | None:
    name = Path(path.replace('\\', '/')).name.lower()
    if 'sample' in name:
        return 'sample'
    if 'dictionary' in name or 'dict' in name:
        return 'dictionary'
    if re.search(r'(^|[_-])train(?:ing)?([_.-]|$)', name):
        return 'train'
    if re.search(r'(^|[_-])test([_.-]|$)', name):
        return 'test'
    return None


def _paths(task_path: Path, text: str) -> dict[str, Path]:
    declarations: dict[str, set[str]] = {key: set() for key in ('train', 'test', 'sample', 'dictionary')}
    for line in text.splitlines():
        references = re.findall(r'`([^`\n]+\.csv)`|(?<![\w/\\])([\w./\\-]+\.csv)\b', line, flags=re.I)
        for quoted, bare in references:
            token = quoted or bare
            role = _role(token)
            if role is None:
                prefix = line.split(token, 1)[0].lower()
                for key, pattern in [('sample', r'sample|шаблон'), ('dictionary', r'dictionary|словар'), ('train', r'train|обуч'), ('test', r'test|тест')]:
                    if re.search(pattern, prefix):
                        role = key
                        break
            if role:
                declarations[role].add(token)
    result = {}
    files = [source for source in task_path.parent.rglob('*.csv')
             if 'research_runs' not in source.relative_to(task_path.parent).parts]
    for role, names in declarations.items():
        existing = { (task_path.parent / name.replace('\\', '/')).resolve()
                     for name in names if (task_path.parent / name.replace('\\', '/')).is_file() }
        if not existing and names:
            # Task text is sometimes distributed beside relocated files. Only an
            # exact unique basename fallback is allowed; no guessed role path.
            basenames = {Path(name.replace('\\', '/')).name for name in names}
            existing = {p.resolve() for p in files if p.name in basenames}
            if not existing:
                raise FileNotFoundError(f'declared {role} CSV not found: {sorted(names)}')
        if not names:
            existing = {p.resolve() for p in files if _role(str(p)) == role}
        if len(existing) > 1:
            raise ValueError(f'ambiguous {role} CSV paths: {sorted(map(str, existing))}')
        if existing:
            result[role] = existing.pop()
        elif role != 'dictionary':
            raise FileNotFoundError(f'no {role} CSV found')
    if len(set(result.values())) != len(result):
        raise ValueError('different input roles must resolve to different files')
    return result


def _read_lexical(path: Path) -> pd.DataFrame:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if not header or any(not name for name in header) or len(header) != len(set(header)):
        raise ValueError(f'invalid or duplicate CSV header: {path}')
    return pd.read_csv(path, dtype={column: str for column in header}, keep_default_na=False, encoding='utf-8-sig')


def _declared_column(text: str, role: str) -> str | None:
    if role == 'target':
        pattern = r'(?:target(?:[ _-]*column)?|целев\w*\s+(?:переменн\w*|столб\w*))'
    else:
        pattern = r'(?:id[ _-]*column|identifier(?:[ _-]*column)?|идентификатор)'
    match = re.search(pattern + r'\s*[:=]\s*[`"\']?([\w .+-]+?)[`"\']?(?:\n|$|[.;])', text, flags=re.I)
    return match.group(1).strip() if match else None


def _numeric_features(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for column in frame:
        present = raw[column].ne('')
        converted = pd.to_numeric(raw[column].where(present), errors='coerce')
        if present.any() and converted[present].notna().all():
            frame[column] = converted
        else:
            # Only empty CSV cells are missing. NA and similar tokens are
            # legitimate categories rather than pandas' default NA aliases.
            frame[column] = raw[column].where(raw[column].ne(''), np.nan)
    return frame


def load_task(path) -> Task:
    task_path = Path(path).resolve()
    text = task_path.read_text(encoding='utf-8-sig')
    paths = _paths(task_path, text)
    train, test, sample = (_read_lexical(paths[key]) for key in ('train', 'test', 'sample'))
    if train.empty or test.empty or len(sample) != len(test):
        raise ValueError('train/test must be nonempty and sample must align with test')
    metric = _metric(text)
    lowered = text.lower()
    classification = bool(re.search(r'classificat|классификац', lowered))
    regression = bool(re.search(r'regression|регресс', lowered))
    if classification and regression:
        # The metric disambiguates a document mentioning an alternative task.
        task_type = 'regression' if metric in {'rmse', 'mae', 'mse', 'rmsle', 'r2'} else 'classification'
    else:
        task_type = 'classification' if classification else 'regression' if regression else ('regression' if metric in {'rmse', 'mae', 'mse', 'rmsle', 'r2'} else 'classification')
    compatible = {'rmse', 'mae', 'mse', 'rmsle', 'r2'} if task_type == 'regression' else {'macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss'}
    if metric not in compatible:
        raise ValueError(f'primary metric {metric} conflicts with task type {task_type}')
    extra = [column for column in train if column not in test]
    probability_objective = metric in {'roc_auc', 'log_loss'}
    target = _declared_column(text, 'target')
    if target is None:
        candidates = [column for column in extra if column in sample]
        if not candidates and probability_objective:
            candidates = extra
        if len(candidates) != 1:
            raise ValueError(f'ambiguous target column: {candidates}')
        target = candidates[0]
    if target not in train or target in test or (target not in sample and not probability_objective):
        raise ValueError('target must exist only in train and label sample')
    if set(extra) != {target} or set(train) - {target} != set(test):
        raise ValueError('train/test feature schema mismatch or extra target columns')
    if len(sample.columns) != 2 and not probability_objective:
        raise ValueError('sample must have exactly identifier and target columns')
    identifier = _declared_column(text, 'id')
    if identifier is None:
        candidates = [column for column in sample if column != target and column in test]
        if len(candidates) != 1:
            raise ValueError(f'ambiguous identifier column: {candidates}')
        identifier = candidates[0]
    if identifier not in test or identifier not in sample or identifier not in train:
        raise ValueError('identifier must exist in train/test/sample')
    for frame in (train, test, sample):
        if frame[identifier].eq('').any() or frame[identifier].duplicated().any():
            raise ValueError('empty or duplicate identifiers')
    if sample[identifier].tolist() != test[identifier].tolist():
        raise ValueError('sample identifiers do not align with test row order')
    y = train[target].copy()
    if task_type == 'regression':
        y = pd.to_numeric(y, errors='raise')
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError('regression target must contain finite numbers')
    elif y.eq('').any():
        raise ValueError('classification target has empty labels')
    class_order = sorted(y.unique().tolist()) if task_type == 'classification' else []
    probability_columns = [column for column in sample if column != identifier] if probability_objective else []
    if probability_objective:
        if len(probability_columns) == 1:
            if len(class_order) != 2 or probability_columns != [target]:
                raise ValueError('single probability column requires a binary target column')
        elif len(probability_columns) != len(class_order) or set(probability_columns) != set(class_order):
            raise ValueError('multiclass probability columns must match lexical class labels')
    raw_X = train.drop(columns=[identifier, target])
    X_test_raw = test.loc[:, raw_X.columns]
    X = _numeric_features(raw_X)
    # Infer numeric/categorical roles on train only, then apply numeric parsing to
    # test. Unknown lexical numeric values remain missing for downstream imputation.
    X_test = X_test_raw.copy()
    for column in X:
        if pd.api.types.is_numeric_dtype(X[column]):
            X_test[column] = pd.to_numeric(X_test_raw[column].where(X_test_raw[column].ne(''), np.nan), errors='coerce')
        else:
            X_test[column] = X_test_raw[column].where(X_test_raw[column].ne(''), np.nan)
    hashes = {key: hashlib.sha256(source.read_bytes()).hexdigest() for key, source in {**paths, 'task': task_path}.items()}
    return Task(X, y, X_test, sample, task_path, task_type, metric, identifier, target, hashes, raw_X, probability_columns, class_order)


def _temporal_column(task: Task):
    candidates = []
    for column in task.X:
        if not re.search(r'(^|[_\W])(year|time|date|period|month|quarter)([_\W]|$)', str(column), re.I):
            continue
        values = pd.to_numeric(task.X[column], errors='coerce')
        future = pd.to_numeric(task.X_test[column], errors='coerce')
        if values.notna().all() and future.notna().all() and values.nunique() >= 2 and future.min() > values.max():
            candidates.append((column, values))
    if len(candidates) > 1:
        raise ValueError('ambiguous future time columns; declare a unique temporal feature')
    return candidates[0] if candidates else None


def split_data(task: Task):
    """Return positional fit/validation indices and a JSON-serializable protocol."""
    n = len(task.y)
    if len(task.X) != n or n < 4:
        raise ValueError('at least four aligned training rows are required')
    if task.task_type == 'regression':
        temporal = _temporal_column(task)
        if temporal is not None:
            column, values = temporal
            levels = np.sort(values.unique())
            cutoff = levels[max(1, int(np.floor(len(levels) * .8)))]
            fit, valid = np.flatnonzero(values.to_numpy() < cutoff), np.flatnonzero(values.to_numpy() >= cutoff)
            return fit, valid, {'kind': 'temporal', 'column': str(column), 'cutoff': float(cutoff), 'fit_rows': len(fit), 'valid_rows': len(valid)}
        fit, valid = train_test_split(np.arange(n), test_size=.2, random_state=42)
        return np.sort(fit), np.sort(valid), {'kind': 'random', 'random_state': 42, 'fit_rows': len(fit), 'valid_rows': len(valid)}
    raw = task.raw_X
    if raw is None or len(raw) != n or list(raw.columns) != list(task.X.columns) or not raw.index.equals(task.X.index):
        raw = task.X
    # Hash exact source rows without target, ID or transformed features.
    groups = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    labels = task.y.to_numpy()
    counts = task.y.value_counts()
    if len(counts) < 2:
        raise ValueError('classification split needs at least two classes')
    if len(np.unique(groups)) < n:
        class_groups = [len(np.unique(groups[labels == label])) for label in counts.index]
        folds = min(5, min(class_groups))
        if folds < 2:
            raise ValueError('each class needs at least two independent raw-X groups')
        # With very few class groups, sklearn's shuffled heuristic can put a
        # complete class on one side. The rare-class protocol is unshuffled.
        shuffled = folds == 5
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=shuffled, random_state=42 if shuffled else None)
        for fold, (fit, valid) in enumerate(splitter.split(task.X, labels, groups)):
            if set(labels[fit]) == set(labels) and set(labels[valid]) == set(labels):
                return fit, valid, {'kind': 'stratified_group', 'n_splits': folds, 'fold': fold, 'shuffle': shuffled, 'random_state': 42 if shuffled else None, 'groups': int(len(np.unique(groups))), 'fit_rows': len(fit), 'valid_rows': len(valid)}
        raise ValueError('cannot make a class-complete grouped validation split')
    if counts.min() < 2:
        raise ValueError('each class needs at least two rows for stratification')
    size = max(len(counts), int(np.ceil(.2 * n)))
    if n - size < len(counts):
        raise ValueError('too few rows for class-complete stratification')
    fit, valid = train_test_split(np.arange(n), test_size=size, stratify=labels, random_state=42)
    return np.sort(fit), np.sort(valid), {'kind': 'stratified', 'random_state': 42, 'fit_rows': len(fit), 'valid_rows': len(valid)}


def score(metric, y, pred) -> float:
    """Probability columns follow sorted unique labels; binary AUC uses column 1."""
    if metric not in _MINIMIZE | _MAXIMIZE:
        raise ValueError(f'unsupported metric: {metric}')
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError('metric inputs must be nonempty and aligned')
    if metric in {'rmse', 'mse', 'mae', 'rmsle', 'r2'}:
        actual, predicted = y.astype(float), pred.astype(float)
        if predicted.ndim != 1 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError('regression metric inputs must be finite vectors')
        if metric == 'rmse':
            result = np.sqrt(metrics.mean_squared_error(actual, predicted))
        elif metric == 'rmsle':
            if np.any(actual < 0) or np.any(predicted < 0):
                raise ValueError('RMSLE requires nonnegative values')
            result = np.sqrt(metrics.mean_squared_log_error(actual, predicted))
        else:
            result = {'mse': metrics.mean_squared_error, 'mae': metrics.mean_absolute_error, 'r2': metrics.r2_score}[metric](actual, predicted)
    elif metric == 'roc_auc':
        labels = np.unique(y)
        probabilities = pred.astype(float)
        if not np.isfinite(probabilities).all():
            raise ValueError('probabilities must be finite')
        if len(labels) == 2:
            result = metrics.roc_auc_score(y, probabilities[:, 1] if probabilities.ndim == 2 else probabilities)
        else:
            result = metrics.roc_auc_score(y, probabilities, labels=labels, multi_class='ovr', average='macro')
    elif metric == 'log_loss':
        probabilities = pred.astype(float)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError('probabilities must be finite values in [0, 1]')
        if probabilities.ndim == 2 and not np.allclose(probabilities.sum(axis=1), 1.):
            raise ValueError('probability rows must sum to one')
        result = metrics.log_loss(y, probabilities, labels=np.unique(y))
    elif metric in {'macro_f1', 'weighted_f1'}:
        result = metrics.f1_score(y, pred, average='macro' if metric == 'macro_f1' else 'weighted', zero_division=0)
    else:
        result = (metrics.accuracy_score if metric == 'accuracy' else metrics.balanced_accuracy_score)(y, pred)
    if not np.isfinite(result):
        raise ValueError('metric produced a nonfinite result')
    return float(result)


def better(metric, new, old) -> bool:
    if metric not in _MINIMIZE | _MAXIMIZE:
        raise ValueError(f'unsupported metric: {metric}')
    if not np.isfinite(new) or not np.isfinite(old):
        raise ValueError('metric comparison requires finite scores')
    return bool(new < old if metric in _MINIMIZE else new > old)


def save_submission(task: Task, pred, path=None) -> Path:
    """Validate before atomic replacement; invalid output never replaces a file."""
    frame = task.sample.copy()
    if isinstance(pred, pd.DataFrame):
        if list(pred.columns) != list(frame.columns):
            raise ValueError('submission columns must match sample columns/order')
        if pred[task.id_column].tolist() != frame[task.id_column].tolist():
            raise ValueError('submission identifier order must match sample')
        values = pred.loc[:, task.probability_columns].to_numpy() if task.probability_columns else pred[task.target_column].to_numpy()
        if len(task.probability_columns) == 1:
            values = values[:, 0]
    else:
        values = np.asarray(pred)
    if values.ndim == 0 or len(values) != len(frame):
        raise ValueError('predictions must match test row count')
    if task.probability_columns:
        required_shape = (len(frame),) if len(task.probability_columns) == 1 else (len(frame), len(task.probability_columns))
        if values.shape != required_shape:
            raise ValueError('prediction probability shape must match sample columns')
        values = values.astype(float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError('submission probabilities must be finite values in [0, 1]')
        if values.ndim == 2 and not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-6):
            raise ValueError('submission probability rows must sum to one')
    elif values.ndim != 1:
        raise ValueError('predictions must be a vector matching test row count')
    elif task.task_type == 'classification':
        allowed = set(task.y.tolist())
        if any(pd.isna(value) or value not in allowed for value in values):
            raise ValueError('submission contains an unknown or missing class')
    else:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('submission predictions must be finite')
    if task.probability_columns:
        if values.ndim == 1:
            frame[task.probability_columns[0]] = values
        else:
            frame[task.probability_columns] = values
    else:
        frame[task.target_column] = values
    if frame[task.id_column].isna().any() or frame[task.id_column].duplicated().any():
        raise ValueError('submission identifiers must be nonmissing and unique')
    destination = Path(path) if path is not None else Path(task.task_path).parent / 'submission.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8', newline='') as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        written = _read_lexical(Path(temporary))
        if list(written.columns) != list(frame.columns) or written[task.id_column].tolist() != frame[task.id_column].tolist() or len(written) != len(task.X_test):
            raise ValueError('serialized submission schema/identifier order mismatch')
        if task.task_type == 'classification' and not task.probability_columns and written[task.target_column].tolist() != [str(value) for value in values]:
            raise ValueError('serialized submission changed lexical classes')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def diagnostics(task_type, y, pred, classes=None, probabilities=None) -> dict:
    """Bounded inner-only evidence; residual bias means predicted minus actual."""
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or pred.ndim != 1 or len(y) != len(pred) or not len(y):
        raise ValueError('diagnostics require nonempty aligned label vectors')
    if task_type == 'classification':
        labels = list(dict.fromkeys(list(classes) if classes is not None else [*y.tolist(), *pred.tolist()]))
        precision, recall, f1, support = metrics.precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        rows = [{'class': str(label), 'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
                for label, p, r, f, s in zip(labels, precision, recall, f1, support)]
        confusion: dict[tuple[str, str], int] = {}
        for actual, predicted in zip(y, pred):
            if actual != predicted:
                key = (str(actual), str(predicted))
                confusion[key] = confusion.get(key, 0) + 1
        pairs = [{'actual': a, 'predicted': p, 'count': count} for (a, p), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0]))[:20]]
        result = {'task_type': task_type, 'rows': len(y), 'classes': len(labels), 'per_class': rows[:30], 'omitted_classes': max(0, len(rows) - 30), 'confusion_pairs': pairs}
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=float)
            if probabilities.ndim != 2 or probabilities.shape != (len(y), len(labels)) or not np.isfinite(probabilities).all():
                raise ValueError('diagnostic probability shape/values are invalid')
            result['mean_confidence'] = float(np.mean(np.max(probabilities, axis=1)))
        return result
    if task_type != 'regression':
        raise ValueError('unknown diagnostic task type')
    actual, predicted = y.astype(float), pred.astype(float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('regression diagnostics require finite values')
    residual = predicted - actual
    segments = []
    for indices in np.array_split(np.argsort(actual, kind='stable'), min(5, len(actual))):
        errors = residual[indices]
        segments.append({'target_min': float(actual[indices].min()), 'target_max': float(actual[indices].max()), 'rows': len(indices), 'rmse': float(np.sqrt(np.mean(errors ** 2))), 'bias': float(errors.mean())})
    return {'task_type': task_type, 'rows': len(y), 'rmse': float(np.sqrt(np.mean(residual ** 2))), 'mae': float(np.mean(np.abs(residual))), 'bias': float(residual.mean()), 'residual_quantiles': [float(value) for value in np.quantile(residual, [0, .1, .5, .9, 1])], 'target_segments': segments}
