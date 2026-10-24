"""Read-only teachers for the original public classification outer split.

``load_class_outer(bundle, fit, valid)`` returns H48 probabilities in the
supplied row order, plus the three component probability dictionaries. Its
training probabilities use OOF base models but an IN-SAMPLE meta learner;
they are suitable soft training targets, never independent inner validation.

``load_class_inner(bundle)`` returns the raw cached fold-0 model's predictions
on its own fit rows and held-out rows. Its ``fit_positions``/``valid_positions``
are GLOBAL bundle positions. No H48 meta state is used in this inner context.

Missing or incompatible artifacts raise; this module never creates directories,
updates cache metadata, or calls fit. Joblib loading is restricted to the
verified original raw fold-0 artifact in this trusted local repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-10-23'
H48 = 'HYP-048-01-multiclass-combinations'
COMPONENT_ORDER = ('raw', 'smote', 'frequency')
OUTER = {'raw': 'HYP-011-02-multiclass-none',
         'smote': 'HYP-038-01-multiclass-spec006',
         'frequency': 'HYP-017-01-multiclass-frequency'}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in (saved.files if keys is None else keys)}


def _indices(values, name, size=None):
    result = np.asarray(values)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in 'iu'
            or len(np.unique(result)) != len(result) or np.any(result < 0)
            or (size is not None and np.any(result >= size))):
        raise ValueError('Invalid or duplicate positions: ' + name)
    return result.astype(np.int64, copy=False)


def validate_probabilities(probabilities, classes, rows):
    """Reject malformed probabilities; normalize accepted row-sum roundoff.

    The historical H48 loader normalized float32 base probabilities in float64;
    the same operation is needed before applying its frozen meta coefficients.
    """
    labels = np.asarray(classes)
    p = np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or not len(labels) or len(np.unique(labels)) != len(labels)
            or p.shape != (rows, len(labels)) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1 + 1e-7)
            or not np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=1e-6)):
        raise ValueError('Invalid cached class probabilities')
    return p / p.sum(axis=1, keepdims=True)


def assemble_oof(folds, positions, classes, expected_folds=None):
    """Assemble already-read fold dictionaries, checking both index systems."""
    positions = _indices(positions, 'original training')
    classes = np.asarray(classes).astype(str)
    result = np.full((len(positions), len(classes)), np.nan)
    fold_id = np.full(len(positions), -1, dtype=int)
    if expected_folds is not None and len(folds) != len(expected_folds):
        raise ValueError('Unexpected OOF fold count')
    for number, saved in enumerate(folds):
        fit = _indices(saved['fit'], 'fold fit', len(positions))
        valid = _indices(saved['valid'], 'fold valid', len(positions))
        if (np.intersect1d(fit, valid).size or len(fit) + len(valid) != len(positions)
                or np.any(fold_id[valid] >= 0)):
            raise ValueError('Overlapping or incomplete OOF fold partition')
        if expected_folds is not None:
            if any(not np.array_equal(got, want) for got, want in
                   zip((fit, valid), expected_folds[number])):
                raise ValueError('OOF local split differs from original grouped split')
        for key, expected in [('fit_positions', positions[fit]),
                              ('valid_positions', positions[valid])]:
            if not np.array_equal(_indices(saved[key], key), expected):
                raise ValueError('OOF global row alignment mismatch: ' + key)
        if not np.array_equal(np.asarray(saved['classes']).astype(str), classes):
            raise ValueError('OOF class order mismatch')
        result[valid] = validate_probabilities(saved['probabilities'], classes, len(valid))
        fold_id[valid] = number
    if np.any(fold_id < 0):
        raise ValueError('Incomplete OOF coverage')
    return {'probabilities': result, 'classes': classes, 'fold_id': fold_id}


def _validate_source(bundle, evidence):
    hashes = bundle.hashes
    expected = evidence.get('hashes', {})
    if not expected:
        raise ValueError('Source evidence lacks data hashes')
    aliases = {'train.csv': 'train', 'test.csv': 'test',
               'sample_submission.csv': 'sample', 'data_dictionary.csv': 'dictionary'}
    for key, value in expected.items():
        if hashes.get(key, hashes.get(aliases.get(key, key))) != value:
            raise ValueError('Source data hash mismatch: ' + key)


def _original_context(bundle, root):
    from automl_data import split_key
    root = Path(root)
    folder = root / 'results' / DATE / H48
    evidence = _json(folder / 'evidence.json')
    _validate_source(bundle, evidence)
    split = _arrays(folder / 'splits.npz')
    fit = _indices(split['fit'], 'outer fit', len(bundle.y))
    valid = _indices(split['valid'], 'outer valid', len(bundle.y))
    if np.intersect1d(fit, valid).size or len(fit) + len(valid) != len(bundle.y):
        raise ValueError('Invalid original outer partition')
    if evidence.get('split_key') != split_key(fit, valid):
        raise ValueError('Original outer evidence split key mismatch')
    return folder, fit, valid


def _read_component(bundle, fit, component, root):
    from experiments.twenty_advanced import inner_fingerprint, inner_folds
    X, y = bundle.X.iloc[fit], np.asarray(bundle.y)[fit]
    if component == 'raw':
        config = {'policy': 'legacy'}
        base = 'series-twenty-hypotheses-v1'
    else:
        config = {'component': component, 'protocol':
                  'H38-bounded-smotenc-v1' if component == 'smote' else 'H17-tagged-frequency-native-v1',
                  'parameters': {}}
        base = 'series-combinations-v1'
    key = inner_fingerprint('multiclass', X, y, fit, config)
    directory = Path(root) / 'results' / DATE / base / 'cache' / key
    metadata = _json(directory / 'metadata.json')
    if metadata.get('key') != key or metadata.get('config') != config:
        raise ValueError('Cached component fingerprint/configuration mismatch')
    expected = inner_folds('multiclass', X, y)
    paths = [directory / f'fold-{number}.npz' for number in range(len(expected))]
    folds = [_arrays(path) for path in paths]
    assembled = assemble_oof(folds, fit, np.unique(y).astype(str), expected)
    assembled.update(directory=directory, folds=folds,
                     audit={'key': key, 'config': config, 'artifacts': list(map(str, paths)),
                            'fingerprint_includes_raw_X_y_positions': True,
                            'model_fits': 0})
    return assembled


def load_class_outer(bundle, fit, valid, *, root=ROOT):
    """Load original-outer H48 teacher; reject any alternative split."""
    from automl_class_combo import predict_meta_state
    folder, expected_fit, expected_valid = _original_context(bundle, root)
    fit = _indices(fit, 'requested fit', len(bundle.y))
    valid = _indices(valid, 'requested valid', len(bundle.y))
    if not np.array_equal(fit, expected_fit) or not np.array_equal(valid, expected_valid):
        raise ValueError('H48 teacher is restricted to the exact original outer split')
    classes = np.unique(np.asarray(bundle.y)[fit]).astype(str)
    cached = {component: _read_component(bundle, fit, component, root)
              for component in COMPONENT_ORDER}
    train = {component: cached[component]['probabilities'] for component in COMPONENT_ORDER}
    outer = {}
    for component in COMPONENT_ORDER:
        source = Path(root) / 'results' / DATE / OUTER[component]
        _validate_source(bundle, _json(source / 'evidence.json'))
        # Early legacy archives contain object-dtype y arrays; read only the
        # numeric/Unicode fields needed here, keeping pickle disabled.
        saved = _arrays(source / 'validation_predictions.npz',
                        keys=['positions', 'classes', 'probabilities'])
        if (not np.array_equal(saved['positions'], valid)
                or not np.array_equal(saved['classes'].astype(str), classes)):
            raise ValueError('Saved outer component row/class mismatch')
        outer[component] = validate_probabilities(saved['probabilities'], classes, len(valid))
    state = _json(folder / 'learned-state.json')['meta_state']
    ptrain = predict_meta_state([train[c] for c in COMPONENT_ORDER], state, classes)
    pvalid = predict_meta_state([outer[c] for c in COMPONENT_ORDER], state, classes)
    reference = _arrays(folder / 'validation_predictions.npz')
    if (not np.array_equal(reference['positions'], valid)
            or not np.array_equal(reference['classes'].astype(str), classes)
            or not np.allclose(pvalid, reference['probabilities'], atol=1e-12, rtol=1e-12)):
        raise ValueError('Reconstructed H48 teacher differs from saved outer teacher')
    return {'train_probabilities': ptrain, 'valid_probabilities': pvalid, 'classes': classes,
            'component_train_probabilities': train, 'component_valid_probabilities': outer,
            'fit_positions': fit, 'valid_positions': valid,
            'audit': {'teacher': 'H48', 'protocol': 'original_outer_only', 'model_fits': 0,
                      'training_base_predictions': 'grouped_oof',
                      'training_meta_predictions': 'meta_in_sample',
                      'independent_training_oof': False, 'inner_validation_available': False,
                      'inner_unavailable_reason': 'Frozen H48 meta learner used all original outer-fit labels',
                      'components': {c: cached[c]['audit'] for c in COMPONENT_ORDER},
                      'meta_state_artifact': str(folder / 'learned-state.json')}}


def load_class_inner(bundle, *, root=ROOT):
    """Load raw fold-0 teacher, whose training excludes every returned valid row."""
    import joblib
    _, outer_fit, _ = _original_context(bundle, root)
    raw = _read_component(bundle, outer_fit, 'raw', root)
    fold = raw['folds'][0]
    fit, valid = fold['fit_positions'], fold['valid_positions']
    model_path = raw['directory'] / 'fold-0-model.joblib'
    model = joblib.load(model_path)
    classes = raw['classes']
    if not np.array_equal(np.asarray(model.classes_).astype(str), classes):
        raise ValueError('Cached raw teacher model class order mismatch')
    ptrain = validate_probabilities(model.predict_proba(bundle.X.iloc[fit]), classes, len(fit))
    pvalid = validate_probabilities(model.predict_proba(bundle.X.iloc[valid]), classes, len(valid))
    if not np.allclose(pvalid, fold['probabilities'], atol=1e-7, rtol=1e-7):
        raise ValueError('Cached raw model does not reproduce held-out probabilities')
    return {'train_probabilities': ptrain, 'valid_probabilities': pvalid, 'classes': classes,
            'fit_positions': fit, 'valid_positions': valid,
            'audit': {'teacher': 'raw_fold_0', 'model_fits': 0,
                      'training_predictions': 'in_sample_raw_teacher',
                      'independent_valid_predictions': True, 'H48_meta_state_used': False,
                      'model_artifact': str(model_path), 'component': raw['audit']}}
