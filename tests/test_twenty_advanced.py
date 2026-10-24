"""Safety contracts for the approved nested advanced experiments."""
import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from experiments import twenty_advanced as advanced


def test_smearing_corrects_exp_before_subtracting_one():
    # Adding log(2) residuals must double (prediction + 1), not prediction.
    factor, diagnostic = advanced.smearing_factor(np.array([3., 7.]), np.log1p([1., 3.]))
    np.testing.assert_allclose(factor, 2.)
    assert diagnostic['accepted']
    np.testing.assert_allclose(advanced.apply_smearing([1., 3.], factor), [3., 7.])


def test_smearing_rejects_unstable_fold_factors():
    factor, diagnostic = advanced.smearing_factor([1., 99.], [0., 0.], [0, 1])
    assert factor == 1.
    assert not diagnostic['accepted']


def test_grouped_inner_folds_never_split_raw_duplicates():
    X = pd.DataFrame({'x': np.repeat(np.arange(30), 2), 'kind': np.repeat(['a', 'b', 'c'] * 10, 2)})
    y = np.repeat(['a', 'b', 'c'] * 10, 2)
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    folds = advanced.inner_folds('multiclass', X, y)
    assert len(folds) == 3
    assert sorted(np.concatenate([v for _, v in folds]).tolist()) == list(range(60))
    for fit, valid in folds:
        assert not set(groups[fit]) & set(groups[valid])


def test_temporal_inner_folds_use_only_prior_years():
    X = pd.DataFrame({'Year_Factor': np.repeat([1, 2, 3, 4], 3)})
    folds = advanced.inner_folds('regression', X, np.arange(12))
    assert len(folds) == 2
    for fit, valid in folds:
        assert X.iloc[fit].Year_Factor.max() < X.iloc[valid].Year_Factor.min()


def test_multiplier_selection_uses_training_oof_and_fixed_anchor():
    p = np.array([[.6, .4], [.55, .45], [.9, .1], [.2, .8]])
    factors, audit = advanced.choose_multipliers(['b', 'b', 'a', 'b'], p, ['a', 'b'])
    assert factors[0] == 1.
    assert factors[1] > 1.
    np.testing.assert_array_equal(np.array(['a', 'b'])[np.argmax(p * factors, axis=1)], ['b', 'b', 'a', 'b'])
    assert audit['selection_source'] == 'training_grouped_oof'


def test_pair_expert_preserves_mass_and_gates_confident_rows():
    p = np.array([[.45, .4, .15], [.8, .1, .1], [.1, .45, .45]])
    result, gate = advanced.redistribute_pair(p, [0, 1], np.array([[.1, .9], [.1, .9], [.1, .9]]))
    np.testing.assert_allclose(result[0], [.085, .765, .15])
    np.testing.assert_allclose(result[1:], p[1:])
    np.testing.assert_allclose(result.sum(axis=1), 1.)
    np.testing.assert_array_equal(gate, [True, False, False])


def test_pair_selection_excludes_normal_and_breaks_ties_lexically():
    classes = np.array(['Normal', 'a', 'b', 'c'])
    y = np.array(['Normal', 'a', 'a', 'b', 'c'])
    pred = np.array(['a', 'b', 'c', 'a', 'a'])
    assert advanced.confused_pair(y, pred, classes) == ('a', 'b')


def test_stability_selection_caps_deletions_and_never_drops_positive():
    importance = np.array([[1., -.2, -.3, .1, -.1], [2., -.1, -.2, .2, .1]])
    assert advanced.stable_feature_selection(list('abcde'), importance) == ['a', 'b', 'd', 'e']


def test_smote_masks_binary_and_protocol_columns_and_bounds_rows():
    X = pd.DataFrame({'proto': [6, 17, 6], 'flag': [0, 1, 0], 'size': [1., 2., 3.]})
    assert advanced.smote_categorical_columns(X) == ['proto', 'flag']
    assert advanced.smote_strategy(np.array(['a'] * 6 + ['b'] * 1000 + ['c'] * 2)) == {'a': 12}


def test_cache_fingerprint_changes_with_raw_x_target_and_positions():
    X = pd.DataFrame({'x': [1., 2.]})
    a = advanced.inner_fingerprint('regression', X, np.array([1., 2.]), np.array([4, 5]), {'policy': 'light'})
    b = advanced.inner_fingerprint('regression', X, np.array([1., 3.]), np.array([4, 5]), {'policy': 'light'})
    c = advanced.inner_fingerprint('regression', X, np.array([1., 2.]), np.array([5, 4]), {'policy': 'light'})
    assert len({a, b, c}) == 3


def test_inner_cache_reuses_models_and_records_raw_positions():
    X = pd.DataFrame({'Year_Factor': np.repeat([1, 2, 3], 4), 'x': np.arange(12.)})
    y = np.arange(12.) + 1
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        context = {'cache_dir': Path(folder), 'fit_positions': np.arange(100, 112)}
        config = {'policy': 'light', 'parameters': {'n_estimators': 3, 'max_depth': 2}}
        first = advanced.cached_oof('regression', X, y, context, config)
        with patch.object(advanced, '_make', side_effect=AssertionError('a cached fit repeated')):
            second = advanced.cached_oof('regression', X, y, context, config)
        np.testing.assert_array_equal(first['prediction'], second['prediction'])
        assert all(fold['status'] == 'reused' for fold in second['audit']['folds'])
        folder = Path(folder) / first['audit']['key']
        assert len(list(folder.glob('*-model.joblib'))) == 2
        with np.load(folder / 'fold-0.npz') as saved:
            np.testing.assert_array_equal(saved['fit_positions'], [100, 101, 102, 103])
            np.testing.assert_array_equal(saved['valid_positions'], [104, 105, 106, 107])


def test_smearing_uses_saved_outer_prediction_without_outer_fit():
    X = pd.DataFrame({'Year_Factor': [1, 2, 3]})
    oof = {'fold_id': np.array([-1, 0, 1]), 'prediction': np.array([np.nan, 1., 3.]), 'audit': {}}
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        with patch.object(advanced, 'cached_oof', return_value=oof), patch.object(advanced, '_make', side_effect=AssertionError('outer baseline refit')):
            result = advanced.evaluate_advanced(33, 'regression', X, [0., 3., 7.], X.iloc[:1],
                                                {'cache_dir': folder, 'baseline_prediction': np.array([5.])})
    np.testing.assert_allclose(result['prediction'], [11.])


def test_selected_route_rejects_saved_predictions_in_wrong_row_order():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        artifact = Path(folder) / 'candidate.npz'
        np.savez(artifact, prediction=[1., 2.], positions=[8, 7])
        with np.testing.assert_raises_regex(ValueError, 'positions'):
            advanced.load_outer_artifact(artifact, 'regression', pd.DataFrame({'x': [1, 2]}),
                                          {'valid_positions': np.array([7, 8])}, {})


def test_smote_domain_rejects_non_tcp_with_unsupported_tcp_measurements():
    X = pd.DataFrame({'proto': ['tcp', 'tcp', 'udp', 'udp'], 'tcprtt': [.2, .1, 0., 0.],
                      'spkts': [3, 4, 2, 3], 'dur': [.2, .3, .1, .2]})
    generated = pd.DataFrame({'proto': ['udp', 'tcp'], 'tcprtt': [.1, .15],
                              'spkts': [2.3, 3.7], 'dur': [.12, .25]})
    fixed, valid, audit = advanced.constrain_synthetic(X, generated, ['spkts', 'tcprtt', 'dur'])
    np.testing.assert_array_equal(valid, [False, True])
    np.testing.assert_allclose(fixed.spkts, [2, 4])
    np.testing.assert_allclose(fixed.dur, [.12, .25])
    assert audit['rounded_integer_fields'] == ['spkts']


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))
