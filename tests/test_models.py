import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from automl_models import build_pipeline, candidate_key, canonical_candidate


def _classification_data():
    X = pd.DataFrame({
        'amount': [0., 1., 2., 3., 4., 5., 6., np.inf],
        'kind': ['a', 'a', 'a', 'b', 'b', 'b', 'rare', None],
    })
    y = pd.Series(['normal', 'normal', 'normal', 'attack', 'attack', 'attack', 'rare class', 'normal'])
    return X, y


def test_rf_classification_labels_probabilities_frequency_and_generated_features():
    X, y = _classification_data()
    candidate = {
        'family': 'random_forest',
        'parameters': {'n_estimators': 50, 'max_depth': 5},
        'frequency_columns': ['kind'],
        'feature_operations': [{'op': 'div', 'name': 'amount_per_count', 'left': 'amount', 'right': 'amount'}],
        'class_weight_power': .3,
        'decision_power': .2,
    }
    model = build_pipeline('classification', candidate, threads=2).fit(X, y)
    X_new = pd.DataFrame({'amount': [1., 3.], 'kind': ['unseen', 'b']})
    probabilities = model.predict_proba(X_new)
    assert probabilities.shape == (2, len(model.classes_))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert set(model.predict(X_new)).issubset(set(y))
    assert model.audit_['fit_rows'] == len(X)
    assert model.audit_['frequency_columns'] == ['kind']
    assert "def engineer(X):" in model.feature_code_
    assert model.frequency_.tables_['kind'].get('unseen') is None


def test_decision_power_changes_only_predicted_decision():
    X, y = _classification_data()
    base = build_pipeline('classification', {'family': 'random_forest', 'parameters': {'n_estimators': 60}}).fit(X, y)
    adjusted = build_pipeline('classification', {'family': 'random_forest', 'parameters': {'n_estimators': 60}, 'decision_power': .7}).fit(X, y)
    np.testing.assert_allclose(base.predict_proba(X), adjusted.predict_proba(X))


def test_integer_labels_keep_numeric_dtype_and_macro_f1_compatibility():
    X = pd.DataFrame({'x': np.arange(15, dtype=float)})
    y = pd.Series([0, 2, 10] * 5)
    model = build_pipeline('classification', {'family': 'random_forest',
        'parameters': {'n_estimators': 40}}).fit(X, y)
    prediction = model.predict(X)
    assert model.classes_.dtype.kind in 'iu'
    assert prediction.dtype.kind in 'iu'
    assert f1_score(y, prediction, average='macro') >= 0


def test_regression_log_target_and_feature_operations():
    X = pd.DataFrame({'x': np.arange(12, dtype=float), 'group': ['a', 'b'] * 6})
    y = pd.Series(np.exp(np.arange(12, dtype=float) / 4))
    candidate = {'family': 'extra_trees', 'parameters': {'n_estimators': 40},
                 'target_transform': 'log1p',
                 'feature_operations': [{'op': 'log1p', 'name': 'log_x', 'left': 'x'}]}
    model = build_pipeline('regression', candidate).fit(X, y)
    pred = model.predict(X.iloc[:3])
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_log_target_uses_its_actual_mathematical_domain():
    X = pd.DataFrame({'x': np.arange(12, dtype=float)})
    candidate = {'family': 'extra_trees', 'parameters': {'n_estimators': 50}, 'target_transform': 'log1p'}
    model = build_pipeline('regression', candidate).fit(X, np.linspace(-.5, 2, len(X)))
    assert np.isfinite(model.predict(X)).all()
    with pytest.raises(ValueError, match='greater than -1'):
        build_pipeline('regression', candidate).fit(X, np.linspace(-1, 2, len(X)))
    assert model.predict(X.iloc[:1]).shape == (1,)


@pytest.mark.parametrize('family', ['catboost', 'lightgbm', 'xgboost'])
def test_native_boosting_families_are_usable(family):
    X, y = _classification_data()
    if family in {'lightgbm', 'xgboost'}:
        try:
            if family == 'lightgbm':
                import lightgbm  # noqa: F401
            else:
                import xgboost  # noqa: F401
        except OSError as exc:
            pytest.skip(f'{family} native runtime unavailable: {exc}')
        except Exception as exc:
            if 'libomp' in str(exc).lower():
                pytest.skip(f'{family} native runtime unavailable: {exc}')
            raise
    params = {'catboost': {'iterations': 60, 'depth': 4},
              'lightgbm': {'n_estimators': 60, 'max_depth': 4},
              'xgboost': {'n_estimators': 60, 'max_depth': 4}}[family]
    model = build_pipeline('classification', {'family': family, 'parameters': params}).fit(X, y)
    assert model.predict_proba(X.iloc[:2]).shape == (2, len(model.classes_))
    assert set(model.predict(X.iloc[:2])).issubset(set(y))
    if family == 'xgboost':
        assert model.model_.get_xgb_params()['objective'] == 'multi:softprob'
        assert model.model_.get_xgb_params()['num_class'] == len(model.classes_)


def test_xgboost_binary_classification_uses_binary_objective():
    try:
        import xgboost  # noqa: F401
    except Exception as exc:
        if 'libomp' in str(exc).lower():
            pytest.skip(f'xgboost native runtime unavailable: {exc}')
        raise
    X = pd.DataFrame({'x': np.arange(24, dtype=float), 'kind': ['a', 'b'] * 12})
    y = pd.Series([0, 1] * 12)
    model = build_pipeline('classification', {'family': 'xgboost',
        'parameters': {'n_estimators': 50, 'max_depth': 3}}).fit(X, y)
    assert model.model_.get_xgb_params()['objective'] == 'binary:logistic'
    assert model.predict_proba(X.iloc[:4]).shape == (4, 2)


@pytest.mark.parametrize('encoding', ['native', 'onehot'])
def test_catboost_categorical_representations_are_distinct_usable_candidates(encoding):
    X, y = _classification_data()
    candidate = {'family': 'catboost', 'catboost_encoding': encoding,
                 'parameters': {'iterations': 50, 'depth': 4}}
    model = build_pipeline('classification', candidate).fit(X, y)
    future = pd.DataFrame({'amount': [np.nan, 4.], 'kind': ['unseen', None]})
    assert model.predict_proba(future).shape == (2, 3)
    assert model.audit_['categorical_handling'] == encoding
    other = {**candidate, 'catboost_encoding': 'onehot' if encoding == 'native' else 'native'}
    assert candidate_key(candidate) != candidate_key(other)


def test_group_experts_train_only_eligible_groups_and_keep_global_fallback():
    rng = np.random.default_rng(77)
    n = 420
    group = np.asarray(['a', 'b', 'c'])[rng.integers(0, 3, n)]
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({'segment': group, 'signal': signal, 'noise': noise})
    y = (100 + 3 * signal + np.where(group == 'a', 6., np.where(group == 'b', -4., 0.)) + .1 * noise)
    candidate = {'family': 'extra_trees', 'parameters': {'n_estimators': 60, 'max_depth': 8},
                 'target_transform': 'log1p', 'expert_column': 'segment',
                 'expert_min_rows': 100, 'max_expert_groups': 2}
    model = build_pipeline('regression', candidate, threads=2).fit(X, y)
    self_prediction = model.predict(X)
    assert np.isfinite(self_prediction).all()
    assert model.audit_['expert_column'] == 'segment'
    assert model.audit_['expert_models'] == 2
    assert model.audit_['expert_skipped_groups'] == []
    unseen = pd.DataFrame({'segment': ['unseen'], 'signal': [0.], 'noise': [0.]})
    assert np.isfinite(model.predict(unseen)).all()
    assert candidate_key(candidate) != candidate_key({**candidate, 'expert_column': 'noise'})

    labels = pd.Series(np.where(y > 103, 'high', 'low'))
    classifier = build_pipeline('classification', {**candidate, 'target_transform': 'none'}, threads=2).fit(X, labels)
    probabilities = classifier.predict_proba(X)
    assert probabilities.shape == (n, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert classifier.audit_['expert_models'] >= 1


def test_blend_rejects_group_experts_outside_members():
    with pytest.raises(ValueError, match='Blend transformations and group experts'):
        canonical_candidate({'family': 'blend', 'expert_column': 'kind',
                             'members': [{'family': 'extra_trees'}, {'family': 'linear'}],
                             'weights': [.5, .5]})


def test_blend_is_bounded_and_aligns_probabilities():
    X, y = _classification_data()
    candidate = {'family': 'blend', 'members': [
        {'family': 'random_forest', 'parameters': {'n_estimators': 40}},
        {'family': 'extra_trees', 'parameters': {'n_estimators': 40}},
    ], 'weights': [.25, .75]}
    model = build_pipeline('classification', candidate).fit(X, y)
    assert model.predict_proba(X).shape == (len(X), len(model.classes_))
    assert len(model.predict(X)) == len(X)
    assert model.feature_code_ == ''
    adjusted_members = {**candidate, 'members': [
        {**candidate['members'][0], 'decision_power': .5}, candidate['members'][1]]}
    assert candidate_key(candidate) == candidate_key(adjusted_members)


def test_canonicalization_ignores_nonsemantic_fields_and_rejects_code_and_nested_blends():
    a = {'family': 'random_forest', 'hypothesis': 'first', 'parameters': {'n_estimators': 150}}
    b = {'family': 'random_forest', 'hypothesis': 'second', 'seed': 999}
    assert candidate_key(a) == candidate_key(b)
    assert canonical_candidate(a)['parameters']['n_estimators'] == 150
    with pytest.raises(ValueError, match='feature_code'):
        canonical_candidate({'family': 'linear', 'feature_code': 'import os'})
    with pytest.raises(ValueError, match='Unknown parameter'):
        canonical_candidate({'family': 'linear', 'parameters': {'surprise': 99}})
    with pytest.raises(ValueError, match='nested'):
        canonical_candidate({'family': 'blend', 'members': [
            {'family': 'blend', 'members': [
                {'family': 'linear'}, {'family': 'linear'}], 'weights': [.5, .5]},
            {'family': 'linear'}], 'weights': [.5, .5]})


def test_canonicalization_is_idempotent_for_native_lightgbm_defaults():
    candidate = {'family': 'lightgbm'}
    assert canonical_candidate(canonical_candidate(candidate)) == canonical_candidate(candidate)
    assert candidate_key(candidate) == candidate_key(canonical_candidate(candidate))


def test_tree_family_defaults_are_bounded_and_part_of_candidate_identity():
    et = canonical_candidate({'family': 'extra_trees'})['parameters']
    rf = canonical_candidate({'family': 'random_forest'})['parameters']
    assert et == {'n_estimators': 200, 'max_depth': 24, 'min_samples_leaf': 2,
                  'min_samples_split': 2, 'max_features': .8}
    assert rf == {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 2,
                  'min_samples_split': 2, 'max_features': .8}
