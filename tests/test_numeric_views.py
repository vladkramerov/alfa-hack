import numpy as np
import pandas as pd
import pytest
from automl_models import _NumericViews, build_pipeline, canonical_candidate, candidate_key


def test_rank_is_fit_only_and_preserves_raw():
    train = pd.DataFrame({'x': [1., 2., 2., 4.], 'category': ['a'] * 4})
    view = _NumericViews('rank').fit(train)
    test = pd.DataFrame({'x': [-100., 2., 900., np.nan], 'category': ['new'] * 4})
    result = view.transform(test)
    assert result['x'].equals(test['x'])
    assert result['category'].equals(test['category'])
    np.testing.assert_allclose(result['__research_rank_0'].iloc[:3], [0, .5, 1])
    assert np.isnan(result['__research_rank_0'].iloc[3])
    assert len(view.tables_['x']) == 4


def test_projection_bounded_deterministic_and_nondefault_index():
    train = pd.DataFrame({'x': [1., 1., np.nan], 'y': [np.nan] * 3}, index=[9, 3, 7])
    test = pd.DataFrame({'x': [np.inf, -1e100], 'y': [np.nan, 3.]}, index=[5, 8])
    a = _NumericViews('rank_projection').fit(train).transform(test)
    b = _NumericViews('rank_projection').fit(train).transform(test)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.index) == [5, 8]
    assert np.isfinite(a.filter(like='__research_projection_').to_numpy()).all()


def test_numeric_modes_are_distinct_and_validated():
    base = {'family': 'extra_trees'}
    assert candidate_key(base) == candidate_key({**base, 'numeric_view': 'none'})
    assert candidate_key(base) != candidate_key({**base, 'numeric_view': 'rank'})
    with pytest.raises(ValueError):
        canonical_candidate({**base, 'numeric_view': 'bad'})
    with pytest.raises(ValueError):
        _NumericViews('rank').fit(pd.DataFrame({'x': [1], '__research_rank_0': [2]}))


@pytest.mark.parametrize('mode', ['rank', 'signed_log', 'projection', 'rank_projection'])
def test_fit_predict_both_tasks(mode):
    X = pd.DataFrame({'x': np.linspace(-10, 10, 60), 'cat': ['a', 'b'] * 30})
    candidate = {'family': 'extra_trees', 'parameters': {'n_estimators': 50}, 'numeric_view': mode}
    reg = build_pipeline('regression', candidate, 1).fit(X, X.x ** 2)
    assert np.isfinite(reg.predict(X)).all()
    clf = build_pipeline('classification', candidate, 1).fit(X, np.where(X.x > 0, 'yes', 'no'))
    assert clf.predict_proba(X).shape == (60, 2)
    assert set(clf.predict(X)) == {'yes', 'no'}


def test_scheduler_proposes_measured_view_only_for_log_tree_regression():
    from automl_engine import offline_proposal
    profile = {'type': 'regression', 'metric': 'rmse', 'shape': [2000, 10],
               'numeric_target_correlations': {}, 'target_min': 0, 'target_skew': 2,
               'categories': {}, 'nunique': {'x': 20}, 'drift_columns': []}
    candidate = canonical_candidate({'family': 'extra_trees', 'target_transform': 'log1p'})
    observations = [{'status': 'completed', 'candidate': candidate, 'score': 5., 'diagnostics': {}}]
    seen = {candidate_key(candidate), candidate_key({'family': 'lightgbm', 'parameters': {'n_estimators': 120}})}
    proposal = offline_proposal(profile, observations, seen)
    assert proposal['numeric_view'] == 'signed_log'
    seen.add(candidate_key(proposal))
    assert offline_proposal(profile, observations, seen).get('numeric_view') is None
    # The same observation must not offer regression target/feature policy to classification.
    classifier = canonical_candidate({'family': 'extra_trees'})
    profile.update(type='classification', metric='macro_f1')
    proposal = offline_proposal(profile, [{'status': 'completed', 'candidate': classifier, 'score': .5}], set())
    assert proposal.get('numeric_view') is None
