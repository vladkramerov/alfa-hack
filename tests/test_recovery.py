from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_data import load_task, validation_split
from automl_engine import _search_windows
from automl_models import build_pipeline, canonical_candidate, candidate_key
from automl_panel import CausalPanelFeatures
from automl_llm import parse_action
from test_temporal_research import _make_task


def test_temporal_nested_positions_are_global_even_when_rows_are_shuffled(tmp_path):
    bundle = load_task(_make_task(tmp_path / 'task', periods=5, rows_per_period=60))
    bundle = replace(bundle, train=bundle.train.sample(frac=1, random_state=91).reset_index(drop=True))
    outer_fit, outer_valid, _ = validation_split(bundle)
    windows, _ = _search_windows(bundle, outer_fit, 1000)
    for w in windows:
        assert set(w['fit']) <= set(outer_fit)
        assert set(w['valid']) <= set(outer_fit)
        assert not set(np.r_[w['fit'], w['valid']]) & set(outer_valid)
        assert bundle.X.iloc[w['fit']].cohort_period.max() < bundle.X.iloc[w['valid']].cohort_period.min()


def test_discovery_and_history_are_train_only_and_strictly_past():
    X = pd.DataFrame({'arbitrary_measure': np.tile(np.arange(30), 3).astype(float),
                      'arbitrary_period': np.repeat([1, 2, 3], 30)})
    y = X.arbitrary_measure + 10 * X.arbitrary_period
    a = CausalPanelFeatures('arbitrary_period').fit(X, y)
    changed = y.copy(); changed.iloc[60:] += 10000
    b = CausalPanelFeatures('arbitrary_period').fit(X, changed)
    assert a.columns_ == ['arbitrary_measure']
    first = a.transform(X)
    pd.testing.assert_frame_equal(first.iloc[:60], b.transform(X).iloc[:60])
    assert first.iloc[:30].__panel_periods.eq(0).all()
    np.testing.assert_allclose(first.iloc[30:60].__panel_last, np.arange(30) + 10)
    unseen = pd.DataFrame({'arbitrary_measure': [100.], 'arbitrary_period': [4]})
    assert a.transform(unseen).__panel_periods.iloc[0] == 0


@pytest.mark.parametrize('family', ['catboost', 'lightgbm'])
def test_stopping_refits_all_rows_and_records_train_only_probe(family):
    rng = np.random.default_rng(93)
    X = pd.DataFrame({'signal': rng.normal(size=500), 'category': ['a', 'b'] * 250})
    y = np.where(X.signal > .5, 'rare', np.where(X.signal < -.5, 'other', 'normal'))
    params = {'iterations': 120, 'depth': 4} if family == 'catboost' else {'n_estimators': 120}
    candidate = {'family': family, 'early_stopping': True, 'parameters': params}
    if family == 'catboost': candidate['catboost_encoding'] = 'native'
    model = build_pipeline('classification', candidate, 2).fit(X, y)
    assert model.audit_['fit_rows'] == 500
    assert model.audit_['early_stopping']['fit_rows'] < 500
    assert 50 <= model.audit_['early_stopping']['iterations'] <= 120
    assert set(model.predict(X)) == set(y)


def test_geometric_blend_aligns_classes_and_has_distinct_identity():
    X = pd.DataFrame({'signal': np.arange(60), 'category': ['a', 'b'] * 30})
    y = np.where(X.signal > 30, 'normal', 'rare')
    member = {'family': 'extra_trees', 'parameters': {'n_estimators': 50}}
    c = {'family': 'blend', 'members': [member, {'family': 'linear'}], 'weights': [.5, .5]}
    g = {**c, 'blend_mode': 'geometric'}
    assert candidate_key(c) != candidate_key(g)
    model = build_pipeline('classification', g, 1).fit(X, y)
    probs = model.predict_proba(X)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert np.isfinite(probs).all()


def test_qwen_content_wrappers_are_parsed_without_executing_anything():
    for content in ['```json\n{"family":"linear"}\n```', '<think>private reasoning</think>{"family":"linear"}']:
        assert parse_action({'choices': [{'message': {'content': content}}]}) == {'family': 'linear'}
    assert parse_action({'choices': [{'finish_reason': 'length', 'message': {'content': '{"family":"linear"}'}}]}) is None


def test_qwen_reasoning_parameter_has_one_bounded_compatibility_retry():
    import io
    import os
    from unittest.mock import patch
    from urllib.error import HTTPError
    from automl_llm import propose
    calls = []
    def response(request, timeout):
        calls.append(json.loads(request.data))
        if len(calls) == 1:
            raise HTTPError('https://example.invalid/v1', 400, 'unsupported field', {}, None)
        return io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"stop\\":true}"}}]}')
    with patch.dict(os.environ, {'OPENAI_API_KEY':'test-only-key','OPENAI_BASE_URL':'https://example.invalid/v1','OPENAI_MODEL':'test-qwen'}, clear=True), patch('urllib.request.urlopen', side_effect=response):
        action, metadata = propose({}, [], timeout=2)
    assert action == {'stop': True}
    assert metadata['attempts'] == 2
    assert calls[0]['reasoning'] == {'enabled': False}
    assert 'reasoning' not in calls[1]
    assert 'test-only-key' not in json.dumps(metadata)


def test_cached_geometric_ensemble_matches_refit_without_training():
    from unittest.mock import patch
    from test_worker_cache import WorkerProbabilityCacheTests
    from automl_worker import run
    case = WorkerProbabilityCacheTests()
    case.setUp()
    try:
        other = {'family': 'linear', 'parameters': {'C': 2.}}
        status, other_dir, _ = case._run('other', other)
        assert status == 0
        c = {'family': 'blend', 'blend_mode': 'geometric', 'members': [case.base_candidate, other],
             'weights': [.5, .5], 'decision_power': .3}
        manifest, output = case._request('cached_blend', c)
        request = json.loads(manifest.read_text())
        request['reuse_members'] = [str(case.cache_path), str(other_dir / 'probability_cache.npz')]
        manifest.write_text(json.dumps(request))
        with patch('automl_worker.build_pipeline', side_effect=AssertionError('No new fit expected')):
            assert run(manifest) == 0
        metrics = json.loads((output / 'metrics.json').read_text())
        assert metrics['training_reused'] and metrics['fit_seconds'] == 0
        status, direct, _ = case._run('direct_blend', c)
        assert status == 0
        pd.testing.assert_frame_equal(pd.read_csv(output / 'candidate_submission.csv'), pd.read_csv(direct / 'candidate_submission.csv'))
        with np.load(output / 'validation_predictions.npz') as a, np.load(direct / 'validation_predictions.npz') as b:
            np.testing.assert_array_equal(a['prediction'], b['prediction'])
    finally:
        case.tearDown()
