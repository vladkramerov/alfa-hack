import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sprint_uncertainty import ConditionalLognormalRegressor


class FixedPoolBase:
    fit_calls = 0

    def __init__(self, candidate=None, threads=4):
        self.candidate, self.threads = candidate, threads

    def fit(self, X, y):
        type(self).fit_calls += 1
        self.audit_ = {'fit_original_rows': len(X), 'fit_retained_rows': len(X)}
        return self

    def predict(self, X):
        return np.full(len(X), 100.)


class FixedPoolLognormal(ConditionalLognormalRegressor):
    fit_calls = 0

    def fit(self, X, y, base_predictor=None):
        type(self).fit_calls += 1
        self.audit_ = {'fit_original_rows': len(X), 'fit_retained_rows': len(X)}
        return self

    def predict(self, X):
        return np.full(len(X), 200.)


def frame():
    return pd.DataFrame({'State_Factor': ['S']*4, 'building_class': ['Commercial']*4,
        'facility_type': ['Office']*4, 'floor_area': [1000.]*4, 'year_built': [1950.]*4,
        'ELEVATION': [10.]*4, 'energy_star_rating': [50.]*4, 'Year_Factor': [1.,2.,3.,4.]})


def candidate(weights=None):
    return {'strategy': 'temporal_transport_pool', 'family': 'extra_trees', 'parameters': {},
        'pool_weights': weights if weights is not None else [.43908725498434387,.5446376329320275,.016275112083628654]}


class SprintPoolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('automl_sprint_pool'), 'pool adapter missing')
        self.module = importlib.import_module('automl_sprint_pool')
        FixedPoolBase.fit_calls = FixedPoolLognormal.fit_calls = 0

    def test_actual_transports_share_one_base_and_fixed_pool_matches_hand_values(self):
        X = frame(); original = candidate(); before = copy.deepcopy(original)
        with patch.object(self.module, 'TemporalProfileModel', FixedPoolBase), patch.object(
                self.module, 'ConditionalLognormalRegressor', FixedPoolLognormal):
            model = self.module.TemporalTransportPool(original, threads=4).fit(X.iloc[:3], [100.,80.,60.])
        self.assertEqual(FixedPoolBase.fit_calls, 1); self.assertEqual(FixedPoolLognormal.fit_calls, 1)
        self.assertIs(model.components_[0].base_predictor_, model.base_)
        self.assertIs(model.components_[1].base_predictor_, model.base_)
        # Three strictly-past donors: w=.95*3/3.25=57/65.
        # Ratio optimum=(100*80+80*60)/(100²+80²)=32/41.
        expected_components = np.array([[(57/65)*60+(8/65)*100,
                                        (57/65)*(60*32/41)+(8/65)*100, 200.]])
        np.testing.assert_allclose(model.predict_components(X.iloc[3:]), expected_components)
        np.testing.assert_allclose(model.predict(X.iloc[3:]), expected_components@np.array(before['pool_weights']))
        self.assertTrue(model.keep_mask_.all()); self.assertEqual(model.removed_positions_.tolist(), [])
        self.assertEqual(model.audit_['fit_retained_rows'], 3)
        self.assertEqual(original, before)
        original['pool_weights'][0] = 999.
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X.iloc[3:]), model.predict(X.iloc[3:]))
        self.assertEqual(restored.components_[0].base_predictor_, restored.base_)
        json.dumps(model.configuration()); json.dumps(model.audit_)

    def test_optional_zero_components_skip_fits_and_prediction_is_batch_invariant(self):
        X = frame()
        with patch.object(self.module, 'TemporalProfileModel', FixedPoolBase), patch.object(
                self.module, 'ConditionalLognormalRegressor', FixedPoolLognormal):
            model = self.module.TemporalTransportPool(candidate([0.,0.,1.]), threads=1).fit(X.iloc[:3], [100.,80.,60.])
        self.assertEqual(FixedPoolBase.fit_calls, 0); self.assertEqual(FixedPoolLognormal.fit_calls, 1)
        self.assertIsNone(model.base_); self.assertIsNone(model.components_[0]); self.assertIsNone(model.components_[1])
        self.assertEqual(model.components_[2].threads, 1)
        np.testing.assert_array_equal(model.predict(X), [200.]*4)
        np.testing.assert_array_equal(model.predict(X.iloc[[3]]), model.predict(X)[[3]])

    def test_canonical_pool_strict_fields_simplex_and_no_mutation(self):
        c = candidate(); original = copy.deepcopy(c)
        result = self.module.canonical_pool(c)
        self.assertEqual(c, original)
        self.assertEqual(result['preprocessing'], 'temporal-transport-pool-v1')
        self.assertEqual(result['target_transform'], 'none')
        self.assertEqual(result['pool_weights'], original['pool_weights'])
        for update in ({'pool_weights': [1.,0.]}, {'pool_weights': [.5,.5,.1]}, {'pool_weights': [1.,-1.,1.]},
                       {'pool_weights': [np.nan,0.,1.]}, {'pool_weights': [True,False,False]},
                       {'family': 'xgboost'}, {'parameters': {'n_estimators':1}},
                       {'preprocessing': 'full-v1'}, {'scale': True}, {'max_outlier_fraction': .05},
                       {'target_transform':'log1p'}, {'feature_code':'x=1'}, {'feature_numbers':[29]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.module.canonical_pool({**c, **update})
        missing = copy.deepcopy(c); missing.pop('pool_weights')
        with self.assertRaises(ValueError): self.module.canonical_pool(missing)

    def test_journal_registers_exactly_base_and_three_components_before_fitting(self):
        X = frame()
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as directory:
            root = Path(directory); (root/'ledger.md').write_text('prior evidence',encoding='utf-8')
            env = {'COMBO_RUN_DIR':str(root/'run'),'COMBO_LEDGER_PATH':str(root/'ledger.md')}
            with patch.dict(os.environ,env), patch.object(self.module,'TemporalProfileModel',FixedPoolBase), patch.object(
                    self.module,'ConditionalLognormalRegressor',FixedPoolLognormal):
                self.module.TemporalTransportPool(candidate()).fit(X.iloc[:3],[100.,80.,60.])
            attempts = [json.loads(v) for v in (root/'run/model-fit-attempts.jsonl').read_text().splitlines()]
            outcomes = [json.loads(v) for v in (root/'run/model-fit-outcomes.jsonl').read_text().splitlines()]
            self.assertEqual(len(attempts),4); self.assertEqual(len(outcomes),4)
            self.assertEqual([v['status'] for v in attempts],['started']*4)
            self.assertEqual([v['status'] for v in outcomes],['completed']*4)
            self.assertEqual(len({v['fingerprint'] for v in attempts}),4)
            self.assertEqual(FixedPoolBase.fit_calls,1)

    def test_invalid_targets_threads_and_prediction_schema_are_rejected(self):
        for threads in (True,0,5,1.5):
            with self.assertRaises(ValueError): self.module.TemporalTransportPool(candidate(),threads)
        model = self.module.TemporalTransportPool(candidate([0.,0.,1.]))
        with self.assertRaises(ValueError): model.predict(frame())
        for target in ([1.,2.], [1.,2.,np.nan,4.], [1.,-1.,2.,3.], ['a','b','a','b']):
            with self.assertRaises(ValueError): model.fit(frame(),target)
        with patch.object(self.module,'ConditionalLognormalRegressor',FixedPoolLognormal):
            model.fit(frame(),[1.,2.,3.,4.])
        with self.assertRaises(ValueError): model.predict(frame().drop(columns=['ELEVATION']))


if __name__ == '__main__':
    unittest.main()
