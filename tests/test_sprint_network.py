"""H82 deterministic network semantics and explicit-teacher contracts."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_network'), 'H82 network module missing')
        self.module = importlib.import_module('sprint_network')

    def test_known_ttl_bins_hop_fractions_and_missing_semantics(self):
        X = pd.DataFrame({'sttl': [31., 32., 33., 63., 64., 65., 128., 129., 255., 0., np.nan, np.inf, 256., -1.],
                          'dtcpb': np.arange(14), 'stcpb': np.arange(14), 'proto': ['tcp'] * 14})
        original = X.copy(deep=True)
        output = self.module.NetworkSemanticFeatures().fit(X).transform(X)
        np.testing.assert_allclose(output.network__sttl_initial.iloc[:9], [32, 32, 64, 64, 64, 128, 128, 255, 255])
        np.testing.assert_allclose(output.network__sttl_hop_fraction.iloc[:9],
            [1 / 32, 0, 31 / 64, 1 / 64, 0, 63 / 128, 0, 126 / 255, 0])
        self.assertTrue(output.network__sttl_initial.iloc[9:].isna().all())
        self.assertTrue(output.network__sttl_hop_fraction.iloc[9:].isna().all())
        self.assertNotIn('stcpb', output)
        self.assertNotIn('dtcpb', output)
        pd.testing.assert_frame_equal(X, original)
        pd.testing.assert_series_equal(output.proto, X.proto)

    def test_directional_imbalances_retransmission_and_rtt_have_hand_checked_values(self):
        X = pd.DataFrame({'spkts': [3., 0., 1.], 'dpkts': [1., 0., np.inf],
            'sbytes': [0., 10., np.nan], 'dbytes': [4., 0., 3.],
            'sload': [9., 2., 0.], 'dload': [0., 2., 0.],
            'sloss': [1., 0., 0.], 'dloss': [2., 0., 0.],
            'tcprtt': [.5, 0., np.inf], 'synack': [.2, 0., .1], 'ackdat': [.1, 0., .1]})
        out = self.module.NetworkSemanticFeatures().fit(X).transform(X)
        np.testing.assert_allclose(out.network__packet_imbalance[:2], [.4, 0.])
        np.testing.assert_allclose(out.network__byte_imbalance[:2], [-.8, 10 / 11])
        np.testing.assert_allclose(out.network__load_imbalance, [.9, 0., 0.])
        np.testing.assert_allclose(out.network__source_loss_fraction, [.25, 0., 0.])
        np.testing.assert_allclose(out.network__destination_loss_fraction[:2], [1., 0.])
        np.testing.assert_allclose(out.network__rtt_residual[:2], [.2, 0.])
        self.assertTrue(np.isnan(out.network__packet_imbalance.iloc[2]))
        self.assertTrue(np.isnan(out.network__rtt_residual.iloc[2]))
        self.assertTrue(np.isnan(out.dpkts.iloc[2]))
        self.assertFalse(any('is_zero' in c or 'applicable' in c for c in out))

    def test_transform_has_no_query_cohort_statistics_and_keeps_index(self):
        X = pd.DataFrame({'sttl': [31., 64.], 'spkts': [1., 9.], 'dpkts': [0., 2.]}, index=[21, 7])
        features = self.module.NetworkSemanticFeatures().fit(X)
        before = pickle.dumps(features)
        all_rows = features.transform(X)
        pd.testing.assert_frame_equal(features.transform(X.iloc[:1]), all_rows.iloc[:1])
        pd.testing.assert_frame_equal(pickle.loads(before).transform(X), all_rows)
        self.assertEqual(pickle.dumps(features), before)
        self.assertEqual(all_rows.index.tolist(), [21, 7])
        with self.assertRaises(ValueError):
            features.transform(X.rename(columns={'spkts': 'other'}))
        with self.assertRaises(ValueError):
            self.module.NetworkSemanticFeatures().fit(pd.DataFrame({'network__sttl_initial': [1.]}))

    def test_wrapper_real_context_learner_preserves_teacher_columns_and_pickle(self):
        rng = np.random.default_rng(82)
        X = pd.DataFrame({'signal': rng.uniform(-2, 2, 120), 'sttl': [63.] * 120,
                          'stcpb': np.arange(120), 'dtcpb': np.arange(120)[::-1], 'proto': ['tcp'] * 120})
        y = np.where(X.signal > 0, 'z', 'a')
        p = np.tile([.6, .4], (120, 1))
        model = self.module.NetworkSemanticTeacherCorrector().fit(X, y, p)
        q = pd.DataFrame({'signal': [-1., 1.], 'sttl': [63., 63.],
                          'stcpb': [123., 999.], 'dtcpb': [24., 19.], 'proto': ['unknown', None]})
        given = np.tile([.4, .6], (2, 1))
        probability = model.predict_proba(q, given)
        self.assertEqual(model.predict(q, given).tolist(), ['a', 'z'])
        np.testing.assert_allclose(probability.sum(1), 1.)
        np.testing.assert_allclose(probability[:1], model.predict_proba(q.iloc[:1], given[:1]))
        q2 = q.copy()
        q2[['stcpb', 'dtcpb']] = [[999999., -999999.], [0., 0.]]
        np.testing.assert_allclose(model.predict_proba(q2, given), probability)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict_proba(q, given), probability)
        self.assertEqual(model.audit_['fit_rows'], 120)
        self.assertEqual(model.audit_['removed_rows'], 0)
        self.assertEqual(model.model_.parameters['iterations'], 400)
        self.assertEqual(model.model_.parameters['depth'], 5)
        with self.assertRaises(ValueError):
            model.predict_proba(q, None)


if __name__ == '__main__':
    unittest.main()
