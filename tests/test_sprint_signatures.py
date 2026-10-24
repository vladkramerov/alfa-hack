"""Hand-derived synthetic checks for fixed H84 signature posterior pooling."""
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd


AVAILABLE = importlib.util.find_spec('sprint_signatures') is not None
if AVAILABLE:
    from sprint_signatures import SignaturePosteriorClassifier


def rows(size=16):
    return pd.DataFrame({'proto': ['tcp'] * size, 'service': ['http'] * size,
        'state': ['FIN'] * size, 'sttl': [64.] * size, 'dttl': [32.] * size,
        'spkts': [3.] * size, 'dpkts': [1.] * size, 'sbytes': [127.] * size,
        'dbytes': [31.] * size})


class SignaturePosteriorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'H84 implementation is not present yet')

    def fit(self, X=None, labels=None, teacher=None):
        X = rows() if X is None else X
        labels = np.array(['a'] * 12 + ['b'] * 4) if labels is None else np.asarray(labels)
        teacher = np.tile([.2, .8], (len(X), 1)) if teacher is None else teacher
        return SignaturePosteriorClassifier().fit(X, labels, teacher_probabilities=teacher)

    def test_manual_posterior_counts_and_mixture(self):
        X = rows()
        X.loc[8:, 'proto'] = 'udp'
        labels = np.array(['a'] * 8 + ['a'] * 4 + ['b'] * 4)
        model = self.fit(X, labels)
        # Prior=(3/4,1/4), tcp n=8 counts=(8,0), q=(23/28,5/28), alpha=2/29.
        # (.2,.8)*(27/29) + (23/28,5/28)*(2/29) = (493/2030,1537/2030).
        p = model.predict_proba(X.iloc[:1], teacher_probabilities=np.array([[.2, .8]]))
        np.testing.assert_allclose(p, [[493 / 2030, 1537 / 2030]], rtol=1e-14, atol=1e-14)

    def test_finest_supported_level_wins_not_broadest(self):
        X = rows(32)
        X.loc[8:, 'sbytes'] = 255.
        labels = np.array(['a'] * 8 + ['b'] * 24)
        model = self.fit(X, labels)
        # Finest cell n=8,a=8; prior=(1/4,3/4), q=(13/28,15/28), alpha=2/29.
        p = model.predict_proba(X.iloc[:1], teacher_probabilities=np.array([[.5, .5]]))
        np.testing.assert_allclose(p, [[101 / 203, 102 / 203]], rtol=1e-14, atol=1e-14)

    def test_backoff_from_insufficient_bytes_cell(self):
        X = rows()
        X.loc[4:, 'sbytes'] = 255.
        model = self.fit(X)
        # Four-row finest cell is unsupported; packet-level cell has all16 rows.
        # It has q=prior=(3/4,1/4), alpha=4/33.
        p = model.predict_proba(X.iloc[:1], teacher_probabilities=np.array([[.2, .8]]))
        np.testing.assert_allclose(p, [[4 / 15, 11 / 15]], rtol=1e-14, atol=1e-14)

    def test_unseen_category_and_entirely_unsupported_return_teacher_exactly(self):
        teacher = np.array([[.13, .87]])
        model = self.fit()
        query = rows(1)
        query['proto'] = 'new-proto'
        np.testing.assert_array_equal(model.predict_proba(query, teacher_probabilities=teacher), teacher)
        tiny = self.fit(rows(4), ['a', 'a', 'b', 'b'])
        np.testing.assert_array_equal(tiny.predict_proba(rows(1), teacher_probabilities=teacher), teacher)

    def test_numeric_bins_match_without_exact_values(self):
        model = self.fit()
        query = rows(1)
        query.loc[0, ['sttl', 'dttl', 'spkts', 'dpkts', 'sbytes', 'dbytes']] = [95., 63., 6., 2., 254., 62.]
        teacher = np.array([[.2, .8]])
        np.testing.assert_allclose(model.predict_proba(query, teacher_probabilities=teacher), [[4 / 15, 11 / 15]])

    def test_missing_numeric_token_does_not_match_zero(self):
        X = rows(32)
        X.loc[:15, 'sttl'] = np.nan
        X.loc[16:, 'sttl'] = 0.
        labels = np.array(['a'] * 16 + ['b'] * 16)
        model = self.fit(X, labels)
        query = X.iloc[[0, 16]]
        p = model.predict_proba(query, teacher_probabilities=np.full((2, 2), .5))
        self.assertGreater(p[0, 0], .5)
        self.assertLess(p[1, 0], .5)

    def test_missing_categorical_tag_cannot_collide_with_real_string(self):
        X = rows(32)
        X.loc[:15, 'proto'] = None
        X.loc[16:, 'proto'] = '__MISSING__'
        labels = np.array(['a'] * 16 + ['b'] * 16)
        model = self.fit(X, labels)
        p = model.predict_proba(X.iloc[[0, 16]], teacher_probabilities=np.full((2, 2), .5))
        self.assertGreater(p[0, 0], .5)
        self.assertLess(p[1, 0], .5)

    def test_fit_teacher_values_do_not_change_table_or_predictions(self):
        first = self.fit(teacher=np.tile([.99, .01], (16, 1)))
        second = self.fit(teacher=np.tile([.01, .99], (16, 1)))
        query, teacher = rows(2), np.array([[.4, .6], [.6, .4]])
        np.testing.assert_array_equal(first.predict_proba(query, teacher_probabilities=teacher),
                                      second.predict_proba(query, teacher_probabilities=teacher))

    def test_id_and_unselected_columns_have_no_effect(self):
        X = rows()
        X['id'], X['target'], X['arbitrary'] = np.arange(16), 'secret', np.arange(16) ** 2
        model = self.fit(X)
        query = X.iloc[:2].copy()
        query[['id', 'target', 'arbitrary']] = -100
        teacher = np.tile([.4, .6], (2, 1))
        np.testing.assert_array_equal(model.predict_proba(query, teacher_probabilities=teacher),
                                      self.fit().predict_proba(rows(2), teacher_probabilities=teacher))

    def test_batch_order_pickle_and_no_query_state_mutation(self):
        model = self.fit()
        query = rows(3)
        query.loc[1, 'proto'] = 'unseen'
        teacher = np.array([[.1, .9], [.3, .7], [.6, .4]])
        state = pickle.dumps(model)
        full = model.predict_proba(query, teacher_probabilities=teacher)
        for i in range(3):
            np.testing.assert_array_equal(full[i], model.predict_proba(query.iloc[[i]], teacher_probabilities=teacher[[i]])[0])
        np.testing.assert_array_equal(model.predict_proba(query.iloc[::-1], teacher_probabilities=teacher[::-1]), full[::-1])
        np.testing.assert_array_equal(pickle.loads(state).predict_proba(query, teacher_probabilities=teacher), full)
        self.assertEqual(pickle.dumps(model), state)

    def test_sorted_lexical_classes_preserve_leading_zero(self):
        model = self.fit(labels=['1'] * 8 + ['01'] * 8)
        self.assertEqual(model.classes_.tolist(), ['01', '1'])
        named = pd.DataFrame([[.2, .8]], columns=['01', '1'])
        self.assertEqual(model.predict(rows(1), teacher_probabilities=named).tolist(), ['1'])
        with self.assertRaises(ValueError):
            model.predict_proba(rows(1), teacher_probabilities=named[['1', '01']])

    def test_invalid_teacher_shape_values_or_named_order_rejected(self):
        for teacher in [np.ones((15, 2)), np.full((16, 2), np.nan), np.tile([-.1, 1.1], (16, 1)),
                        np.zeros((16, 2)), np.tile([.2, .2], (16, 1)), None]:
            with self.subTest(teacher=type(teacher)), self.assertRaises(ValueError):
                self.fit(teacher=teacher if teacher is not None else np.array([[1.]]))
        named = pd.DataFrame(np.full((16, 2), .5), columns=['b', 'a'])
        with self.assertRaises(ValueError):
            self.fit(teacher=named)

    def test_missing_features_or_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            self.fit(X=rows().drop(columns=['dttl']))
        for labels in [['a'] * 15, ['a'] * 15 + [None], np.arange(16)]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.fit(labels=labels)

    def test_empty_query_preserves_shape(self):
        model = self.fit()
        self.assertEqual(model.predict_proba(rows(0), teacher_probabilities=np.empty((0, 2))).shape, (0, 2))
        self.assertEqual(model.predict(rows(0), teacher_probabilities=np.empty((0, 2))).shape, (0,))

    def test_probabilities_are_finite_normalized_and_configuration_audited(self):
        model = self.fit()
        p = model.predict_proba(rows(2), teacher_probabilities=np.array([[1., 0.], [0., 1.]]))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1.)
        self.assertEqual(model.audit_['configuration'], model.configuration())
        self.assertEqual(model.audit_['fit_rows'], 16)
        self.assertEqual(model.audit_['removed_rows'], 0)
        self.assertFalse(model.audit_['teacher_training_values_used'])


if __name__ == '__main__':
    unittest.main()
