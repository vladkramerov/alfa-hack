"""H98 exact empirical-F1 offset search; no learner fits or real data."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class EmpiricalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_empirical'), 'H98 exact-offset module absent')
        self.module=importlib.import_module('sprint_empirical')

    def test_coordinate_matches_independent_exhaustive_threshold_search(self):
        p=np.array([[.7,.2,.1],[.6,.3,.1],[.2,.7,.1],[.3,.5,.2],[.1,.2,.7],[.3,.3,.4]])
        y=np.array([0,1,1,2,2,0]); b=np.array([0.,.03,-.02])
        for c in range(3):
            score=p+b;score[:,c]=-np.inf
            r=score.max(1)-p[:,c];values=np.unique(r)
            offsets=[np.nextafter(values[0],-np.inf), *[(a+z)/2 for a,z in zip(values[:-1],values[1:])],np.nextafter(values[-1],np.inf),b[c]]
            best=-1.
            for value in offsets:
                proposed=b.copy();proposed[c]=value
                best=max(best,f1_score(y,np.argmax(p+proposed,axis=1),labels=np.arange(3),average='macro',zero_division=0))
            actual=self.module.best_class_offset(p,y,b,c)
            self.assertAlmostEqual(actual['score'],best)

    def test_identical_probability_ties_are_never_arbitrarily_split(self):
        p=np.tile([.6,.3,.1],(12,1));y=np.array([0,1,2]*4)
        model=self.module.EmpiricalMacroF1Offsets().fit(pd.DataFrame(index=range(12)),y,p)
        prediction=model.predict(pd.DataFrame(index=range(12)),p)
        self.assertEqual(len(np.unique(prediction)),1)
        self.assertGreaterEqual(model.audit_['selected_training_macro_f1'],model.audit_['initial_training_macro_f1'])

    def test_direct_true_label_f1_improves_on_small_hand_fixture(self):
        p=np.array([[.6,.4],[.55,.45],[.52,.48],[.8,.2]])
        y=np.array(['a','z','z','a']);X=pd.DataFrame({'x':range(4)})
        model=self.module.EmpiricalMacroF1Offsets().fit(X,y,p)
        self.assertEqual(model.predict(X,p).tolist(),['a','z','z','a'])
        self.assertAlmostEqual(model.audit_['selected_training_macro_f1'],1.)
        self.assertEqual(model.audit_['objective'],'actual_training_labels_macro_f1')

    def test_row_permutation_and_query_batch_do_not_change_frozen_predictions(self):
        rng=np.random.default_rng(98);p=rng.dirichlet(np.ones(3),size=36)
        y=np.array(['a']*6+['b']*12+['c']*18);X=pd.DataFrame({'x':range(36)})
        model=self.module.EmpiricalMacroF1Offsets().fit(X,y,p)
        order=rng.permutation(36)
        other=self.module.EmpiricalMacroF1Offsets().fit(X.iloc[order],y[order],p[order])
        np.testing.assert_allclose(model.offset_,other.offset_,atol=1e-15)
        np.testing.assert_array_equal(model.predict(X,p),other.predict(X,p))
        whole=model.predict_proba(X,p)
        np.testing.assert_allclose(whole[:1],model.predict_proba(X.iloc[:1],p[:1]))
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict_proba(X,p),whole)

    def test_class_order_is_rarest_first_and_every_accepted_step_nondecreasing(self):
        rng=np.random.default_rng(3);p=rng.dirichlet(np.ones(3),size=30)
        y=np.array(['a']*15+['b']*10+['c']*5)
        model=self.module.EmpiricalMacroF1Offsets().fit(pd.DataFrame(index=range(30)),y,p)
        self.assertEqual(model.audit_['class_visit_order'],['c','b','a'])
        history=model.audit_['training_macro_f1_history']
        self.assertTrue(all(b>a+1e-12 for a,b in zip(history[:-1],history[1:])))
        self.assertGreaterEqual(history[-1],history[0])
        self.assertLessEqual(model.audit_['passes_completed'],3)

    def test_current_state_is_retained_when_scores_tie(self):
        p=np.array([[.9,.1],[.1,.9]]);y=np.array([0,1]);b=np.zeros(2)
        actual=self.module.best_class_offset(p,y,b,0)
        self.assertEqual(actual['offset'],0.)
        self.assertAlmostEqual(actual['score'],1.)

    def test_rejects_invalid_probabilities_labels_and_named_class_order(self):
        X=pd.DataFrame({'x':[1,2]});y=np.array(['a','b'])
        for p in (None,[[np.nan,1],[1,0]],[[0,0],[.5,.5]],[[-1,2],[1,0]],np.ones((2,3)),
                  pd.DataFrame([[.5,.5],[.5,.5]],columns=['b','a'])):
            with self.assertRaises(ValueError):self.module.EmpiricalMacroF1Offsets().fit(X,y,p)
        with self.assertRaises(ValueError):self.module.EmpiricalMacroF1Offsets().fit(X,['a'],np.ones((2,2)))

    def test_scores_are_probability_offsets_not_log_probability_multipliers(self):
        model=self.module.EmpiricalMacroF1Offsets()
        p=np.array([[.2,.3,.5],[.7,.1,.2],[.4,.4,.2]])
        X=pd.DataFrame(index=range(3));model.fit(X,['a','b','c'],p)
        model.offset_=np.array([.1,-.2,.3])
        expected=p+model.offset_;expected=np.exp(expected-expected.max(1,keepdims=True));expected/=expected.sum(1,keepdims=True)
        np.testing.assert_allclose(model.predict_proba(X,p),expected)


if __name__=='__main__':unittest.main()
