import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from automl_sergey_fusion import SergeyIvanFusion,canonical_fusion


class FakeParent:
    def __init__(self,value):self.value=value;self.audit_={'fake':True}
    def predict(self,X):return np.full(len(X),self.value)


class SergeyFusionTests(unittest.TestCase):
    def frame(self):
        years=np.repeat([1.,2.,3.],600)
        return pd.DataFrame({'Year_Factor':years,'State_Factor':'a','building_class':'b',
            'facility_type':'c','floor_area':10.,'year_built':2000.,'energy_star_rating':50.,'ELEVATION':10.})

    def test_no_embedded_weights_and_fixed_policy(self):
        base={'strategy':'sergey_ivan_fusion','family':'extra_trees'}
        self.assertEqual(canonical_fusion(base)['preprocessing'],'sergey-ivan-fusion-v1')
        for key in ['weights','pool_weights','meta_state']:
            with self.assertRaises(ValueError):canonical_fusion({**base,key:[.5,.5]})

    def test_weights_use_only_train_prefix_and_latest_train_labels(self):
        X=self.frame();calls=[]
        def fit_parents(model,frame,y,active=None):
            calls.append((frame.Year_Factor.to_numpy().copy(),np.asarray(y).copy(),active))
            return [FakeParent(v) for v in [0.,2.,4.,6.]]
        with patch.object(SergeyIvanFusion,'_fit_components',fit_parents):
            model=SergeyIvanFusion().fit(X,np.ones(len(X)))
        self.assertEqual(len(calls),2)
        self.assertTrue((calls[0][0]<3).all());self.assertEqual(len(calls[0][1]),1200)
        self.assertEqual(len(calls[1][1]),1800)
        self.assertFalse(model.audit_['embedded_learned_state'])
        self.assertTrue(model.audit_['training_only'])
        np.testing.assert_allclose(model.predict(X.iloc[:5]),np.ones(5),atol=1e-7)
        frozen=model.weights_.copy()
        model.predict(X.iloc[:10].assign(energy_star_rating=-100))
        np.testing.assert_array_equal(model.weights_,frozen)

    def test_small_train_uses_sergey_fallback(self):
        X=self.frame().iloc[:20]
        with patch.object(SergeyIvanFusion,'_fit_components',return_value=[FakeParent(2),None,None,None]) as mocked:
            model=SergeyIvanFusion().fit(X,np.ones(len(X)))
        self.assertEqual(mocked.call_count,1)
        np.testing.assert_array_equal(model.weights_,[1.,0.,0.,0.])
        np.testing.assert_array_equal(model.predict(X),np.full(len(X),2.))


if __name__=='__main__':unittest.main()
