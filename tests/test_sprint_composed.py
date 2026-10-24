import copy
import unittest
import numpy as np
import pandas as pd
from scipy.special import softmax
from sprint_composed import ComposedEmpiricalDecision

class ComposedTests(unittest.TestCase):
    def test_frozen_input_transform_once_and_state_copy(self):
        X=pd.DataFrame({'x':[1,2]});p=np.array([[.9,.1],[.1,.9]]);state={'classes':['a','b'],'gain':[1,1],'offset':[0,0],'theta':[0,0,0]}
        m=ComposedEmpiricalDecision(state).fit(X,['a','b'],p)
        np.testing.assert_allclose(m.predict_proba(X,p),softmax(softmax(p,axis=1),axis=1))
        state['gain'][0]=999
        self.assertEqual(m.state['gain'][0],1)
        self.assertEqual(m.audit_['initial_training_macro_f1'],1.)
        self.assertFalse(m.configuration()['input_state_refitted'])
    def test_invalid_state_class_alignment(self):
        X=pd.DataFrame({'x':[1,2]})
        with self.assertRaises(ValueError):ComposedEmpiricalDecision({'classes':['b','a'],'gain':[1,1],'offset':[0,0],'theta':[0,0,0]}).fit(X,['a','b'],[[.9,.1],[.1,.9]])
