import unittest
import numpy as np
import pandas as pd
from sprint_cohort import CohortMacroDecision

class CohortTests(unittest.TestCase):
    def test_query_expected_objective_and_permutation(self):
        X=pd.DataFrame({'x':range(4)});p=np.array([[.9,.1],[.6,.4],[.4,.6],[.2,.8]])
        m=CohortMacroDecision(calibration_state={'theta':[0.,0.,0.]}).fit(X,['a','b','a','b'],p)
        prob=m.predict_proba(X,p);audit=m.last_prediction_audit_
        np.testing.assert_allclose(prob.sum(axis=1),1)
        self.assertGreaterEqual(audit['selected_expected_macro_f1'],audit['initial_expected_macro_f1'])
        permutation=[2,0,3,1]
        np.testing.assert_allclose(m.predict_proba(X.iloc[permutation],p[permutation]),prob[permutation])
        self.assertFalse(m.configuration()['query_labels_used'])
    def test_invalid_frozen_state_and_empty(self):
        X=pd.DataFrame({'x':[0,1]})
        with self.assertRaises(ValueError):CohortMacroDecision({'theta':[np.nan,0,0]}).fit(X,['a','b'],[[.7,.3],[.3,.7]])
