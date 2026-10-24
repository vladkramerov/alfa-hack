import unittest
import numpy as np
from scipy.special import softmax
from sprint_softboost import MacroTeacherBoost
from sprint_classification import soft_macro_f1_loss_and_gradient

class SoftBoostTests(unittest.TestCase):
    def test_boosting_gradient_equals_exact_coupled_objective(self):
        model=MacroTeacherBoost();model.loss_history_=[]
        z=np.array([[.1,.4,-.2],[.3,-.2,.6],[.1,-.4,.6]]);y=np.array([0,1,2])
        gradient,hessian=model._objective(y,z)
        loss,expected=soft_macro_f1_loss_and_gradient(z,y)
        np.testing.assert_allclose(gradient,len(y)*expected)
        self.assertEqual(model.loss_history_,[loss]);self.assertTrue(np.all(hessian>0))
        np.testing.assert_allclose(hessian,np.maximum(.5*softmax(z,axis=1)*(1-softmax(z,axis=1)),.001))
    def test_configuration_reports_surrogate_and_distinct_method(self):
        model=MacroTeacherBoost()
        self.assertEqual(model.configuration()['parameters']['n_estimators'],300)
        self.assertEqual(model.configuration()['family'],'teacher_residual_soft_macro_f1_lightgbm')
        self.assertIn('not_exact',model.configuration()['curvature'])
