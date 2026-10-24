"""H95 generalized-CE derivatives and train-only class balance contracts."""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax


class RobustTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_robust'), 'H95 robust objective absent')
        self.module = importlib.import_module('sprint_robust')

    def test_exact_weighted_gradient_matches_central_finite_differences_q07_and_ce_limit(self):
        z = np.array([[1.2, -.4, .1], [-.2, .9, .3], [.1, -.7, 1.4]])
        y, weight = np.array([0, 2, 1]), np.array([.5, 2., .8])
        for q in (0., .7):
            loss, analytical = self.module.generalized_ce_loss_and_gradient(z, y, q=q, weights=weight)
            numerical = np.zeros_like(z)
            for i, j in np.ndindex(z.shape):
                a, b = z.copy(), z.copy()
                a[i,j] += 1e-6; b[i,j] -= 1e-6
                numerical[i,j] = (self.module.generalized_ce_loss_and_gradient(a,y,q=q,weights=weight)[0]
                    - self.module.generalized_ce_loss_and_gradient(b,y,q=q,weights=weight)[0]) / 2e-6
            self.assertGreater(loss, 0.)
            np.testing.assert_allclose(analytical,numerical,atol=2e-9,rtol=1e-6)
            np.testing.assert_allclose(analytical.sum(1),0.,atol=1e-15)

    def test_ce_limit_and_manual_weighted_gce_value(self):
        z, y = np.log([[.8,.2],[.3,.7]]), np.array([0,1])
        loss, grad = self.module.generalized_ce_loss_and_gradient(z,y,q=0,weights=[2.,.5])
        self.assertAlmostEqual(loss,(-2*np.log(.8)-.5*np.log(.7))/2)
        np.testing.assert_allclose(grad,[[-.2,.2],[.075,-.075]])
        loss, _ = self.module.generalized_ce_loss_and_gradient(z,y,q=.7,weights=[2.,.5])
        self.assertAlmostEqual(loss,(2*(1-.8**.7)+.5*(1-.7**.7))/1.4)

    def test_improbable_label_gradient_is_suppressed_and_extremes_stay_finite(self):
        z, y = np.log([[.01,.99],[.9,.1]]), np.array([0,0])
        _, gce = self.module.generalized_ce_loss_and_gradient(z,y)
        _, ce = self.module.generalized_ce_loss_and_gradient(z,y,q=0)
        np.testing.assert_allclose(gce[:,0]/ce[:,0],[.01**.7,.9**.7])
        loss, grad = self.module.generalized_ce_loss_and_gradient([[1000.,-1000.],[-1000.,1000.]],y)
        self.assertTrue(np.isfinite(grad).all())
        self.assertAlmostEqual(loss,1/1.4)
        np.testing.assert_allclose(grad,0.,atol=1e-300)
        for weight in ([1.],[-1.,1.],[np.nan,1.]):
            with self.assertRaises(ValueError): self.module.generalized_ce_loss_and_gradient(z,y,weights=weight)

    def test_fit_class_weights_are_inverse_sqrt_counts_normalized_mean_one(self):
        X = pd.DataFrame({'x':np.arange(5)})
        model = self.module.RobustTeacherBoost(iterations=0).fit(X,['a']*4+['z'],np.full((5,2),.5))
        np.testing.assert_allclose(model.class_weights_,[5/6,5/3])
        self.assertAlmostEqual(model.audit_['sample_weight_mean'],1.)
        np.testing.assert_array_equal(model.audit_['class_counts'],[4,1])
        logits = np.log(np.tile([.4,.6],(5,1)))
        grad, hess = model._objective(np.array([0,0,0,0,1]),logits)
        weights = np.array([5/6]*4+[5/3])
        expected = self.module.generalized_ce_loss_and_gradient(logits,np.array([0,0,0,0,1]),weights=weights)[1]*5
        np.testing.assert_allclose(grad,expected)
        np.testing.assert_allclose(hess,np.maximum(weights[:,None]*.24,.001)*np.ones((5,2)))
        p=np.array([[.2,.8],[.9,.1]])
        np.testing.assert_allclose(model.predict_proba(X.iloc[:2],p),p)

    def test_query_teacher_added_once_and_classifier_pickles(self):
        X=pd.DataFrame({'x':np.linspace(-2,2,160)})
        y=np.where(X.x>0,'z','a');p=np.tile([.6,.4],(160,1))
        model=self.module.RobustTeacherBoost(iterations=30,learning_rate=.15,num_leaves=7,min_child_samples=5).fit(X,y,p)
        q=pd.DataFrame({'x':[-1.,1.]});given=np.tile([.4,.6],(2,1))
        actual=model.predict_proba(q,given)
        raw=model.model_.predict(model.encoder_.transform(q),raw_score=True)
        np.testing.assert_allclose(actual,softmax(raw+np.log(given),axis=1),atol=1e-12)
        np.testing.assert_allclose(model.predict_proba(q.iloc[:1],given[:1]),actual[:1])
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict_proba(q,given),actual)
        self.assertEqual(model.predict(q,given).tolist(),['a','z'])
        with self.assertRaises(ValueError):self.module.RobustTeacherBoost(iterations=0).fit(X,y,None)


if __name__=='__main__':unittest.main()
