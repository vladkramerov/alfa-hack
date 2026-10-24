"""Derivative and integration checks using only synthetic data."""
import unittest, io
import numpy as np, pandas as pd, joblib
from scipy.optimize._numdiff import approx_derivative
from focal_boosting import FocalBoostingClassifier, FocalObjective, focal_loss, focal_derivatives


class FocalTests(unittest.TestCase):
    def test_exact_gradient_against_finite_differences(self):
        z=np.array([[.5,-.7,1.1],[-3.,1.,.4],[6.,-1.,-5.]])
        y=np.array([2,0,1]);alpha=np.array([.3,1.7,2.1])
        analytic,h=focal_derivatives(z,y,alpha)
        numeric=approx_derivative(lambda flat:focal_loss(flat.reshape(z.shape),y,alpha),
                                  z.ravel(),method='3-point').reshape(z.shape)
        np.testing.assert_allclose(analytic,numeric,rtol=3e-6,atol=1e-8)
        self.assertTrue(np.isfinite(h).all() and (h>=1e-6).all())
        np.testing.assert_allclose(analytic.sum(1),0,atol=1e-14)

    def test_extreme_logits_have_finite_derivatives_and_loss(self):
        z=np.array([[1000.,-1000.,0.],[-1000.,1000.,0.],[0.,0.,0.]])
        g,h=focal_derivatives(z,[0,0,2],[1,2,3])
        self.assertTrue(np.isfinite(g).all() and np.isfinite(h).all() and (h>0).all())
        self.assertTrue(np.isfinite(focal_loss(z,[0,0,2],[1,2,3])))

    def test_objective_callback_binary_gradient_and_invalid_labels(self):
        objective=FocalObjective([.5,2.])
        y=np.array([0,1]);z=np.array([.3,-.7]);g,h=objective(y,z)
        finite=approx_derivative(lambda zz:focal_loss(np.column_stack([np.zeros(2),zz]),y,[.5,2.]),z)
        np.testing.assert_allclose(g,finite.ravel(),rtol=1e-6,atol=1e-8)
        self.assertTrue((h>0).all())
        with self.assertRaises(ValueError):focal_derivatives(np.zeros((2,3)),[0,3],[1,1,1])

    def test_multiclass_wrapper_softprob_codec_and_serialization(self):
        rng=np.random.RandomState(37)
        y=np.array(['009','NA','normal']*30)
        X=pd.DataFrame({'n':np.tile([-2.,0.,2.],30)+rng.normal(0,.05,90),
                        'c':np.tile(['left','center','right'],30)})
        model=FocalBoostingClassifier().fit(X,y,estimators=12)
        self.assertEqual(model.audit_['prediction_objective'],'multi:softprob')
        self.assertEqual(model.audit_['objective_calls'],12)
        self.assertAlmostEqual(model.audit_['mean_sample_alpha'],1.)
        p=model.predict_proba(X)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-8)
        self.assertGreater(np.mean(model.predict(X)==y),.95)
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
        restored=joblib.load(stream)
        np.testing.assert_array_equal(restored.classes_,model.classes_)
        np.testing.assert_allclose(restored.predict_proba(X),p,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)
