import unittest
import numpy as np
import pandas as pd
from sprint_posterior import calibrated_ce_gradient, CalibratedMacroDecision


class PosteriorTests(unittest.TestCase):
    def test_calibration_gradient(self):
        logits=np.array([[-1.,-.2,.4],[.3,-.8,-.2],[.7,.2,-1.]])
        target=np.array([2,0,1]);theta=np.array([.2,.1,-.3,.2])
        loss,grad=calibrated_ce_gradient(theta,logits,target)
        numerical=[]
        for j in range(len(theta)):
            step=np.eye(len(theta))[j]*1e-6
            numerical.append((calibrated_ce_gradient(theta+step,logits,target)[0]-calibrated_ce_gradient(theta-step,logits,target)[0])/2e-6)
        np.testing.assert_allclose(grad,numerical,atol=1e-7)

    def test_named_class_order_batch_invariance_and_pickle(self):
        import pickle
        X=pd.DataFrame({'x':[0,1,2,3,4,5]});y=np.array(['a','b','a','b','a','b'])
        p=pd.DataFrame({'b':[.8,.9,.7,.8,.3,.7],'a':[.2,.1,.3,.2,.7,.3]})
        model=CalibratedMacroDecision().fit(X,y,p)
        whole=model.predict_proba(X,p)
        np.testing.assert_allclose(whole.sum(axis=1),1.)
        np.testing.assert_allclose(whole[:1],model.predict_proba(X.iloc[:1],p.iloc[:1]))
        np.testing.assert_allclose(whole,pickle.loads(pickle.dumps(model)).predict_proba(X,p))
        self.assertLessEqual(model.audit_['final_training_ce'],model.audit_['initial_training_ce']+1e-9)

    def test_invalid_teacher_rejected(self):
        with self.assertRaises(ValueError):
            CalibratedMacroDecision().fit(pd.DataFrame({'x':[1,2]}),['a','b'],[[1.,-1.],[.2,.8]])

if __name__=='__main__':unittest.main()
