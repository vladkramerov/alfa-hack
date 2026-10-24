import pickle
import unittest
import numpy as np
import pandas as pd
import torch


class TorchTabMTests(unittest.TestCase):
    def test_pwl_is_fit_only_and_handles_constants_and_extremes(self):
        from automl_torch_tabm import PiecewiseEncoder
        X=np.array([[0.,3.],[1.,3.],[2.,3.],[3.,3.]])
        enc=PiecewiseEncoder(bins=2).fit(X)
        frozen=pickle.dumps(enc)
        z=enc.transform(np.array([[1.5,3.],[-100.,3.],[100.,3.]]))
        self.assertEqual(z.shape,(3,2,2))
        np.testing.assert_allclose(z[0,0],[1.,0.])
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(pickle.dumps(enc),frozen)

    def test_full_ensemble_layer_matches_hand_computation_and_backprop(self):
        from automl_torch_tabm import EnsembleLinear
        layer=EnsembleLinear(2,1,2)
        with torch.no_grad():
            layer.weight.fill_(1.)
            layer.r.copy_(torch.tensor([[1.,2.],[3.,4.]]))
            layer.s.copy_(torch.tensor([[2.],[3.]]))
            layer.bias.copy_(torch.tensor([[1.],[2.]]))
        x=torch.tensor([[[1.,1.],[1.,1.]]],requires_grad=True)
        value=layer(x)
        torch.testing.assert_close(value,torch.tensor([[[7.],[23.]]]))
        value.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_regression_learns_without_query_fit_and_survives_pickle(self):
        from automl_torch_tabm import TorchTabM
        x=np.linspace(-2,2,100)
        X=pd.DataFrame({'x':x,'kind':np.where(x>0,'p','n')})
        y=10+3*x
        model=TorchTabM(epochs=55,width=16,members=3,batch_size=16).fit(X,y)
        p=model.predict(X)
        self.assertLess(np.mean((p-y)**2),np.var(y)*.25)
        q=X.iloc[[4,2]].copy();q.loc[:,'kind']='unseen'
        self.assertTrue(np.isfinite(model.predict(q)).all())
        frozen=pickle.dumps(model.encoder_)
        model.predict(X.iloc[[8,0]])
        self.assertEqual(pickle.dumps(model.encoder_),frozen)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(X),p,atol=1e-6)

    def test_classification_preserves_labels_and_is_batch_invariant(self):
        from automl_torch_tabm import TorchTabM
        x=np.linspace(-2,2,120)
        X=pd.DataFrame({'x':x,'all_missing':np.nan})
        y=np.where(x>0,'rare-class','normal')
        model=TorchTabM(kind='classification',epochs=50,width=16,members=3,batch_size=16).fit(X,y)
        p=model.predict_proba(X)
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-6)
        self.assertGreater(np.mean(model.predict(X)==y),.95)
        np.testing.assert_allclose(model.predict_proba(X.iloc[[17,4]]),p[[17,4]],atol=1e-6)
        np.testing.assert_array_equal(model.classes_,['normal','rare-class'])
        with self.assertRaises(ValueError):model.predict(X.drop(columns='x'))

    def test_invalid_config_or_target_rejected(self):
        from automl_torch_tabm import TorchTabM
        X=pd.DataFrame({'x':[0.,1.,2.,3.]})
        for kwargs in ({'epochs':0},{'members':0},{'width':2000},{'seconds':0}):
            with self.assertRaises(ValueError):TorchTabM(**kwargs).fit(X,np.arange(4.))
        with self.assertRaises(ValueError):TorchTabM().fit(X,[0.,1.,np.inf,3.])
        with self.assertRaises(ValueError):TorchTabM(kind='classification').fit(X,['a']*4)


if __name__=='__main__':unittest.main()
