import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor


class FullPolicyTests(unittest.TestCase):
    def api(self): return importlib.import_module('automl_models')

    def test_mandatory_policy_cannot_be_disabled_by_candidate(self):
        api=self.api()
        candidate=api.canonical_candidate({'family':'extra_trees','parameters':{},'preprocessing':'legacy_standard','scale':False,'max_outlier_fraction':.9})
        self.assertEqual(candidate['preprocessing'],'full-v1')
        self.assertTrue(candidate['scale']); self.assertEqual(candidate['max_outlier_fraction'],.05)

    def test_policy_and_target_transform_are_part_of_experiment_identity(self):
        from automl_history import candidate_key
        base={'family':'extra_trees','parameters':{}}
        full={**base,'preprocessing':'full-v1','scale':True,'max_outlier_fraction':.05}
        self.assertNotEqual(candidate_key(base),candidate_key(full))
        self.assertNotEqual(candidate_key(full),candidate_key({**full,'target_transform':'log1p'}))

    def test_preparation_model_fits_transform_only_on_retained_train_and_inverts_target(self):
        api=self.api(); rng=np.random.default_rng(8)
        X=pd.DataFrame({'value':rng.lognormal(size=150),'cat':['a','b']*75,'constant':1.})
        y=np.arange(150,dtype=float)+1
        model=api.PreparedModel(DummyRegressor(strategy='mean'),kind='regression',target_transform='log1p').fit(X,y)
        keep=model.keep_mask_
        expected=np.expm1(np.log1p(y[keep]).mean())
        future=pd.DataFrame({'value':[1e10,np.nan],'cat':['new','a'],'constant':[1.,1.]},index=[81,82])
        np.testing.assert_allclose(model.predict(future),expected)
        self.assertEqual(len(model.predict(future)),2)
        self.assertEqual(model.audit_['policy'],'full-v1')
        self.assertLessEqual(np.count_nonzero(~keep),int(.05*len(X)))
        before=model.preprocessor_.scaler_.mean_.copy(); model.predict(future)
        np.testing.assert_array_equal(before,model.preprocessor_.scaler_.mean_)

    def test_all_supported_families_train_on_small_mixed_data(self):
        api=self.api(); signal=np.linspace(-2,2,90)
        X=pd.DataFrame({'signal':signal,'category':['a','b','c']*30})
        test=pd.DataFrame({'signal':[1e20,np.nan],'category':['new',None]})
        for family in ['xgboost','lightgbm','extra_trees','catboost','random_forest']:
            parameters={'iterations':5,'depth':2} if family=='catboost' else {'n_estimators':5,'max_depth':3}
            if family=='lightgbm': parameters['min_child_samples']=2
            for kind in ['regression','classification']:
                y=signal**2+1 if kind=='regression' else np.tile(['a','b','c'],30)
                with self.subTest(family=family,kind=kind):
                    model=api.build_pipeline(kind,{'family':family,'parameters':parameters}).fit(X,y)
                    pred=model.predict(test); self.assertEqual(len(pred),2)
                    self.assertEqual(model.audit_['policy'],'full-v1')
                    if kind=='classification':
                        self.assertTrue(set(pred)<=set(y)); self.assertEqual(model.predict_proba(test).shape,(2,3))


if __name__=='__main__': unittest.main()
