"""Synthetic contracts for optional model-specific preprocessing and feedback."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from automl_models import canonical_candidate,build_pipeline,PreparedModel,LightPreprocessor
from automl_history import candidate_key,compatible_policy
from automl_worker import error_diagnostics
from automl_llm import research_context


class GenericPolicyTests(unittest.TestCase):
    def test_default_contract_and_explicit_policy_identities(self):
        base={'family':'extra_trees','parameters':{}}
        old=canonical_candidate(base)
        light=canonical_candidate({**base,'preprocessing':'light','scale':True,'max_outlier_fraction':.9})
        self.assertEqual(old['preprocessing'],'full-v1')
        self.assertEqual(light['preprocessing'],'light-v1')
        self.assertFalse(light['scale']);self.assertEqual(light['max_outlier_fraction'],0)
        self.assertNotEqual(candidate_key(old),candidate_key(light))
        self.assertEqual(candidate_key(light),candidate_key(canonical_candidate({**base,'preprocessing':'light-v1'})))
        self.assertTrue(compatible_policy(light))
        self.assertFalse(compatible_policy({**light,'max_outlier_fraction':.05}))
        self.assertFalse(compatible_policy({**light,'scale':True}))
        for bad in ['raw-ish','native-v2','',42]:
            if bad=='':continue
            with self.subTest(bad=bad),self.assertRaises(ValueError):canonical_candidate({**base,'preprocessing':bad})
        with self.assertRaises(ValueError):canonical_candidate({**base,'preprocessing':'native-v1'})
        native=canonical_candidate({'family':'catboost','preprocessing':'native'})
        self.assertTrue(compatible_policy(native))
        self.assertNotEqual(candidate_key(native),candidate_key(canonical_candidate({'family':'catboost','preprocessing':'light-v1'})))

    def test_light_retains_all_rows_and_statistics_fit_only(self):
        X=pd.DataFrame({'x':[0.,2.,np.nan,100000.],'cat':['a',None,'__MISSING__','b']},index=[8,9,10,11])
        original=X.copy(deep=True);y=np.array([1.,2.,3.,8.])
        with patch('automl_preprocessing.TrainingOutlierFilter.fit_filter',side_effect=AssertionError('removed rows')):
            model=PreparedModel(DummyRegressor(),preprocessing='light-v1').fit(X,y)
        self.assertEqual(model.keep_mask_.tolist(),[True]*4)
        self.assertEqual(model.audit_['fit_retained_rows'],4)
        np.testing.assert_allclose(model.predict(X),y.mean())
        before=model.preprocessor_.imputer_.statistics_.copy()
        future=pd.DataFrame({'x':[np.nan,np.inf,1e10],'cat':['new',None,'__MISSING__']})
        model.predict(future)
        np.testing.assert_array_equal(before,model.preprocessor_.imputer_.statistics_)
        self.assertEqual(before.tolist(),[2.])
        pd.testing.assert_frame_equal(X,original)
        values=model.preprocessor_._categories(X)['cat']
        self.assertEqual(values.iloc[1],'m:');self.assertEqual(values.iloc[2],'v:__MISSING__')

    def test_bounded_categories_empty_schema_and_dense_allocation(self):
        X=pd.DataFrame({'cat':[str(i) for i in range(160)]})
        pre=LightPreprocessor().fit(X)
        self.assertLessEqual(pre.transform(X).shape[1],64)
        self.assertEqual(len(pre.transform(pd.DataFrame({'cat':['new',None]}))),2)
        empty=pd.DataFrame(index=[1,2,3]);bias=LightPreprocessor().fit_transform(empty)
        np.testing.assert_array_equal(bias,np.zeros((3,1),dtype=np.float32))
        with patch('automl_preprocessing.MAX_DENSE_CELLS',2),self.assertRaises(MemoryError):LightPreprocessor().fit(X)
        self.assertEqual(pre.transform(X.iloc[:0]).shape[0],0)

    def test_each_generic_family_supports_light_without_filtering(self):
        X=pd.DataFrame({'value':np.arange(45,dtype=float),'cat':['a','b',None]*15})
        query=pd.DataFrame({'value':[np.nan,10.,np.inf],'cat':['new',None,'a']})
        for family in ['extra_trees','random_forest','xgboost','lightgbm','catboost']:
            p={'iterations':3,'depth':2} if family=='catboost' else {'n_estimators':3,'max_depth':3}
            if family=='lightgbm':p['min_child_samples']=2
            for kind in ['regression','classification']:
                with self.subTest(family=family,kind=kind):
                    y=np.arange(45,dtype=float)+1 if kind=='regression' else np.array(['low','high','low']*15)
                    model=build_pipeline(kind,{'family':family,'parameters':p,'preprocessing':'light-v1'},threads=1).fit(X,y)
                    self.assertEqual(model.audit_['policy'],'light-v1')
                    self.assertEqual(model.audit_['fit_retained_rows'],45)
                    self.assertEqual(len(model.predict(query)),3)
                    if kind=='classification':np.testing.assert_allclose(model.predict_proba(query).sum(axis=1),1.)

    def test_native_categories_missing_unknown_and_target_codec(self):
        X=pd.DataFrame({'value':np.arange(60,dtype=float),'cat':['NA',None,'__MISSING__']*20})
        query=pd.DataFrame({'value':[np.nan,5.,np.inf],'cat':['new',None,'__MISSING__']})
        for family in ['lightgbm','catboost']:
            p={'iterations':4,'depth':2} if family=='catboost' else {'n_estimators':4,'max_depth':3,'min_child_samples':2}
            for kind in ['regression','classification']:
                with self.subTest(family=family,kind=kind):
                    y=np.arange(60,dtype=float)+1 if kind=='regression' else np.array(['01','1']*30)
                    model=build_pipeline(kind,{'family':family,'parameters':p,'preprocessing':'native-v1'},threads=1).fit(X,y)
                    self.assertEqual(model.audit_['policy'],'native-v1')
                    self.assertEqual(len(model.predict(query)),3)
                    values=model.preprocessor_.transform(query)
                    if family=='catboost':
                        self.assertEqual(values['cat'].tolist(),['v:new','m:','v:__MISSING__'])
                        self.assertEqual(model.estimator_.get_cat_feature_indices(),[1])
                    if kind=='classification':
                        self.assertTrue(set(model.predict(query))<=set(y))
                        np.testing.assert_allclose(model.predict_proba(query).sum(axis=1),1.)

    def test_diagnostics_handchecked_class_errors_and_bounds(self):
        d=error_diagnostics('classification',['a','a','b','b'],['a','b','b','b'])
        self.assertEqual(d['top_confusions'],[{'actual':'a','predicted':'b','count':1}])
        a=next(item for item in d['per_class'] if item['class']=='a')
        self.assertEqual(a['support'],2);self.assertEqual(a['recall'],.5);self.assertEqual(a['precision'],1.)
        self.assertAlmostEqual(a['f1'],2/3)
        many=np.array([str(i) for i in range(100)])
        bounded=error_diagnostics('classification',many,many[::-1])
        self.assertEqual(len(bounded['per_class']),24);self.assertEqual(len(bounded['top_confusions']),12)
        self.assertLess(len(json.dumps(bounded)),10000)

    def test_regression_slices_train_quantiles_and_aggregate_only_feedback(self):
        X=pd.DataFrame({'Year_Factor':[3,3,4,4],'year_built':[2000]*4})
        d=error_diagnostics('regression',[1.,2.,3.,4.],[2.,2.,4.,2.],X,fit_target=np.arange(8.))
        self.assertAlmostEqual(d['overall']['rmse'],np.sqrt(1.5))
        self.assertAlmostEqual(d['overall']['mae'],1.)
        self.assertAlmostEqual(d['overall']['bias'],0.)
        self.assertEqual(sum(v['rows'] for v in d['target_buckets']),4)
        self.assertEqual(len(d['period_slices']),2)
        self.assertEqual(d['bucket_definition'],'quintiles_of_fit_targets')
        context=json.loads(research_context({'task':'minimal'},[{'status':'completed','score':.5,'diagnostics':{**d,'raw_targets':[1,2]}}]))
        received=context['observations'][0]['diagnostics']
        self.assertEqual(received['overall'],d['overall']);self.assertNotIn('raw_targets',received)
        self.assertNotIn('positions',json.dumps(received))


if __name__=='__main__':unittest.main()
