import unittest
from automl_models import canonical_candidate,build_pipeline
from automl_history import compatible_policy,already_known
class TemporalIntegrationTests(unittest.TestCase):
    def test_versioned_temporal_profile_model_factory(self):
        c=canonical_candidate({'family':'extra_trees','parameters':{},'strategy':'temporal_profile_memory'})
        self.assertEqual(c['preprocessing'],'temporal-profile-memory-v1');self.assertTrue(compatible_policy(c))
        self.assertEqual(type(build_pipeline('regression',c)).__name__,'TemporalProfileModel')
        with self.assertRaises(ValueError):build_pipeline('classification',c)
    def test_previous_alternative_failures_are_not_retried(self):
        prior={'records':[],'spec009_attempts':[{'number':51,'candidate':{'family':'xgboost','parameters':{},'strategy':'balanced_softmax'},'status':'completed'}]}
        self.assertTrue(already_known({'family':'xgboost','parameters':{},'strategy':'balanced_softmax','hypothesis':'renamed'},[],prior))
        self.assertTrue(already_known({'family':'xgboost','parameters':{},'hypothesis':'HYP-051 renamed'},[],prior))
    def test_generic_temporal_candidate_requires_full_profile_and_year(self):
        import pandas as pd
        import numpy as np
        import solution
        X=pd.DataFrame({'State_Factor':['s']*1000,'building_class':['a']*500+['b']*500,'facility_type':['f']*1000,
                        'floor_area':[100.]*1000,'year_built':[2000.]*1000,'ELEVATION':[1.]*1000,'Year_Factor':[1,2]*500})
        jobs=solution.portfolio_candidates('regression','rmse',X,np.ones(1000))
        self.assertEqual(jobs[0].get('strategy'),'temporal_profile_memory')
        other=solution.portfolio_candidates('regression','rmse',X.drop(columns='ELEVATION'),np.ones(1000))
        self.assertFalse(any(j.get('strategy')=='temporal_profile_memory' for j in other))
