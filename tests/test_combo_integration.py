import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from automl_models import canonical_candidate,build_pipeline
from automl_history import candidate_key,compatible_policy,already_known
from solution import refit_candidate

class IntegrationTests(unittest.TestCase):
    def test_expert_composition_canonical_identity(self):
        c={'family':'extra_trees','parameters':{},'strategy':'reg_feature_experts','feature_numbers':[29,30,31]}
        c=canonical_candidate(c)
        self.assertEqual(c['preprocessing'],'reg-feature-experts-v1')
        self.assertTrue(compatible_policy(c))
        self.assertNotEqual(candidate_key(c),candidate_key({**c,'feature_numbers':[30]}))
        self.assertEqual(type(build_pipeline('regression',c)).__name__,'FeatureGroupExperts')
    def test_train_only_choices_frozen_without_mutating_attempt(self):
        c={'family':'xgboost','parameters':{},'strategy':'frequency_blend_macro_f1'};old=copy.deepcopy(c)
        got=refit_candidate(c,{'learned_state':{'class_multipliers':{'a':1.,'b':1.5}}})
        self.assertEqual(c,old);self.assertEqual(got['class_multipliers'],{'a':1.,'b':1.5})
        self.assertTrue(compatible_policy(got))
        self.assertNotEqual(candidate_key(got),candidate_key(c))
    def test_invalid_class_parameters_rejected(self):
        for p in [{'n_estimators':999999},{'device':'cuda'},{'arbitrary':1}]:
            with self.assertRaises(ValueError):canonical_candidate({'family':'xgboost','parameters':p,'strategy':'smote_macro_f1'})
    def test_regression_and_class_strategies_cannot_cross_task_kind(self):
        with self.assertRaises(ValueError):build_pipeline('classification',{'family':'extra_trees','parameters':{},'strategy':'reg_feature_experts'})
        with self.assertRaises(ValueError):build_pipeline('regression',{'family':'xgboost','parameters':{},'strategy':'probability_stacking'})
    def test_freeze_blend_and_stack_does_not_modify_evaluation_identity(self):
        for c,state in [({'family':'extra_trees','parameters':{},'strategy':'reg_convex_blend'},{'blend_weights':[.1,.2,.3,.4]}),
                        ({'family':'xgboost','parameters':{},'strategy':'probability_stacking'},{'meta_state':{'coef':[[1,0,0]],'intercept':[0],'classes':['a'],'component_order':['raw','smote','frequency'],'clip':1e-6}})]:
            before=copy.deepcopy(c);result=refit_candidate(c,{'learned_state':state})
            self.assertEqual(c,before)
            for key,value in state.items():np.testing.assert_equal(result[key],value)
            self.assertNotEqual(candidate_key(c),candidate_key(result))
    def test_irrelevant_fields_cannot_create_new_combination_identity(self):
        with self.assertRaises(ValueError):canonical_candidate({'family':'xgboost','strategy':'smote_macro_f1','meta_state':{}})
        with self.assertRaises(ValueError):canonical_candidate({'family':'extra_trees','strategy':'reg_feature_experts','shrinkage':True})
    def test_negative_combination_is_known_after_rename_and_coefficient_change(self):
        prior={'records':[],'spec008_attempts':[{'number':46,'candidate':{'family':'xgboost','parameters':{},'strategy':'smote_macro_f1'},'status':'completed'}]}
        c={'family':'xgboost','parameters':{'random_state':77},'strategy':'smote_macro_f1','hypothesis':'different name','class_multipliers':{'a':1.,'b':9.}}
        self.assertTrue(already_known(c,[],prior))
        self.assertFalse(already_known(c,[],None))
    def test_frozen_stack_coefficients_participate_identity(self):
        c={'family':'xgboost','parameters':{},'strategy':'probability_stacking','meta_state':{'coef':[[1,0,0]],'intercept':[0],'classes':['a'],'component_order':['raw','smote','frequency'],'clip':1e-6}}
        other=copy.deepcopy(c);other['meta_state']['coef']=[[2,0,0]]
        self.assertNotEqual(candidate_key(c),candidate_key(other))
        canonical_candidate(c)
        with self.assertRaises(ValueError):canonical_candidate({**c,'meta_state':{**c['meta_state'],'coef':[[1]]}})
    def test_worker_metrics_transport_learned_state_to_refit(self):
        import automl_worker
        from tests.test_platform_agent import PlatformContractTests
        class Fake:
            def fit(self,X,y):
                self.learned_state_={'class_multipliers':{'blue':1.,'red':1.5}}
                self.audit_={'fit_retained_rows':len(X)};self.keep_mask_=np.ones(len(X),bool);self.removed_positions_=np.array([],int)
                self.classes_=np.array(['blue','red']);return self
            def predict(self,X):return np.repeat('blue',len(X))
            def predict_proba(self,X):return np.tile([.6,.4],(len(X),1))
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as d:
            root=Path(d);task=PlatformContractTests().fixture(root,True);out=root/'run';out.mkdir()
            np.savez(root/'split.npz',fit=np.arange(30),valid=np.arange(30,40))
            c={'family':'xgboost','parameters':{},'strategy':'frequency_blend_macro_f1'}
            req={'task_path':str(task),'output':str(out),'candidate':c,'mode':'evaluate','split_path':str(root/'split.npz')}
            (root/'request.json').write_text(json.dumps(req),encoding='utf-8')
            with patch.object(automl_worker,'build_pipeline',return_value=Fake()):self.assertEqual(automl_worker.run(root/'request.json'),0)
            result=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(result['learned_state']['class_multipliers']['red'],1.5)
            self.assertEqual(refit_candidate(c,result)['class_multipliers']['red'],1.5)
    def test_frozen_equal_stack_fallback_never_retries_inner_selection(self):
        import pandas as pd
        from automl_class_combo import ClassCombination
        c=refit_candidate({'family':'xgboost','strategy':'probability_stacking','parameters':{'n_estimators':3}}, {'learned_state':{'meta_state':None}})
        X=pd.DataFrame({'x':np.arange(18.),'proto':['tcp','udp']*9,'service':['-']*18,'state':['FIN']*18})
        y=np.array(['a','b']*9)
        with patch('automl_class_combo.calibration_folds',side_effect=AssertionError('Frozen fallback retried OOF')):
            model=ClassCombination(c,threads=1).fit(X,y)
        self.assertIsNone(model.learned_state_['meta_state']);self.assertEqual(model.audit_['state_source'],'frozen_candidate')

if __name__=='__main__':unittest.main()
