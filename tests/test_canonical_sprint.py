import copy
import unittest
from automl_models import canonical_candidate, build_pipeline
from automl_history import candidate_key, compatible_policy, already_known
from tests.test_sprint_integration import decision_candidate

class CanonicalSprintTests(unittest.TestCase):
    def test_temporal_pool_factory_and_weight_identity(self):
        c=canonical_candidate({'family':'extra_trees','strategy':'temporal_transport_pool','pool_weights':[.4,.5,.1]})
        self.assertTrue(compatible_policy(c));self.assertEqual(c['target_transform'],'none')
        self.assertEqual(type(build_pipeline('regression',c)).__name__,'TemporalTransportPool')
        self.assertNotEqual(candidate_key(c),candidate_key({**c,'pool_weights':[.5,.4,.1]}))
        with self.assertRaises(ValueError):build_pipeline('classification',c)
    def test_fixed_regression_factory_and_policy(self):
        c=canonical_candidate({'family':'extra_trees','parameters':{},'strategy':'entity_linked_history'})
        self.assertEqual(c['preprocessing'],'entity-linked-history-v1');self.assertTrue(compatible_policy(c))
        self.assertEqual(type(build_pipeline('regression',c)).__name__,'SprintRegression')
        with self.assertRaises(ValueError):build_pipeline('classification',c)
        with self.assertRaises(ValueError):canonical_candidate({**c,'parameters':{'n_estimators':1}})

    def test_decision_state_is_part_of_identity(self):
        c=decision_candidate();c['strategy']='proper_macro_f1_stacking';c['decision_state']['theta']=[0.,0.,0.]
        normalized=canonical_candidate(c)
        self.assertTrue(compatible_policy(normalized))
        self.assertEqual(type(build_pipeline('classification',normalized)).__name__,'SprintDecision')
        changed=copy.deepcopy(normalized);changed['decision_state']['gain'][0]+=1
        self.assertNotEqual(candidate_key(normalized),candidate_key(changed))
        with self.assertRaises(ValueError):build_pipeline('regression',normalized)
        with self.assertRaises(ValueError):canonical_candidate({**c,'feature_numbers':[30]})

    def test_previous_sprint_attempts_block_renamed_replay(self):
        prior={'records':[],'spec011_attempts':[{'number':78,'candidate':{'strategy':'proper_macro_f1_stacking'}}]}
        self.assertTrue(already_known({'family':'xgboost','strategy':'proper_macro_f1_stacking'},[],prior))
        self.assertTrue(already_known({'family':'xgboost','hypothesis':'HYP-078 rename'},[],prior))
