import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import numpy as np
import pandas as pd
from automl_history import candidate_key
from automl_models import FeatureCode
from automl_llm import propose
from tests import test_platform_agent as fixtures


class IntegrationContracts(unittest.TestCase):
    def test_defaults_and_explicit_defaults_are_the_same_experiment(self):
        self.assertEqual(candidate_key({'family':'extra_trees','parameters':{}}),candidate_key({'family':'extra_trees','parameters':dict(n_estimators=200,max_depth=24,min_samples_leaf=2,max_features=.8)}))

    def test_io_and_non_rowwise_numpy_are_rejected(self):
        frame=pd.DataFrame({'x':[1,2]})
        for action in ['np.loadtxt("file")','np.fromfile("file")','np.average(X["x"])']:
            with self.assertRaises(ValueError): FeatureCode('def engineer(X):\n    return '+action).fit_transform(frame)

    def test_api_errors_and_timeouts_are_bounded(self):
        env={'OPENAI_BASE_URL':'http://localhost:1/v1','OPENAI_API_KEY':'local-test-key','OPENAI_MODEL':'mock'}
        for error in [TimeoutError(),urllib.error.HTTPError('http://localhost',429,'rate limit',{},None)]:
            with patch.dict('os.environ',env),patch('urllib.request.urlopen',side_effect=error) as call:
                action,status=propose({},[],timeout=.01)
                self.assertIsNone(action); self.assertEqual(status['status'],'failed'); self.assertEqual(call.call_count,2)

    def test_llm_context_retains_latest_measured_feedback_as_valid_json(self):
        from automl_llm import research_context
        profile={'task':'x'*100000,'numeric_summary':{str(i):{'mean':i,'max':i+10} for i in range(1000)}}
        context=research_context(profile,[{'candidate':{'family':'extra_trees'},'score':.6123,'status':'completed'}])
        decoded=json.loads(context)
        self.assertEqual(decoded['observations'][-1]['score'],.6123)
        self.assertLess(len(context),30000)

    def test_missing_api_configuration_is_explicit(self):
        with patch.dict('os.environ',{},clear=True):
            action,status=propose({},[])
            self.assertIsNone(action); self.assertEqual(status['status'],'unconfigured')

    def test_persistent_hypothesis_file_separates_data_and_survives_restart(self):
        from automl_history import read_ledger,write_ledger
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as root:
            path=Path(root)/'hypotheses.json'
            write_ledger(path,'data-a',{'records':[{'candidate_key':'already-attempted','status':'failed'}]})
            write_ledger(path,'data-b',{'records':[{'candidate_key':'another','status':'completed','score':.8}]})
            self.assertEqual(read_ledger(path,'data-a')['records'][0]['candidate_key'],'already-attempted')
            self.assertEqual(read_ledger(path,'data-b')['records'][0]['score'],.8)
            self.assertEqual(read_ledger(path,'unseen')['records'],[])

    def test_probability_columns_follow_named_classes(self):
        from automl_data import load_task,fallback_submission
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as root:
            task=fixtures.PlatformContractTests().fixture(root,True)
            task.write_text(task.read_text().replace('Macro F1','Log Loss'),encoding='utf-8')
            sample=Path(root)/'nested/inputs/sample_submission.csv'
            pd.DataFrame({'record_key':[100,101],'red':[0.,0.],'blue':[0.,0.]}).to_csv(sample,index=False)
            bundle=load_task(task)
            np.testing.assert_allclose(fallback_submission(bundle)[['red','blue']],.5)
