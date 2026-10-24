import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


class PlatformContractTests(unittest.TestCase):
    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), f'{name} not implemented')
        return importlib.import_module(name)

    def fixture(self, root, classification=False):
        folder=Path(root); nested=folder/'nested'/'inputs'; nested.mkdir(parents=True)
        y=np.tile(['red','blue'],20) if classification else np.arange(40,dtype=float)*1.25
        pd.DataFrame({'record_key':np.arange(40),'signal':np.arange(40),'kind':['a','b']*20,'outcome':y}).to_csv(nested/'train.csv',index=False)
        pd.DataFrame({'record_key':[100,101],'signal':[10,11],'kind':['new','a']}).to_csv(nested/'test.csv',index=False)
        pd.DataFrame({'record_key':[100,101],'outcome':['red','red'] if classification else [0.,0.]}).to_csv(nested/'sample_submission.csv',index=False)
        pd.DataFrame({'feature':['record_key','signal','kind','outcome'],'role':['id','feature','feature','target'],'dtype':['int','float','category','str' if classification else 'float']}).to_csv(nested/'data_dictionary.csv',index=False)
        task=folder/'task.md'
        task.write_text(('Multiclass classification. Metric Macro F1.\n' if classification else 'Regression. Metric RMSE.\n')+'Data: `nested/inputs/train.csv`, `nested/inputs/test.csv`, `nested/inputs/sample_submission.csv`, `nested/inputs/data_dictionary.csv`. Save submission.csv next to task.md.',encoding='utf-8')
        return task

    def test_nested_paths_and_renamed_target_identifier(self):
        api=self.module('automl_data')
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as root:
            bundle=api.load_task(self.fixture(root))
            self.assertEqual(bundle.target,'outcome'); self.assertEqual(bundle.id_column,'record_key')
            self.assertEqual(bundle.task_type,'regression'); self.assertEqual(bundle.metric,'rmse')
            self.assertEqual(bundle.X.columns.tolist(),['signal','kind'])

    def test_classification_is_not_inferred_from_sample_placeholder(self):
        api=self.module('automl_data')
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as root:
            bundle=api.load_task(self.fixture(root,True))
            self.assertEqual(bundle.task_type,'classification'); self.assertEqual(bundle.metric,'macro_f1')
            answer=api.fallback_submission(bundle)
            api.validate_submission(bundle,answer)
            bad=answer.iloc[::-1]
            with self.assertRaises(ValueError): api.validate_submission(bundle,bad)

    def test_candidate_fingerprint_ignores_name_and_seed(self):
        api=self.module('automl_history')
        a={'family':'extra_trees','parameters':{'n_estimators':200,'random_state':42},'feature_code':'','hypothesis':'one'}
        b={'family':'extra_trees','parameters':{'random_state':7,'n_estimators':200},'feature_code':'','hypothesis':'renamed'}
        self.assertEqual(api.candidate_key(a),api.candidate_key(b))

    def test_empty_choices_and_truncated_responses_are_rejected(self):
        api=self.module('automl_llm')
        self.assertIsNone(api.parse_action({'choices':[]}))
        self.assertIsNone(api.parse_action({'choices':[{'finish_reason':'length','message':{'content':'{"family"'}}]}))
        self.assertIsNone(api.parse_action({'choices':[{'message':{'content':None}}]}))

    def test_generated_code_is_rowwise_and_cannot_read_files(self):
        api=self.module('automl_models')
        frame=pd.DataFrame({'a':[1.,2.],'b':[3.,4.]})
        transform=api.FeatureCode('def engineer(X):\n    X = X.copy()\n    X["interaction"] = X["a"] * X["b"]\n    return X\n')
        out=transform.fit_transform(frame)
        np.testing.assert_array_equal(out.interaction.to_numpy(),[3.,8.])
        for code in ['def engineer(X):\n    return pd.read_csv("train.csv")', 'def engineer(X):\n    X["x"]=X["a"].mean()\n    return X']:
            with self.assertRaises(ValueError): api.FeatureCode(code).fit_transform(frame)


if __name__=='__main__': unittest.main()
