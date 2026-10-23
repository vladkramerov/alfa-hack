"""Lossless lexical ID/class transport; numeric outputs retain numeric semantics."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from automl_data import load_task,make_submission,atomic_submission,read_submission


class CsvCodecTests(unittest.TestCase):
    def fixture(self,root,ids,labels,declared_strings=True,regression=False,probabilities=False):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        target=list(labels)*4;n=len(target)
        pd.DataFrame({'key':np.arange(n),'signal':np.arange(n),'response':target}).to_csv(root/'train.csv',index=False)
        pd.DataFrame({'key':ids,'signal':np.arange(len(ids))}).to_csv(root/'test.csv',index=False)
        if probabilities:sample=pd.DataFrame({'key':ids,**{str(label):np.zeros(len(ids)) for label in reversed(labels)}})
        else:sample=pd.DataFrame({'key':ids,'response':[labels[0]]*len(ids)})
        sample.to_csv(root/'sample_submission.csv',index=False)
        pd.DataFrame({'feature':['key','signal','response'],'role':['id','feature','target'],
                      'dtype':['str' if declared_strings else 'numeric','numeric','numeric' if regression or not declared_strings else 'str']}).to_csv(root/'data_dictionary.csv',index=False)
        task=root/'task.md';metric='RMSE' if regression else 'Log Loss' if probabilities else 'Macro F1'
        task.write_text(('Regression' if regression else 'Classification')+f'.\nMetric: {metric}.\n'
                        '`train.csv` `test.csv` `sample_submission.csv` `data_dictionary.csv`\n',encoding='utf-8')
        return load_task(task)

    def test_declared_strings_survive_literal_na_signs_and_zeroes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            ids=['01','1','NA'];labels=['+1','001','NA'];bundle=self.fixture(folder,ids,labels)
            self.assertEqual(bundle.id_column,'key');self.assertEqual(bundle.test.key.tolist(),ids)
            self.assertEqual(bundle.y.tolist(),labels*4)
            frame=make_submission(bundle,np.array(labels));atomic_submission(bundle,frame)
            result=read_submission(bundle,Path(folder)/'submission.csv')
            self.assertEqual(result.key.tolist(),ids);self.assertEqual(result.response.tolist(),labels)

    def test_undeclared_lexical_id_restored_before_unique_id_discovery(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            bundle=self.fixture(folder,['01','1','+1'],['red','blue'],declared_strings=False)
            self.assertEqual(bundle.id_column,'key');self.assertEqual(bundle.test.key.tolist(),['01','1','+1'])
            self.assertNotIn('key',bundle.X)

    def test_irreversible_numeric_class_coercion_preserves_distinct_labels(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            bundle=self.fixture(folder,[100,101],['+1','001'],declared_strings=False)
            self.assertEqual(set(bundle.y),{'+1','001'})
            atomic_submission(bundle,make_submission(bundle,np.array(['+1','001'])))
            self.assertEqual(read_submission(bundle,Path(folder)/'submission.csv').response.tolist(),['+1','001'])

    def test_ordinary_numeric_ids_and_regression_target_keep_types(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            bundle=self.fixture(folder,[100,101],['001','002'],declared_strings=False,regression=True)
            self.assertTrue(pd.api.types.is_integer_dtype(bundle.test.key))
            self.assertTrue(pd.api.types.is_integer_dtype(bundle.y));self.assertEqual(bundle.y.iloc[0],1)
            atomic_submission(bundle,make_submission(bundle,np.array([1.25,2.5])))
            result=read_submission(bundle,Path(folder)/'submission.csv')
            self.assertTrue(pd.api.types.is_integer_dtype(result.key));self.assertTrue(pd.api.types.is_numeric_dtype(result.response))

    def test_probability_values_numeric_with_lexical_class_names(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            bundle=self.fixture(folder,['001','+1'],['001','010'],probabilities=True)
            probability=np.array([[.8,.2],[.3,.7]])
            atomic_submission(bundle,make_submission(bundle,probability))
            result=read_submission(bundle,Path(folder)/'submission.csv')
            self.assertEqual(result.key.tolist(),['001','+1']);np.testing.assert_allclose(result[['010','001']],probability)
            self.assertTrue(all(pd.api.types.is_numeric_dtype(result[c]) for c in ['010','001']))

    def test_blank_target_is_missing_even_with_declared_strings(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            with self.assertRaisesRegex(ValueError,'missing'):self.fixture(folder,['a','b'],['ok',''])

    def test_declared_boolean_looking_labels_are_not_boolean_coerced(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            bundle=self.fixture(folder,['True','False'],['true','FALSE'])
            self.assertEqual(bundle.test.key.tolist(),['True','False']);self.assertEqual(set(bundle.y),{'true','FALSE'})
            atomic_submission(bundle,make_submission(bundle,np.array(['true','FALSE'])))
            result=read_submission(bundle,Path(folder)/'submission.csv')
            self.assertEqual(result.key.tolist(),['True','False']);self.assertEqual(result.response.tolist(),['true','FALSE'])

    def test_numeric_output_does_not_accept_csv_boolean_text(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            bundle=self.fixture(folder,[100,101],[1.,2.],declared_strings=False,regression=True)
            path=Path(folder)/'bad.csv';path.write_text('key,response\n100,True\n101,False\n',encoding='utf-8')
            with self.assertRaises(ValueError):read_submission(bundle,path)


if __name__=='__main__':unittest.main()
