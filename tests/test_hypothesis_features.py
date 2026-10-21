import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hypothesis_suite as suite


class FeatureTests(unittest.TestCase):
    def test_ratios_handle_zero_without_mutating_original(self):
        X=pd.DataFrame({'sbytes':[0,10],'dbytes':[0,5],'spkts':[0,2],'dpkts':[0,1]})
        original=X.copy()
        result=suite.engineer(X,'flow_ratios')
        pd.testing.assert_frame_equal(X,original)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertEqual(result.loc[0,'byte_ratio'],1)
        self.assertEqual(result.loc[1,'source_bytes_per_packet'],10/3)

    def test_zero_year_only_changes_zero(self):
        X=pd.DataFrame({'year_built':[0,2000,np.nan]})
        result=suite.engineer(X,'zero_year')
        self.assertTrue(np.isnan(result.loc[0,'year_built']))
        self.assertEqual(result.loc[1,'year_built'],2000)
        self.assertEqual(X.loc[0,'year_built'],0)

    def test_native_numeric_statistics_use_only_fit_rows(self):
        fit=pd.DataFrame({'number':[1.,3.,np.nan],'cat':['a',None,'b']})
        valid=pd.DataFrame({'number':[np.nan,1000.],'cat':[None,'new']})
        prep=suite.NativePreprocessor().fit(fit)
        transformed=prep.transform(valid)
        self.assertEqual(prep.numeric_pipeline_.named_steps['fill'].statistics_[0],2)
        self.assertEqual(transformed.cat.tolist(),['missing','new'])
        self.assertTrue(np.isfinite(transformed.number).all())

    def test_ensemble_aligns_class_order(self):
        prob=np.array([[0.8,0.2]])
        aligned=suite.align_probabilities(prob,['B','A'],['A','B'])
        np.testing.assert_allclose(aligned,[[0.2,0.8]])


if __name__=='__main__':unittest.main()
