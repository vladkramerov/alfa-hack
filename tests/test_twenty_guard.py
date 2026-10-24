"""Regression test: reduced feature data must not be confused with old baseline."""
import os,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

class BaselineGuardTests(unittest.TestCase):
    def test_reduced_outer_matrix_is_a_new_fit_not_a_baseline_repeat(self):
        from experiments.twenty_common import make_model
        X=pd.DataFrame({'selected_column':np.arange(35000.)})
        with patch.dict(os.environ,{'SPEC006_ENFORCE_BASELINE_GUARD':'1','SPEC006_RUN_DIR':''}), patch('sklearn.ensemble.ExtraTreesRegressor.fit') as estimator_fit:
            make_model('regression',policy='legacy').fit(X,np.ones(35000))
            estimator_fit.assert_called_once()
    def test_exact_saved_outer_fingerprint_remains_prohibited(self):
        from experiments.twenty_common import make_model
        X=pd.DataFrame({'selected_column':np.arange(35000.)})
        with patch.dict(os.environ,{'SPEC006_ENFORCE_BASELINE_GUARD':'1','SPEC006_RUN_DIR':''}), patch('experiments.twenty_common.digest_frame',return_value='a97337d5f120b102c5d437909c504f7fecbc2c17b59c8aa6a761fd5ee55c496e'):
            with self.assertRaisesRegex(RuntimeError,'Known outer baseline'):
                make_model('regression',policy='legacy').fit(X,np.ones(35000))

if __name__=='__main__': unittest.main()
