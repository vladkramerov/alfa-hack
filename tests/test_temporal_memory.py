import unittest
import numpy as np
import pandas as pd

class TemporalMemoryTests(unittest.TestCase):
    def frame(self,years,areas=None):
        n=len(years)
        return pd.DataFrame({'Year_Factor':years,'State_Factor':['s']*n,'building_class':['c']*n,'facility_type':['f']*n,
                             'floor_area':areas or [100]*n,'year_built':[2000]*n,'ELEVATION':[np.nan]*n,'id':range(n)})
    def test_future_targets_never_used(self):
        from automl_temporal_memory import ProfileHistory
        model=ProfileHistory().fit(self.frame([1,2,5]),np.array([10.,20.,99999.]))
        mean,alpha,audit=model.lookup(self.frame([3]))
        self.assertAlmostEqual(mean[0],(10*.5+20)/1.5)
        self.assertAlmostEqual(alpha[0],2/3)
        self.assertEqual(audit['matched_rows'],1)
    def test_unknown_profile_falls_back_and_keys_ignore_id(self):
        from automl_temporal_memory import ProfileHistory
        model=ProfileHistory().fit(self.frame([1]),np.array([10.]))
        q=self.frame([2,2],[100.,999.]);q['id']=[100000,0]
        mean,alpha,_=model.lookup(q)
        self.assertEqual(mean[0],10.);self.assertEqual(alpha[0],.5);self.assertEqual(alpha[1],0.)
    def test_no_lookahead_in_same_year_and_staleness_reduces_weight(self):
        from automl_temporal_memory import ProfileHistory
        model=ProfileHistory().fit(self.frame([1,2]),[10.,999.])
        mean,alpha,_=model.lookup(self.frame([2,5]))
        self.assertEqual(mean[0],10.);self.assertAlmostEqual(alpha[0],.5)
        self.assertLess(alpha[1],2/3)
    def test_absent_time_is_rejected_before_fit(self):
        from automl_temporal_memory import ProfileHistory
        with self.assertRaises(ValueError):ProfileHistory().fit(self.frame([1]).drop(columns='Year_Factor'),[1.])

if __name__=='__main__':unittest.main()
