import unittest
import numpy as np
import pandas as pd


class AutoKeyTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'Year_Factor':np.repeat([1,2,3],8),
            'opaque_profile':np.tile(np.arange(8),3),'weather':np.repeat([10.,20.,30.],8)})

    def test_fit_only_discovery_and_strict_past(self):
        from external_key import AutoKeyMemory
        X=self.frame();y=np.tile(np.arange(8)*10.+10,3)
        m=AutoKeyMemory().fit(X,y)
        self.assertEqual(m.columns_,['opaque_profile'])
        a=m.predict_with_fallback(X.iloc[:8],np.full(8,100.))
        np.testing.assert_array_equal(a,np.full(8,100.))
        q=X.iloc[16:];a=m.predict_with_fallback(q,np.full(8,100.))
        # Freeze fitted discovery, then poison only same-period/future donors.
        before=m.lookup(q)[0].copy()
        m.records_={k:[(year,target+1e6 if year>=3 else target) for year,target in rows]
                    for k,rows in m.records_.items()}
        np.testing.assert_array_equal(m.lookup(q)[0],before)
        np.testing.assert_array_equal(before,y[16:])
        self.assertEqual(m.columns_,['opaque_profile'])

    def test_batch_order_and_missing_fallback(self):
        from external_key import AutoKeyMemory
        X=self.frame();y=np.tile(np.arange(8)*10.+10,3);m=AutoKeyMemory().fit(X,y)
        q=X.iloc[[16,18,16]].copy();p=m.predict_with_fallback(q,np.full(3,100.))
        self.assertEqual(p[0],p[2]);self.assertEqual(p[0],m.predict_with_fallback(q.iloc[:1],[100.])[0])
        q['opaque_profile']=np.nan
        np.testing.assert_array_equal(m.predict_with_fallback(q,[100.,100.,100.]),np.full(3,100.))

    def test_fixed_configuration_and_no_period_fallback(self):
        from external_key import AutoKeyMemory
        X=self.frame().iloc[:8];m=AutoKeyMemory().fit(X,np.arange(8)+1.)
        np.testing.assert_array_equal(m.predict_with_fallback(X,np.ones(8)),np.ones(8))
        self.assertEqual(m.configuration()['hypothesis'],'HYP-106')
