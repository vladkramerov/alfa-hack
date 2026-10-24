import unittest
import numpy as np
import pandas as pd


def frame():
    return pd.DataFrame({'State_Factor':['S']*4,'building_class':['A']*4,
        'facility_type':['Office']*4,'floor_area':[100.]*4,'year_built':[1980.]*4,
        'Year_Factor':[1.,2.,3.,4.],'energy_star_rating':[30.,40.,50.,60.]})


class MultiLagTests(unittest.TestCase):
    def test_exact_lag_gap_self_and_future_excluded(self):
        from external_multilag import CausalMultiLag
        X=frame();h=CausalMultiLag().fit(X,[10.,20.,3000.,4000.])
        f=h.transform(X)
        self.assertTrue(np.isnan(f.iloc[0]['__ml_lag1']))
        self.assertEqual(f.iloc[1]['__ml_lag1'],10.)
        self.assertEqual(f.iloc[2]['__ml_lag2'],10.)
        self.assertEqual(f.iloc[3]['__ml_lag3'],10.)
        altered=CausalMultiLag().fit(X,[10.,20.,9e8,1e10]).transform(X)
        np.testing.assert_allclose(f.iloc[:2].select_dtypes('number'),altered.iloc[:2].select_dtypes('number'),equal_nan=True)
        h=CausalMultiLag().fit(X.iloc[[0,2]],[10.,30.]);query=X.iloc[[3]]
        f=h.transform(query)
        self.assertEqual(f.iloc[0]['__ml_lag1'],30.)
        self.assertTrue(np.isnan(f.iloc[0]['__ml_lag2']))
        self.assertEqual(f.iloc[0]['__ml_lag3'],10.)

    def test_collision_missing_profile_and_unknown_category(self):
        from external_multilag import CausalMultiLag,NativeFrame
        X=frame();dup=pd.concat([X.iloc[:2],X.iloc[[1]]],ignore_index=True)
        h=CausalMultiLag().fit(dup,[1.,2.,3.]);f=h.transform(X.iloc[[3]])
        self.assertEqual(f.iloc[0]['__ml_count'],0.)
        missing=X.iloc[:2].copy();missing['year_built']=0
        h=CausalMultiLag().fit(missing,[1.,2.]);self.assertEqual(h.transform(missing).iloc[1]['__ml_count'],0.)
        enc=NativeFrame().fit(X);q=X.iloc[[0]].copy();q['facility_type']='unseen'
        self.assertTrue(pd.isna(enc.transform(q).iloc[0]['facility_type']))

    def test_selection_fit_only_and_batch_invariance(self):
        from external_multilag import CausalMultiLag
        X=frame();h=CausalMultiLag().fit(X,[10.,20.,30.,40.])
        self.assertEqual(h.changing_columns_,[])
        combined=h.transform(X.iloc[[3,0,3]])
        single=h.transform(X.iloc[[3]])
        pd.testing.assert_frame_equal(combined.iloc[[0]].reset_index(drop=True),single.reset_index(drop=True))
        pd.testing.assert_frame_equal(combined.iloc[[2]].reset_index(drop=True),single.reset_index(drop=True))

    def test_changing_field_selection_freezes_before_query(self):
        from external_multilag import CausalMultiLag
        n=240
        X=pd.concat([frame().iloc[[0]].copy() for _ in range(2*n)],ignore_index=True)
        X['floor_area']=np.tile(np.arange(n)+100.,2)
        X['Year_Factor']=np.repeat([1.,2.],n)
        r=np.arange(n,dtype=float)
        X['energy_star_rating']=np.concatenate([r,r+np.where(r%2==0,1.,0.)])
        h=CausalMultiLag().fit(X,X.energy_star_rating.to_numpy()*2)
        self.assertEqual(h.changing_columns_,['energy_star_rating'])
        q=X.iloc[[0]].copy();q['Year_Factor']=3.;q['energy_star_rating']=1e9
        f=h.transform(q)
        self.assertEqual(h.changing_columns_,['energy_star_rating'])
        self.assertEqual(f['__ml_energy_star_rating_prev'].iloc[0],1.)


if __name__=='__main__':unittest.main()
