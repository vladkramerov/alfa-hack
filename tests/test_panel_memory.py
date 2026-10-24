import unittest
import numpy as np
import pandas as pd
from automl_panel_memory import PanelHistory, panel_keys, PANEL_FIELDS


class PanelMemoryTests(unittest.TestCase):
    def frame(self, years, area=None):
        n = len(years)
        return pd.DataFrame({'Year_Factor': years, 'State_Factor': ['S']*n,
            'building_class': ['C']*n, 'facility_type': ['F']*n,
            'floor_area': area or [100.]*n, 'year_built': [2000.]*n,
            'ELEVATION': np.arange(n), 'opaque_id': np.arange(n)})

    def test_query_year_and_future_targets_do_not_enter_memory(self):
        X = self.frame([1,2,3,7])
        a = PanelHistory().fit(X,[10.,20.,999.,1e9])
        b = PanelHistory().fit(X,[10.,20.,-888.,-1e9])
        query = self.frame([3])
        for model in [a,b]:
            mean,alpha,_=model.lookup(query)
            np.testing.assert_allclose(mean,[(.5*10+20)/1.5])
            np.testing.assert_allclose(alpha,[2/3])
        pd.testing.assert_frame_equal(a.features(query)[0], b.features(query)[0])

    def test_training_feature_has_no_own_or_future_label(self):
        X=self.frame([1,2,3])
        a=PanelHistory().fit(X,[10.,20.,30.]).features(X)[0]
        b=PanelHistory().fit(X,[10.,20.,1e20]).features(X)[0]
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(a.iloc[0].isna().all())
        self.assertEqual(a.iloc[1]['__panel_mean'],10.)

    def test_unknown_profile_and_new_identifier_elevation(self):
        model=PanelHistory().fit(self.frame([1]),[10.])
        query=self.frame([2,2],[100.,999.]); query['ELEVATION']=99999;query['opaque_id']=-1
        mean,alpha,audit=model.lookup(query)
        np.testing.assert_array_equal(mean,[10.,0.]);np.testing.assert_array_equal(alpha,[.5,0.])
        self.assertEqual(audit['matched_rows'],1)

    def test_batch_order_and_nondefault_indices_preserved(self):
        model=PanelHistory().fit(self.frame([1,2]),[10.,20.])
        query=self.frame([3,8,2]);query.index=[8,4,20]
        means,weights,_=model.lookup(query)
        reversed_means,reversed_weights,_=model.lookup(query.iloc[::-1])
        np.testing.assert_array_equal(means,reversed_means[::-1])
        np.testing.assert_array_equal(weights,reversed_weights[::-1])
        self.assertTrue(model.features(query)[0].index.equals(query.index))

    def test_collision_is_reported_and_keys_do_not_confuse_missing_literal(self):
        X=self.frame([1,1,2]);X.loc[2,'State_Factor']='M:'
        model=PanelHistory().fit(X,[10.,20.,30.])
        self.assertEqual(model.audit_['same_profile_year_multiple_rows'],1)
        X.loc[2,'State_Factor']=None
        self.assertNotEqual(panel_keys(X,PANEL_FIELDS)[2],panel_keys(self.frame([1]),PANEL_FIELDS)[0])

    def test_factory_bounds_and_identity(self):
        from automl_models import canonical_candidate,build_pipeline
        from automl_history import candidate_key,compatible_policy
        c=canonical_candidate({'family':'extra_trees','parameters':{},'strategy':'panel_hybrid_memory'})
        self.assertTrue(compatible_policy(c))
        self.assertEqual(type(build_pipeline('regression',c)).__name__,'PanelHybridModel')
        with self.assertRaises(ValueError):build_pipeline('classification',c)
        with self.assertRaises(ValueError):canonical_candidate({**c,'parameters':{'n_estimators':120000}})
        old=canonical_candidate({'family':'extra_trees','parameters':{},'strategy':'temporal_profile_memory'})
        self.assertNotEqual(candidate_key(c),candidate_key(old))

    def test_generic_unfamiliar_data_does_not_get_domain_strategy(self):
        from solution import portfolio_candidates
        X=pd.DataFrame({'unfamiliar_signal':np.arange(1000),'group':['a','b']*500})
        jobs=portfolio_candidates('regression','rmse',X,np.arange(1000)+1.)
        self.assertFalse(any(j.get('strategy')=='panel_hybrid_memory' for j in jobs))


if __name__=='__main__':unittest.main()
