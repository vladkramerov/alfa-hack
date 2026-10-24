import unittest
import numpy as np
import pandas as pd
from automl_models import canonical_candidate, build_pipeline
from automl_residual import CausalResidual
from automl_panel import CausalPanelFeatures, stable_panel_columns
from automl_engine import offline_proposal


class CausalHybridTests(unittest.TestCase):
    def fixture(self):
        entity=np.tile(np.arange(100),3)
        period=np.repeat([1.,2.,3.],100)
        X=pd.DataFrame({'clock':period,'identifier_like_measure':entity+100.,
                        'stable_sector':np.where(entity%2,'left','right'),
                        'changing_measure':np.sin(entity)+period})
        return X, 10+entity*.1+period

    def test_dispatch_and_scope(self):
        c={'family':'extra_trees','panel_time_column':'clock','panel_strategy':'hybrid'}
        self.assertEqual(type(build_pipeline('regression',c)).__name__,'CausalHybridModel')
        with self.assertRaises(ValueError):build_pipeline('classification',c)
        for wrong in [{'family':'extra_trees','panel_strategy':'hybrid'},
                      {'family':'lightgbm','panel_time_column':'clock','panel_strategy':'hybrid'}]:
            with self.assertRaises(ValueError):canonical_candidate(wrong)

    def test_history_strict_past_and_generic_schema(self):
        X,y=self.fixture()
        a=CausalResidual('clock',strict_key=True).fit(X,y)
        future=y.copy();future[X.clock>=3]+=10000
        b=CausalResidual('clock',strict_key=True).fit(X,future)
        query=X.loc[X.clock==3]
        np.testing.assert_equal(a._features(query).to_numpy(),b._features(query).to_numpy())
        columns=stable_panel_columns(X,'clock',CausalPanelFeatures('clock').fit(X,y).columns_)
        self.assertIn('stable_sector',columns)
        self.assertNotIn('changing_measure',columns)
        renamed=X.rename(columns={'identifier_like_measure':'a','stable_sector':'b','changing_measure':'c'})
        c=CausalResidual('clock',strict_key=True).fit(renamed,y)
        np.testing.assert_allclose(a.stable(query,np.ones(len(query))),
                                   c.stable(renamed.loc[query.index],np.ones(len(query))))

    def test_cold_entities_and_order(self):
        X,y=self.fixture();m=CausalResidual('clock',strict_key=True).fit(X,y)
        query=X.iloc[:20].copy();query.clock=4
        baseline=np.arange(20,dtype=float)
        normal=m.stable(query,baseline)
        permutation=np.arange(20)[::-1]
        np.testing.assert_allclose(m.stable(query.iloc[permutation],baseline[permutation]),normal[permutation])
        query.identifier_like_measure+=10000
        np.testing.assert_array_equal(m.stable(query,baseline),baseline)
        np.testing.assert_array_equal(m.predict(query,baseline),baseline)

    def test_portfolio_has_real_dispatch(self):
        profile={'type':'regression','time_column':'clock','metric':'rmse','target_min':0,
                 'target_skew':0,'numeric_target_correlations':{},'shape':[1000,4],
                 'categories':{},'drift_columns':[],'nunique':{}}
        from automl_models import candidate_key
        observations=[{'status':'completed','score':5,'candidate':{'family':'lightgbm'}}]
        first=offline_proposal(profile,observations,set())
        second=offline_proposal(profile,observations,{candidate_key(first)})
        self.assertEqual(second['panel_strategy'],'hybrid')
        self.assertEqual(type(build_pipeline('regression',second)).__name__,'CausalHybridModel')
