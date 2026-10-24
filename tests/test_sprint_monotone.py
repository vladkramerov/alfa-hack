import importlib
import importlib.util
import json
import pickle
import unittest

import numpy as np
import pandas as pd


class MonotoneFixedBase:
    def fit(self,X,y):
        raise AssertionError('H70 baseline must not be refit')

    def predict(self,X):
        return np.full(len(X),100.)


def frame(n=420):
    rating = np.r_[np.tile(np.arange(100),4),np.full(max(0,n-400),np.nan)][:n]
    return pd.DataFrame({'Year_Factor': np.ones(n), 'State_Factor': ['S']*n,
        'building_class': ['Commercial']*n, 'facility_type': ['Office']*n,
        'floor_area': np.arange(n,dtype=float)+1000., 'year_built': np.full(n,1950.),
        'ELEVATION': np.zeros(n), 'energy_star_rating': rating,
        'direction_max_wind_speed': np.zeros(n)},index=np.arange(1000,1000+n))


class MonotoneExpertTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_monotone'),'H57 module missing')
        self.module=importlib.import_module('sprint_monotone')

    def test_actual_known_subset_basic_constraint_and_monotone_unmatched_rating_grid(self):
        X=frame(); X.iloc[1,X.columns.get_loc('floor_area')]=1000. # Ambiguous profile.
        source=X.copy(deep=True)
        target=np.r_[400.-3*np.tile(np.arange(100),4),np.full(20,1e9)]
        events=[]
        def hook(estimator,features,y,configuration,**kwargs):
            events.append((features.copy(),np.asarray(y).copy(),configuration,dict(kwargs)))
            return estimator.fit(features,y,**kwargs)
        model=self.module.MonotoneUnmatchedRegressor(fit_hook=hook).fit(X,target,MonotoneFixedBase())
        self.assertEqual(len(events),1)
        self.assertEqual(events[0][0].index.tolist(),list(range(1000,1400)))
        np.testing.assert_array_equal(events[0][1],target[:400])
        self.assertEqual(model.feature_.audit_['fit_rows'],400)
        area_column=model.encoder_.numeric_.index('floor_area')
        self.assertAlmostEqual(model.encoder_.imputer_.statistics_[area_column],1199.5)
        self.assertEqual(model.feature_.numbers,(29,30))
        columns=model.audit_['native_feature_columns']; constraint=model.audit_['monotone_constraints']
        self.assertEqual(len(columns),len(constraint)); self.assertEqual(sum(constraint),-1)
        self.assertEqual(constraint[columns.index('energy_star_rating')],-1)
        self.assertTrue(all(v==0 for name,v in zip(columns,constraint) if name!='energy_star_rating'))
        self.assertEqual(model.model_.get_params()['monotone_constraints_method'],'basic')
        self.assertEqual(model.model_.get_params()['n_estimators'],700)
        np.testing.assert_array_equal(model.fit_positions_,np.arange(400))
        q=pd.concat([X.iloc[[2]]]*101,ignore_index=True)
        q['floor_area']=999999.; q['Year_Factor']=2.; q['energy_star_rating']=np.arange(101)
        prediction=model.predict(q)
        self.assertTrue((np.diff(prediction)<=1e-10).all())
        self.assertGreater(prediction[0]-prediction[-1],100.)
        self.assertEqual(model.last_prediction_audit_['expert_applied_rows'],101)
        # Known exact past profile, missing rating, same-year profile, ambiguous past profile.
        gates=X.iloc[[2,3,2,0,2]].copy(); gates['Year_Factor']=[2.,2.,1.,2.,0.]
        gates['energy_star_rating']=[50.,np.nan,0.,0.,0.]
        gated=model.predict(gates)
        np.testing.assert_array_equal(gated[:2],[100.,100.])
        self.assertGreater(gated[2],150.); self.assertGreater(gated[3],150.)
        self.assertGreater(gated[4],150.)
        self.assertEqual(model.last_prediction_audit_['expert_applied_rows'],3)
        unseen=q.iloc[:2].copy(); unseen['facility_type']='UNSEEN'
        self.assertTrue(np.isfinite(model.predict(unseen)).all())
        restored=pickle.loads(pickle.dumps(model))
        self.assertIsNone(restored.fit_hook)
        np.testing.assert_array_equal(restored.predict(q),prediction)
        json.dumps(model.configuration()); json.dumps(model.audit_)
        pd.testing.assert_frame_equal(X,source)

    def test_small_known_subset_skips_learner_and_preserves_baseline(self):
        X=frame(20); model=self.module.MonotoneUnmatchedRegressor().fit(X,np.arange(20.),MonotoneFixedBase())
        self.assertIsNone(model.model_); self.assertEqual(model.audit_['expert_fit_rows'],20)
        q=X.copy();q['Year_Factor']=2.;q['floor_area']+=10000
        np.testing.assert_array_equal(model.predict(q),np.full(20,100.))

    def test_string_rating_coercion_and_zero_observed_hook_pickle(self):
        X=frame(400); X['energy_star_rating']=X['energy_star_rating'].astype(str)
        model=self.module.MonotoneUnmatchedRegressor().fit(X,np.arange(400.)%100,MonotoneFixedBase())
        self.assertEqual(model.audit_['expert_fit_rows'],400)
        self.assertIn('energy_star_rating',model.encoder_.numeric_)
        self.assertNotIn('energy_star_rating',model.encoder_.categorical_)
        self.assertTrue(model.fit_mask_[0])

    def test_baseline_required_invalid_inputs_and_query_schema(self):
        X=frame(20)
        with self.assertRaises(ValueError): self.module.MonotoneUnmatchedRegressor().fit(X,np.arange(20.))
        with self.assertRaises(ValueError): self.module.MonotoneUnmatchedRegressor(threads=5)
        with self.assertRaises(ValueError): self.module.MonotoneUnmatchedRegressor().fit(X,-np.ones(20),MonotoneFixedBase())
        model=self.module.MonotoneUnmatchedRegressor().fit(X,np.arange(20.),MonotoneFixedBase())
        with self.assertRaises(ValueError):model.predict(X.drop(columns=['ELEVATION']))


if __name__=='__main__':
    unittest.main()
