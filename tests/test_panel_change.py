import pickle
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from automl_panel_memory import PANEL_FIELDS, panel_keys
from automl_panel_change import PanelChangeForecaster, PanelChangeHybridModel


class ConstantAnnual:
    def predict(self, values):
        return np.full(len(values), 2.)


class FakeBase:
    def predict(self, X):
        return np.full(len(X), 7.)


class FakePanel:
    def __init__(self):
        self.base_ = FakeBase()
        self.audit_ = {'test_double':True}
    def fit(self, X, y):
        self.columns_ = list(X)
        return self
    def predict(self, X):
        return np.full(len(X), 10.)


class FakeChange:
    def fit(self, X, y):
        self.audit_ = {'test_double':True}
        return self
    def predict_with_fallback(self, X, fallback):
        np.testing.assert_array_equal(fallback, np.full(len(X), 7.))
        return np.full(len(X), 30.)


def sample_frame(years=(1., 3.), profiles=501):
    rows = []
    for profile in range(profiles):
        for year in years:
            rows.append({'Year_Factor':year, 'State_Factor':'S', 'building_class':'C',
                'facility_type':'F', 'floor_area':float(1000+profile), 'year_built':2000.,
                'temperature':10.+year*4., 'energy_star_rating':50.+year,
                'opaque_id':len(rows)})
    frame = pd.DataFrame(rows)
    frame.index = np.arange(len(frame))*3+10
    return frame


class PanelChangeTests(unittest.TestCase):
    def fit_without_tree_training(self, X, y):
        captured = {}
        def no_fit(estimator, values, target, configuration):
            captured.update(target=np.asarray(target),configuration=configuration,values=values.copy())
            return ConstantAnnual()
        with patch('automl_combo_memory.fit_once',side_effect=no_fit):
            model = PanelChangeForecaster(threads=1).fit(X, y)
        return model, captured

    def test_same_year_future_covariates_and_targets_cannot_enter_features(self):
        X = sample_frame((1., 2., 3., 7.))
        y = 10+X.Year_Factor.to_numpy()*2
        original,_ = self.fit_without_tree_training(X,y)
        changed = X.copy()
        future = changed.Year_Factor >= 3
        changed.loc[future, 'temperature'] = 1e12
        changed.loc[future, 'energy_star_rating'] = -1e10
        changed_y = y.copy(); changed_y[future.to_numpy()] = 1e15
        altered,_ = self.fit_without_tree_training(changed,changed_y)
        query = X[X.Year_Factor.eq(3)].iloc[[0,10,22]].copy()
        for left,right in zip(original._frame(query),altered._frame(query)):
            if isinstance(left,pd.DataFrame):pd.testing.assert_frame_equal(left,right)
            else:np.testing.assert_array_equal(left,right)

    def test_annual_target_gap_and_prediction_scaling(self):
        X = sample_frame()
        y = 8.+2*X.Year_Factor.to_numpy()
        model,captured = self.fit_without_tree_training(X,y)
        np.testing.assert_allclose(captured['target'],2.)
        query = X.iloc[[0,2]].copy(); query['Year_Factor']=[5.,8.]
        query['temperature']=[30.,46.]
        frame,_,matched,gaps = model._frame(query)
        np.testing.assert_array_equal(gaps,[2.,5.])
        np.testing.assert_allclose(frame['__last_x_temperature'],22.)
        np.testing.assert_allclose(frame['__annual_dx_temperature'],[4.,4.8])
        np.testing.assert_allclose(model.predict_with_fallback(query,[999.,999.]),[18.,24.])
        self.assertTrue(matched.all())
        self.assertEqual(captured['configuration']['parameters']['n_estimators'],600)

    def test_unknown_profile_and_no_past_keep_fallback(self):
        X = sample_frame();model,_ = self.fit_without_tree_training(X,8+2*X.Year_Factor.to_numpy())
        query = X.iloc[:2].copy();query['Year_Factor']=[4.,1.]
        query.iloc[0,query.columns.get_loc('floor_area')]=-999.
        np.testing.assert_array_equal(model.predict_with_fallback(query,[77.,88.]),[77.,88.])
        with self.assertRaises(ValueError):model.predict_with_fallback(query,[np.nan,1.])
        with self.assertRaises(ValueError):model.predict_with_fallback(query,[1.])

    def test_batch_order_nondefault_index_and_serialization(self):
        X = sample_frame();model,_ = self.fit_without_tree_training(X,8+2*X.Year_Factor.to_numpy())
        query = X.iloc[[0,10,30]].copy();query['Year_Factor']=[4.,6.,8.]
        query.index=[41,7,90]
        output = model.predict_with_fallback(query,[1.,2.,3.])
        reverse = model.predict_with_fallback(query.iloc[::-1],[3.,2.,1.])
        np.testing.assert_array_equal(output,reverse[::-1])
        self.assertTrue(model._frame(query)[0].index.equals(query.index))
        reloaded = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(output,reloaded.predict_with_fallback(query,[1.,2.,3.]))

    def test_identifiers_are_neither_profile_keys_nor_predictors(self):
        X = sample_frame();model,captured = self.fit_without_tree_training(X,8+2*X.Year_Factor.to_numpy())
        self.assertNotIn('opaque_id',model.numeric_)
        self.assertFalse(any('opaque_id' in str(c) for c in captured['values'].columns))
        query = X.iloc[[0,2]].copy();query['Year_Factor']=5.
        altered = query.copy();altered['opaque_id']=[-1e15,1e15]
        self.assertEqual(panel_keys(query,PANEL_FIELDS),panel_keys(altered,PANEL_FIELDS))
        np.testing.assert_array_equal(model.predict_with_fallback(query,[1.,2.]),model.predict_with_fallback(altered,[1.,2.]))
        self.assertFalse(model.audit_['id_feature_used'])

    def test_factory_identity_contract_and_no_learned_weight_knobs(self):
        from automl_models import build_pipeline,canonical_candidate
        from automl_history import candidate_key,compatible_policy
        candidate = canonical_candidate({'family':'extra_trees','parameters':{},'strategy':'panel_change_hybrid'})
        self.assertIsInstance(build_pipeline('regression',candidate),PanelChangeHybridModel)
        self.assertTrue(compatible_policy(candidate))
        self.assertFalse(compatible_policy({**candidate,'preprocessing':'wrong'}))
        with self.assertRaises(ValueError):build_pipeline('classification',candidate)
        for modifications in [{'parameters':{'n_estimators':90000}}, {'family':'lightgbm'},
                              {'feature_code':'def engineer(X): return X'},
                              {'target_transform':'none'}, {'panel_method':'reliability'},
                              {'blend_weights':[.1,.9]}, {'meta_state':{'coef':1}}]:
            with self.subTest(modifications=modifications):
                with self.assertRaises(ValueError):canonical_candidate({**candidate,**modifications})
        old=canonical_candidate({'family':'extra_trees','parameters':{},'strategy':'panel_hybrid_memory'})
        self.assertNotEqual(candidate_key(candidate),candidate_key(old))
        self.assertEqual(candidate_key(candidate),candidate_key({**candidate,'hypothesis':'different name'}))

    def test_fixed_ensemble_and_wrapper_serialization(self):
        X=sample_frame(profiles=2);panel=FakePanel();change=FakeChange()
        with patch('automl_panel_memory.PanelHybridModel',return_value=panel),patch('automl_panel_change.PanelChangeForecaster',return_value=change):
            model=PanelChangeHybridModel().fit(X,np.arange(len(X))+1.)
        np.testing.assert_array_equal(model.predict(X),np.full(len(X),20.))
        self.assertNotIn('opaque_id',panel.columns_)
        self.assertEqual(model.audit_['fixed_blend_weights'],[.5,.5])
        self.assertTrue(model.keep_mask_.all())
        self.assertEqual(len(model.removed_positions_),0)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(X),model.predict(X))

    def test_eligibility_order_and_unfamiliar_schema(self):
        from solution import portfolio_candidates
        X=sample_frame();jobs=portfolio_candidates('regression','rmse',X,np.arange(len(X))+1.)
        self.assertEqual([job.get('strategy') for job in jobs[:2]],['panel_change_hybrid','panel_hybrid_memory'])
        unknown=pd.DataFrame({'signal':np.arange(1200.),'group':['a','b']*600})
        jobs=portfolio_candidates('regression','rmse',unknown,np.arange(len(unknown))+1.)
        self.assertFalse(any(job.get('strategy')=='panel_change_hybrid' for job in jobs))

    def test_insufficient_pairs_and_feature_collision(self):
        X=sample_frame(profiles=2)
        with self.assertRaises(ValueError):PanelChangeForecaster().fit(X,np.arange(len(X))+1.)
        X=sample_frame();X['__panel_last']=1.
        with self.assertRaises(ValueError):PanelChangeForecaster().fit(X,np.arange(len(X))+1.)


if __name__=='__main__':unittest.main()
