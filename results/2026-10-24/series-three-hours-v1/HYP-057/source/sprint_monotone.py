"""H57 known-rating monotone global expert for H70-unmatched queries."""
from copy import deepcopy

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from automl_models import Encoder
from automl_reg_combo import FeatureComposition
from sprint_regression import LINK_FIELDS, _keys, _years


class MonotoneUnmatchedRegressor:
    """Constrain only raw rating; preserve H70 on missing or linked queries."""

    def __init__(self, threads=2, fit_hook=None):
        if (isinstance(threads,bool) or not isinstance(threads,(int,np.integer)) or not 1<=threads<=4
                or (fit_hook is not None and not callable(fit_hook))):
            raise ValueError('Need one to four CPU threads and optional callable fit hook')
        self.threads,self.fit_hook=int(threads),fit_hook
        self.parameters={'n_estimators':700,'learning_rate':.04,'num_leaves':31,
            'min_child_samples':40,'reg_lambda':10.,'objective':'regression',
            'monotone_constraints_method':'basic','random_state':42,'n_jobs':self.threads,'verbosity':-1}

    def __getstate__(self):
        state=self.__dict__.copy()
        state['fit_hook']=None
        return state

    def configuration(self):
        return {'hypothesis':'HYP-057','strategy':'monotone_known_rating_unmatched_expert','version':1,
            'family':'lightgbm','parameters':deepcopy(self.parameters),'feature_numbers':[29,30],
            'feature_statistics_source':'known_rating_train_subset_only','encoder':'native_categorical_no_scale',
            'constraint_rule':'raw_energy_star_rating_minus_one_all_other_encoded_features_zero',
            'monotone_semantics':'research_assumption_not_established_causal_law',
            'monotonicity_scope':'expert_with_other_features_fixed_not_arbitrary_H70_blend',
            'target_transform':'none','loss':'original_scale_squared_error','minimum_training_known_rows':300,
            'link_fields':list(LINK_FIELDS),'ambiguity':'reject_profile_with_any_same_year_collision',
            'query_gate':'finite_rating_and_no_unambiguous_strictly_past_H70_profile',
            'zero_rating':'observed','expert_weight':.7,'base_weight':.3,
            'negative_expert_prediction':'clip_zero_before_blend','base_refit':False,
            'query_statistics':False,'target_outlier_deletion':False,
            'selection':'fixed_all_rounds_no_validation_earlystop'}

    @staticmethod
    def _known(X):
        if 'energy_star_rating' not in X:
            raise ValueError('Explicit energy_star_rating required')
        return np.isfinite(pd.to_numeric(X['energy_star_rating'],errors='coerce').to_numpy(float))

    def _expert_input(self,X):
        out=X.loc[:,self.feature_input_columns_].copy()
        out['energy_star_rating']=pd.to_numeric(out['energy_star_rating'],errors='coerce').astype(float)
        return out

    def fit(self,X,y,base_predictor=None):
        target=np.asarray(y,float)
        if (not isinstance(X,pd.DataFrame) or not len(X) or not X.columns.is_unique
                or not all(isinstance(column,str) for column in X) or target.shape!=(len(X),)
                or not np.isfinite(target).all() or np.any(target<0)):
            raise ValueError('Expected unique-column raw frame and finite nonnegative aligned targets')
        if base_predictor is None or not (hasattr(base_predictor,'predict') or callable(base_predictor)):
            raise ValueError('Explicit compatible H70 baseline required')
        years,keys=_years(X),_keys(X)
        self.columns_,self.base_predictor_=list(X.columns),base_predictor
        self.feature_input_columns_=[column for column in X
            if column.lower() not in {'id','target','site_eui','label'}]
        grouped={}
        for key,year in zip(keys,years):
            grouped.setdefault(key,[]).append(float(year))
        self.profiles_={key:sorted(value) for key,value in grouped.items() if len(set(value))==len(value)}
        rejected=len(grouped)-len(self.profiles_)
        self.fit_mask_=self._known(X)
        self.fit_positions_=np.flatnonzero(self.fit_mask_)
        self.model_,self.feature_,self.encoder_=None,None,None
        self.native_feature_columns_,self.constraints_=[],[]
        if len(self.fit_positions_)>=300:
            subset=self._expert_input(X.iloc[self.fit_positions_])
            self.feature_=FeatureComposition((29,30)).fit(subset)
            frame=self.feature_.transform(subset)
            self.encoder_=Encoder(native=True,scale=False).fit(frame)
            native=self.encoder_.transform(frame)
            self.native_feature_columns_=list(native.columns)
            if 'energy_star_rating' not in self.encoder_.numeric_:
                raise ValueError('Monotone rating must remain a numeric feature')
            self.constraints_=[-1 if column=='energy_star_rating' else 0 for column in native]
            estimator=LGBMRegressor(**self.parameters,monotone_constraints=self.constraints_)
            actual_config={**self.configuration(),'stage':'known_rating_actual_monotone_learner',
                'training_subset_rows':len(subset),'feature_columns':self.native_feature_columns_.copy(),
                'monotone_constraints':self.constraints_.copy()}
            if self.fit_hook is None:
                self.model_=estimator.fit(native,target[self.fit_positions_],categorical_feature='auto')
            else:
                self.model_=self.fit_hook(estimator,native,target[self.fit_positions_],actual_config,
                                          categorical_feature='auto')
            if self.model_ is None or not hasattr(self.model_,'predict'):
                raise ValueError('Actual known-subset learner hook must return a fitted predictor')
        self.keep_mask_=np.ones(len(X),dtype=bool)
        self.removed_positions_=np.empty(0,dtype=int)
        self.audit_={'fit_rows':len(X),'fit_original_rows':len(X),'fit_retained_rows':len(X),
            'expert_fit_rows':len(self.fit_positions_),'expert_fit_count':int(self.model_ is not None),
            'expert_fit_mask':self.fit_mask_.tolist(),'expert_fit_positions':self.fit_positions_.tolist(),
            'expert_fit_original_indices':X.index[self.fit_positions_].tolist(),
            'profile_count':len(grouped),'retained_profiles':len(self.profiles_),
            'rejected_ambiguous_profiles':rejected,'profile_index_target_used':False,
            'feature_audit':deepcopy(self.feature_.audit_) if self.feature_ is not None else None,
            'native_feature_columns':self.native_feature_columns_.copy(),
            'monotone_constraints':self.constraints_.copy(),
            'categorical_columns':list(self.encoder_.categorical_) if self.encoder_ is not None else [],
            'removed_rows':0,'target_outlier_deletion':False,'base_refit':False,
            'base_source':'caller_supplied_compatible_H70','configuration':self.configuration()}
        return self

    def predict(self,X):
        if not hasattr(self,'base_predictor_'):
            raise ValueError('Fit MonotoneUnmatchedRegressor before prediction')
        if not isinstance(X,pd.DataFrame) or list(X.columns)!=self.columns_:
            raise ValueError('Query raw feature schema must match training columns in order')
        predictor=self.base_predictor_.predict if hasattr(self.base_predictor_,'predict') else self.base_predictor_
        prediction=np.asarray(predictor(X),float).copy()
        if prediction.shape!=(len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Finite aligned compatible H70 predictions required')
        known=self._known(X)
        matched=np.asarray([any(past<year for past in self.profiles_.get(key,()))
            for key,year in zip(_keys(X),_years(X))],dtype=bool)
        positions=np.flatnonzero(known & ~matched)
        used=0
        if self.model_ is not None and len(positions):
            frame=self.feature_.transform(self._expert_input(X.iloc[positions]))
            native=self.encoder_.transform(frame)
            expert=np.asarray(self.model_.predict(native),float)
            if expert.shape!=(len(positions),) or not np.isfinite(expert).all():
                raise ValueError('Finite aligned monotone expert predictions required')
            prediction[positions]=.7*np.maximum(expert,0.)+.3*prediction[positions]
            used=len(positions)
        self.last_prediction_audit_={'query_rows':len(X),'query_known_rating_rows':int(known.sum()),
            'query_matched_past_rows':int(matched.sum()),'eligible_unmatched_known_rows':len(positions),
            'expert_applied_rows':used,'matched_or_missing_unchanged':True,'query_statistics':False}
        return prediction
