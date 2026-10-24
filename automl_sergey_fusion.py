"""Sergey causal dynamics plus Ivan transport, weights learned on current train."""
from copy import deepcopy
import numpy as np
import pandas as pd
from fusion_weights import fit_simplex, combine


def canonical_fusion(candidate):
    result=deepcopy(candidate)
    if (result.get('strategy')!='sergey_ivan_fusion' or result.get('family')!='extra_trees'
            or result.get('parameters',{}) or result.get('feature_code','').strip()):
        raise ValueError('Fusion has fixed family, components and features')
    forbidden={'pool_weights','blend_weights','meta_state','weights','class_multipliers'}
    if forbidden.intersection(result):
        raise ValueError('Fusion weights must be learned on the current task')
    if result.get('target_transform') not in [None,'none']:
        raise ValueError('Each fusion component owns its target transform')
    result.update(parameters={},preprocessing='sergey-ivan-fusion-v1',scale=False,
                  max_outlier_fraction=0,target_transform='none')
    return result


class _TransportFit:
    def __init__(self,model,base):self.model,self.base=model,base
    def fit(self,X,y):return self.model.fit(X,y,base_predictor=self.base)


class _SimplexFit:
    def fit(self,X,y):self.weights_=fit_simplex(X,y);return self


class SergeyIvanFusion:
    """One temporal train split for weights, followed by fresh full-training fit."""
    def __init__(self,candidate=None,threads=4):
        candidate=candidate or {'strategy':'sergey_ivan_fusion','family':'extra_trees'}
        self.candidate=canonical_fusion(candidate)
        if isinstance(threads,bool) or not isinstance(threads,(int,np.integer)) or not 1<=threads<=4:
            raise ValueError('One to four CPU threads required')
        self.threads=int(threads)

    def _fit_components(self,X,y,active=None):
        from automl_combo_memory import fit_once
        from automl_panel_memory import PanelHybridModel
        from automl_temporal_memory import TemporalProfileModel,ProfileHistory
        from sprint_regression import TemporalEntityLinker
        from sprint_ratio_transport import OriginalLossTransport
        from sprint_uncertainty import ConditionalLognormalRegressor
        active=np.ones(4,dtype=bool) if active is None else np.asarray(active,bool)
        # One H43 fallback shared by both branches, as in the measured graph.
        panel=PanelHybridModel({'parameters':{}},self.threads).fit(X,y)
        h69=TemporalProfileModel({'parameters':{}},self.threads)
        h69.base_=panel.base_
        h69.memory_=fit_once(ProfileHistory(),X,y,{'strategy':'fusion_profile_history','version':1})
        h69.audit_={**h69.memory_.audit_,'base':deepcopy(panel.base_.audit_)}
        models=[panel,None,None,None]
        for i,model in [(1,TemporalEntityLinker(alpha=.95)),(2,OriginalLossTransport())]:
            if active[i]:
                models[i]=fit_once(_TransportFit(model,h69),X,y,{
                    'strategy':'fusion_transport','component':model.configuration()})
        if active[3]:
            uncertainty=ConditionalLognormalRegressor(threads=min(2,self.threads))
            models[3]=fit_once(uncertainty,X,y,uncertainty.configuration())
        return models

    @staticmethod
    def _matrix(models,X):
        matrix=np.zeros((len(X),4))
        for i,model in enumerate(models):
            if model is not None:matrix[:,i]=np.asarray(model.predict(X),float)
        if not np.isfinite(matrix).all():raise ValueError('Nonfinite fusion component prediction')
        return matrix

    def fit(self,X,y):
        from automl_combo_memory import fit_once
        target=np.asarray(y,float)
        required={'Year_Factor','State_Factor','building_class','facility_type','floor_area',
                  'year_built','energy_star_rating','ELEVATION'}
        if (not isinstance(X,pd.DataFrame) or not len(X) or not X.columns.is_unique
                or not required.issubset(X.columns) or target.shape!=(len(X),)
                or not np.isfinite(target).all() or (target<0).any()):
            raise ValueError('Fusion needs finite nonnegative aligned panel data')
        years=pd.to_numeric(X['Year_Factor'],errors='raise').to_numpy(float)
        if not np.isfinite(years).all():raise ValueError('Finite temporal periods required')
        self.columns_=list(X.columns)
        last=float(years.max());fit=np.flatnonzero(years<last);valid=np.flatnonzero(years==last)
        if len(np.unique(years[fit]))<2 or len(fit)<1000 or len(valid)<100:
            # The selected portfolio falls back safely when temporal fitting is infeasible.
            self.weights_=np.array([1.,0.,0.,0.]);self.selection_={'status':'insufficient_temporal_training'}
        else:
            parents=self._fit_components(X.iloc[fit],target[fit])
            matrix=self._matrix(parents,X.iloc[valid])
            calibration=fit_once(_SimplexFit(),matrix,target[valid],{
                'strategy':'fusion_train_only_simplex','components':['SergeyH73','IvanH70','IvanH83','IvanH85'],
                'maxiter':200,'ftol':1e-12,'split':'latest_train_period'})
            self.weights_=calibration.weights_.copy()
            self.weights_[self.weights_<1e-8]=0.
            self.weights_/=self.weights_.sum()
            self.selection_={'status':'trained','prefix_rows':len(fit),'weight_fit_rows':len(valid),
                'maximum_prefix_period':float(years[fit].max()),'weight_period':last,
                'external_validation_targets_used':False,'public_cached_state_used':False}
            del parents,matrix
        self.models_=self._fit_components(X,target,self.weights_>0)
        self.keep_mask_=np.ones(len(X),bool);self.removed_positions_=np.empty(0,int)
        self.audit_={'strategy':'sergey_ivan_fusion','fit_original_rows':len(X),
            'fit_retained_rows':len(X),'row_deletion':False,'weights':self.weights_.tolist(),
            'weight_selection':self.selection_,'training_only':True,'embedded_learned_state':False,
            'query_targets_used':False,'full_fit_components':{
                name:deepcopy(model.audit_) for name,model in zip(
                    ['SergeyH73','IvanH70','IvanH83','IvanH85'],self.models_) if model is not None}}
        return self

    def predict(self,X):
        if not hasattr(self,'models_'):raise ValueError('Fit fusion before prediction')
        if not isinstance(X,pd.DataFrame) or list(X.columns)!=self.columns_:
            raise ValueError('Fusion query schema mismatch')
        return combine(self._matrix(self.models_,X),self.weights_)
