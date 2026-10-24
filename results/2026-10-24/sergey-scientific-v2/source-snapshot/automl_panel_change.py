"""Causal first-difference covariates, registered H83. No query labels."""
import numpy as np
import pandas as pd
from automl_panel_memory import PanelHistory,panel_keys,PANEL_FIELDS
class PanelChangeForecaster:
 def __init__(self,threads=4):self.threads=threads
 def _frame(self,X):
  temporal,matched=self.history_.features(X)
  keys=panel_keys(X,PANEL_FIELDS);years=X.Year_Factor.to_numpy(float)
  last=np.full((len(X),len(self.numeric_)),np.nan);gaps=np.ones(len(X))
  for i,(key,year) in enumerate(zip(keys,years)):
   record=self.past_x_.get(key)
   if record is None:continue
   past=record[record[:,0]<year]
   if not len(past):continue
   latest=past[-1];gaps[i]=year-latest[0];last[i]=latest[1:]
  current=X[self.numeric_].replace([np.inf,-np.inf],np.nan).to_numpy(float)
  extra=pd.DataFrame(np.column_stack([last,(current-last)/gaps[:,None]]),index=X.index,columns=['__last_x_'+c for c in self.numeric_]+['__annual_dx_'+c for c in self.numeric_])
  frame=pd.concat([X.drop(columns=['Year_Factor']),temporal,extra],axis=1)
  assert frame.columns.is_unique
  return frame,temporal,matched,gaps
 def fit(self,X,y):
  from lightgbm import LGBMRegressor
  from automl_models import Encoder
  from automl_combo_memory import fit_once
  self.history_=PanelHistory().fit(X,y)
  self.numeric_=[c for c in X.select_dtypes(include='number') if c!='Year_Factor']
  data=X[self.numeric_].replace([np.inf,-np.inf],np.nan).copy();data['__key']=panel_keys(X,PANEL_FIELDS);data['__year']=X.Year_Factor.to_numpy(float)
  moments=data.groupby(['__key','__year'],sort=False)[self.numeric_].mean().reset_index()
  self.past_x_={key:part.sort_values('__year')[['__year']+self.numeric_].to_numpy(float) for key,part in moments.groupby('__key',sort=False)}
  frame,temporal,matched,gaps=self._frame(X)
  if matched.sum()<500:raise ValueError('Insufficient strictly causal covariate pairs')
  self.encoder_=Encoder(native=True,scale=False);values=self.encoder_.fit_transform(frame.loc[matched])
  target=(np.asarray(y,float)[matched]-temporal.loc[matched,'__panel_last'].to_numpy())/gaps[matched]
  settings=dict(n_estimators=600,learning_rate=.03,num_leaves=15,min_child_samples=40,reg_lambda=10.,verbosity=-1,random_state=42,n_jobs=self.threads,force_col_wise=True)
  self.model_=fit_once(LGBMRegressor(**settings),values,target,{'strategy':'causal_covariate_first_differences','parameters':settings,'columns':list(frame.columns),'target':'annual_current_minus_last_strict_past_year_mean'})
  self.nonnegative_=bool((np.asarray(y)>=0).all());self.audit_={**self.history_.audit_,'strategy':'causal_covariate_first_differences','causal_rows':int(matched.sum()),'numeric_past_and_delta_fields':self.numeric_,'parameters':settings,'validation_targets_used':False,'year_gap_scaling':True}
  return self
 def predict_with_fallback(self,X,fallback):
  frame,temporal,matched,gaps=self._frame(X);pred=np.asarray(fallback,float).copy()
  if pred.shape!=(len(X),):raise ValueError('Invalid fallback shape')
  if matched.any():
   pred[matched]=temporal.loc[matched,'__panel_last'].to_numpy()+gaps[matched]*self.model_.predict(self.encoder_.transform(frame.loc[matched]))
   if self.nonnegative_:pred[matched]=np.maximum(0,pred[matched])
  return pred
