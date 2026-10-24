"""Past-only profile retrieval, with a fitted feature-expert fallback."""
import json
import numpy as np
import pandas as pd

PROFILE=['State_Factor','building_class','facility_type','floor_area','year_built','ELEVATION']
NUMERIC={'floor_area','year_built','ELEVATION'}
def profile_keys(X):
    missing=set(PROFILE)-set(X.columns)
    if missing:raise ValueError('Missing profile fields: '+str(sorted(missing)))
    columns=[]
    for c in PROFILE:
        if c in NUMERIC:
            values=pd.to_numeric(X[c],errors='coerce')
            columns.append(values.map(lambda v:'M:' if pd.isna(v) or not np.isfinite(v) else 'N:'+format(float(v),'.17g')).to_numpy())
        else:columns.append(('V:'+X[c].astype('string')).fillna('M:').to_numpy())
    return [json.dumps(list(row),ensure_ascii=False,separators=(',',':')) for row in zip(*columns)]

class ProfileHistory:
    def fit(self,X,y):
        if 'Year_Factor' not in X:raise ValueError('Explicit Year_Factor required')
        years=pd.to_numeric(X['Year_Factor'],errors='raise').to_numpy(float);y=np.asarray(y,float)
        if not np.isfinite(years).all() or y.shape!=(len(X),) or not np.isfinite(y).all():raise ValueError('Invalid profile training data')
        keys=profile_keys(X);self.records_={}
        for key,year,target in zip(keys,years,y):self.records_.setdefault(key,[]).append((year,target))
        self.records_={k:np.asarray(v,float) for k,v in self.records_.items()}
        frame=pd.DataFrame({'profile':keys,'year':years})
        counts=frame.groupby(['profile','year']).size()
        self.audit_={'fit_original_rows':len(X),'fit_retained_rows':len(X),'profile_fields':PROFILE,
          'profile_count':len(self.records_),'same_profile_year_multiple_rows':int((counts>1).sum()),
          'profile_is_proven_building_identity':False,'strict_past_only':True,'id_or_weather_as_key':False,
          'age_decay':.5,'maximum_memory_weight':.9,'staleness_decay':.25}
        return self
    def lookup(self,X):
        if 'Year_Factor' not in X:raise ValueError('Explicit query year required')
        years=pd.to_numeric(X['Year_Factor'],errors='raise').to_numpy(float)
        if not np.isfinite(years).all():raise ValueError('Invalid query year')
        mean=np.zeros(len(X));alpha=np.zeros(len(X));past_counts=np.zeros(len(X),int)
        for i,(key,year) in enumerate(zip(profile_keys(X),years)):
            records=self.records_.get(key)
            if records is None:continue
            past=records[records[:,0]<year]
            if not len(past):continue
            latest=past[:,0].max();weights=.5**(latest-past[:,0])
            mean[i]=np.average(past[:,1],weights=weights);past_counts[i]=len(past)
            alpha[i]=min(.9,len(past)/(len(past)+1))*np.exp(-.25*max(0,year-latest-1))
        return mean,alpha,{'query_rows':len(X),'matched_rows':int((alpha>0).sum()),'matched_fraction':float((alpha>0).mean()),
             'memory_weight_mean':float(alpha.mean()),'past_rows_sum':int(past_counts.sum()),'no_query_targets_used':True}

class TemporalProfileModel:
    def __init__(self,candidate=None,threads=4):self.candidate=dict(candidate or {});self.threads=threads
    def fit(self,X,y):
        from automl_combo_memory import fit_once
        from automl_reg_combo import FeatureGroupExperts
        c={'strategy':'reg_feature_experts','feature_numbers':[29,30,31],'group_column':'building_class',
           'minimum_expert_rows':500,'shrinkage':500.,'parameters':self.candidate.get('parameters',{})}
        self.base_=FeatureGroupExperts(c,self.threads).fit(X,y)
        self.memory_=fit_once(ProfileHistory(),X,y,{'strategy':'temporal_profile_memory','version':1,'profile':PROFILE,
                         'strict_past_only':True,'age_decay':.5,'cap':.9,'staleness_decay':.25})
        self.audit_={**self.memory_.audit_,'strategy':'temporal_profile_memory','base':self.base_.audit_}
        self.keep_mask_=np.ones(len(X),bool);self.removed_positions_=np.array([],int);return self
    def predict(self,X):
        mean,alpha,_=self.memory_.lookup(X);return alpha*mean+(1-alpha)*self.base_.predict(X)

def evaluate_memory(X,y,Xvalid,baseline):
    from automl_combo_memory import fit_once
    model=fit_once(ProfileHistory(),X,y,{'strategy':'temporal_profile_memory','version':1,'profile':PROFILE,
                      'strict_past_only':True,'age_decay':.5,'cap':.9,'staleness_decay':.25})
    mean,alpha,audit=model.lookup(Xvalid)
    baseline=np.asarray(baseline,float)
    if baseline.shape!=(len(Xvalid),) or not np.isfinite(baseline).all():raise ValueError('Invalid saved baseline predictions')
    return {'prediction':alpha*mean+(1-alpha)*baseline,'audit':{**model.audit_,**audit,'baseline':'read_only_H43_outer_predictions'},
            'learned_state':{},'memory_mean':mean,'memory_weight':alpha}
