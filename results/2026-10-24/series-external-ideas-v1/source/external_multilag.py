"""Independent causal multi-lag representation inspired by external research."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd

FIELDS=('State_Factor','building_class','facility_type','floor_area','year_built')


def keys(X):
    if not set(FIELDS).issubset(X):raise ValueError('Physical profile fields required')
    result=[]
    for values in X.loc[:,FIELDS].itertuples(index=False,name=None):
        row=[];valid=True
        for name,value in zip(FIELDS,values):
            if pd.isna(value):valid=False;break
            if name in ['floor_area','year_built']:
                value=pd.to_numeric(value,errors='coerce')
                if not np.isfinite(value) or (name=='year_built' and value==0):valid=False;break
                row.append(float(value))
            else:row.append(str(value))
        result.append(tuple(row) if valid else None)
    return result


def years(X):
    out=pd.to_numeric(X['Year_Factor'],errors='raise').to_numpy(float)
    if not np.isfinite(out).all():raise ValueError('Finite periods required')
    return out


class CausalMultiLag:
    def fit(self,X,y):
        y=np.asarray(y,float)
        if not isinstance(X,pd.DataFrame) or len(X)!=len(y) or not len(y) or not np.isfinite(y).all():
            raise ValueError('Aligned nonempty training frame and finite targets required')
        if any(str(c).startswith('__ml_') for c in X):raise ValueError('Reserved feature prefix')
        self.columns_=list(X.columns)
        self.numeric_=[c for c in X if pd.api.types.is_numeric_dtype(X[c]) and c not in FIELDS and c!='Year_Factor']
        self.numeric_=[c for c in self.numeric_ if str(c).lower() not in ['id','target','site_eui','label']]
        values=X[self.numeric_].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(float)
        grouped={}
        for i,(key,year) in enumerate(zip(keys(X),years(X))):
            if key is not None:grouped.setdefault(key,[]).append((float(year),float(y[i]),i))
        self.records_={};collisions=0
        for key,rows in grouped.items():
            if len({r[0] for r in rows})!=len(rows):collisions+=1;continue
            self.records_[key]=sorted(rows)
        pairs=[(a[2],b[2]) for rows in self.records_.values() for a,b in zip(rows[:-1],rows[1:])]
        selected=[]
        if pairs:
            a,b=np.asarray(pairs,int).T
            for j,column in enumerate(self.numeric_):
                old,new=values[a,j],values[b,j];ok=np.isfinite(old)&np.isfinite(new)
                if ok.sum()<100:continue
                rate=float(np.mean(old[ok]!=new[ok]))
                if not .01<=rate<=.99:continue
                corr=pd.Series(values[:,j]).corr(pd.Series(y),method='spearman')
                if np.isfinite(corr):selected.append((-abs(float(corr)),str(column),column))
        self.changing_columns_=[r[2] for r in sorted(selected)[:3]]
        self.selected_values_={c:pd.to_numeric(X[c],errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(float) for c in self.changing_columns_}
        self.audit_={'fit_rows':len(X),'key_fields':list(FIELDS),'retained_profiles':len(self.records_),
            'collision_profiles_rejected':collisions,'changing_columns':self.changing_columns_.copy(),
            'selection_source':'fit_only','strict_past_only':True,'rows_deleted':0,'query_statistics':False}
        return self

    def transform(self,X):
        if list(X.columns)!=self.columns_:raise ValueError('Raw feature schema mismatch')
        out=X.drop(columns=['Year_Factor']).reset_index(drop=True).copy()
        names=['lag1','lag2','lag3','last','mean','std','min','max','count','gap','trend']
        names+=sum(([c+'_prev',c+'_diff'] for c in self.changing_columns_),[])
        features=np.full((len(X),len(names)),np.nan)
        features[:,8]=0.
        for i,(key,year) in enumerate(zip(keys(X),years(X))):
            if key is None:continue
            past=[row for row in self.records_.get(key,[]) if row[0]<year]
            if not past:continue
            moment={row[0]:row for row in past};targets=np.array([r[1] for r in past])
            row=[moment[year-lag][1] if year-lag in moment else np.nan for lag in [1,2,3]]
            row.extend([targets[-1],targets.mean(),targets.std(),targets.min(),targets.max(),len(targets),year-past[-1][0],
                        targets[-1]-targets[-2] if len(targets)>1 else np.nan])
            donor=moment.get(year-1)
            for c in self.changing_columns_:
                old=self.selected_values_[c][donor[2]] if donor else np.nan
                current=pd.to_numeric(X[c].iloc[i],errors='coerce')
                row.extend([old,float(current)-old])
            features[i]=row
        for j,name in enumerate(names):out['__ml_'+name]=features[:,j]
        return out


class NativeFrame:
    def fit(self,X):
        self.columns_=list(X.columns)
        self.categories_={c:sorted(X[c].dropna().astype(str).unique()) for c in X
                          if not pd.api.types.is_numeric_dtype(X[c])}
        return self

    def transform(self,X):
        if list(X.columns)!=self.columns_:raise ValueError('Feature schema mismatch')
        out=X.copy()
        for c in out:
            if c in self.categories_:
                out[c]=pd.Categorical(out[c].astype('string'),categories=self.categories_[c])
            else:out[c]=pd.to_numeric(out[c],errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
        return out


class MultiLagRegressor:
    def __init__(self,threads=4):
        if threads not in [1,2,3,4]:raise ValueError('CPU bound')
        self.parameters=dict(n_estimators=1200,learning_rate=.03,num_leaves=63,min_child_samples=20,
            colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_lambda=5.,
            objective='regression',random_state=42,n_jobs=threads,verbosity=-1,force_col_wise=True)

    def configuration(self):
        return {'hypothesis':'HYP-105','strategy':'causal_exact_multilag_native_lgb','parameters':deepcopy(self.parameters),
            'target_transform':'none','lags':[1,2,3],'history_collision':'reject_whole_profile',
            'native_nan':True,'changing_columns':'top3_fit_Spearman_rate01_99_min100pairs',
            'early_stopping':False,'query_statistics':False,'source_idea':'hilhool17bcb1_exactlags_independent_implementation'}

    def fit(self,X,y):
        from lightgbm import LGBMRegressor
        target=np.asarray(y,float)
        if not np.isfinite(target).all() or np.any(target<0):raise ValueError('Nonnegative target required')
        self.history_=CausalMultiLag().fit(X,target)
        frame=self.history_.transform(X)
        self.encoder_=NativeFrame().fit(frame)
        self.model_=LGBMRegressor(**self.parameters).fit(self.encoder_.transform(frame),target,categorical_feature='auto')
        self.audit_={'configuration':self.configuration(),'history':self.history_.audit_,
            'feature_names':list(frame),'categorical_columns':list(self.encoder_.categories_),
            'fit_original_rows':len(X),'fit_retained_rows':len(X),'no_validation_target':True}
        self.keep_mask_=np.ones(len(X),bool);self.removed_positions_=np.array([],int)
        return self

    def predict(self,X):
        return np.maximum(0.,self.model_.predict(self.encoder_.transform(self.history_.transform(X))))
