"""Portable row-wise generated features and train-only encoders."""
import ast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline


class FeatureCode(TransformerMixin,BaseEstimator):
    def __init__(self,code=''): self.code=code
    def fit(self,X,y=None):
        self.transform(X.iloc[:min(3,len(X))]); return self
    def transform(self,X):
        if not self.code.strip(): return X.copy()
        tree=ast.parse(self.code)
        forbidden={'open','eval','exec','compile','__import__','getattr','setattr','globals','locals','input','breakpoint','exit','quit'}
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom,ast.While,ast.With,ast.AsyncWith,ast.Global,ast.Nonlocal)):
                raise ValueError('Feature code must be bounded row-wise Python without imports or I/O')
            if isinstance(node,ast.Name) and (node.id in forbidden or node.id.startswith('__')): raise ValueError('Forbidden feature operation')
            if isinstance(node,ast.Attribute):
                allowed={'copy','columns','index','shape','iloc','loc','dtypes','select_dtypes','replace','fillna','isna','astype','clip','where','mask','isin','notna','items','dtype','str','lower','upper','len','contains','get','to_numpy','log1p','sqrt','abs','maximum','minimum','nan','inf','log','exp','sign','mean','median','std','var','quantile','min','max','sum','drop'}
                if node.attr not in allowed: raise ValueError('Feature attribute outside approved row-wise API: '+node.attr)
                if node.attr.startswith(('_','read_','to_')) or node.attr in ['eval','query','load','save','system','popen','fit','fit_transform']:
                    raise ValueError('Feature code cannot perform I/O or learned transformations')
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['mean','median','std','var','quantile','min','max','sum']:
                if not any(k.arg=='axis' and isinstance(k.value,ast.Constant) and k.value.value==1 for k in node.keywords):
                    raise ValueError('Generated statistics must be row-wise (axis=1)')
        builtins={k:v for k,v in vars(__import__('builtins')).items() if k in ['len','range','zip','min','max','abs','float','int','str','list','dict','tuple','set','enumerate']}
        namespace={'np':np,'pd':pd,'__builtins__':builtins}
        exec(compile(tree,'<generated-row-features>','exec'),namespace)
        if not callable(namespace.get('engineer')): raise ValueError('Feature code must define engineer(X)')
        output=namespace['engineer'](X.copy())
        if not isinstance(output,pd.DataFrame) or not output.index.equals(X.index) or len(output)!=len(X) or not output.columns.is_unique:
            raise ValueError('Feature code must preserve rows/index and return a DataFrame')
        return output


class Encoder(TransformerMixin,BaseEstimator):
    def __init__(self,native=False,scale=False): self.native=native; self.scale=scale
    def fit(self,X,y=None):
        X=X.replace([np.inf,-np.inf],np.nan)
        self.numeric_=X.select_dtypes(include='number').columns.tolist()
        self.categorical_=[c for c in X if c not in self.numeric_]
        self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
        a=self.imputer_.fit_transform(X[self.numeric_]) if self.numeric_ else np.empty((len(X),0))
        self.scaler_=StandardScaler().fit(a) if self.scale and len(self.numeric_) else None
        self.levels_={c:sorted(X[c].dropna().astype(str).unique()) for c in self.categorical_}
        if self.categorical_ and not self.native:
            self.ohe_=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64)
            self.ohe_.fit(X[self.categorical_].astype('string').fillna('__MISSING__'))
        return self
    def transform(self,X):
        X=X.replace([np.inf,-np.inf],np.nan)
        a=self.imputer_.transform(X[self.numeric_]) if self.numeric_ else np.empty((len(X),0))
        if self.scaler_ is not None: a=self.scaler_.transform(a)
        if self.native:
            out=pd.DataFrame(a,index=X.index,columns=self.numeric_)
            for c in self.categorical_: out[c]=pd.Categorical(X[c].astype('string'),categories=self.levels_[c])
            return out
        if self.categorical_:
            a=np.concatenate([a,self.ohe_.transform(X[self.categorical_].astype('string').fillna('__MISSING__'))],axis=1)
        return a if self.scale else a.astype(np.float32)


def build_pipeline(kind,candidate,threads=4):
    family=candidate['family']; regression=kind=='regression'; p=dict(candidate.get('parameters',{}))
    for k in ['random_state','random_seed','seed','n_jobs','thread_count','verbose']: p.pop(k,None)
    allowed_parameters={'n_estimators','iterations','learning_rate','num_leaves','min_child_samples','reg_lambda','reg_alpha','max_depth','min_samples_leaf','max_features','depth','l2_leaf_reg','subsample','colsample_bytree','class_weight','criterion'}
    if set(p)-allowed_parameters: raise ValueError('Unsupported model parameters')
    for key,upper in [('n_estimators',1200),('iterations',1200),('max_depth',32),('depth',10),('num_leaves',128)]:
        if key in p and (not isinstance(p[key],int) or not 1<=p[key]<=upper): raise ValueError('Parameter exceeds worker resource bounds: '+key)
    if family=='lightgbm':
        from lightgbm import LGBMRegressor,LGBMClassifier
        settings=dict(n_estimators=500,learning_rate=.05,num_leaves=31,min_child_samples=20,reg_lambda=1)
        settings.update(p); model=(LGBMRegressor if regression else LGBMClassifier)(**settings,random_state=42,n_jobs=threads,verbosity=-1)
    elif family in ['extra_trees','random_forest']:
        from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier,RandomForestRegressor,RandomForestClassifier
        cls=(ExtraTreesRegressor if regression else ExtraTreesClassifier) if family=='extra_trees' else (RandomForestRegressor if regression else RandomForestClassifier)
        settings=dict(n_estimators=200 if family=='extra_trees' else 150,max_depth=24 if family=='extra_trees' else 20,min_samples_leaf=2,max_features=.8)
        settings.update(p); model=cls(**settings,random_state=42,n_jobs=threads)
    elif family=='catboost':
        from catboost import CatBoostRegressor,CatBoostClassifier
        settings=dict(iterations=500,depth=6,learning_rate=.05,l2_leaf_reg=3)
        settings.update(p); model=(CatBoostRegressor if regression else CatBoostClassifier)(**settings,random_seed=42,thread_count=threads,verbose=False,allow_writing_files=False)
    else: raise ValueError('Unsupported model family')
    encoder=Encoder(native=family=='lightgbm',scale=candidate.get('scale',False))
    if candidate.get('preprocessing')=='legacy_standard':
        from sklearn.compose import ColumnTransformer,make_column_selector
        encoder=ColumnTransformer([
            ('numeric',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),make_column_selector(dtype_include=np.number)),
            ('categorical',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='missing')),('encode',OneHotEncoder(handle_unknown='ignore'))]),make_column_selector(dtype_exclude=np.number))])
    return Pipeline([('features',FeatureCode(candidate.get('feature_code',''))),('encode',encoder),('model',model)])


def predict_for_metric(model,X,metric):
    if metric in ['roc_auc','log_loss']:
        p=model.predict_proba(X)
        return p[:,-1] if metric=='roc_auc' and p.shape[1]==2 else p
    return np.asarray(model.predict(X)).reshape(-1)
