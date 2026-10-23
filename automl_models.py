"""Portable row-wise generated features and train-only encoders."""
import ast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def canonical_candidate(candidate):
    result=dict(candidate)
    result['parameters']=dict(candidate.get('parameters',{}))
    strategy=result.get('strategy')
    if strategy in {'reg_feature_experts','reg_convex_blend','target_tail_mixture','smote_macro_f1','frequency_blend_macro_f1','probability_stacking','temporal_profile_memory'}:
        from automl_combo_policy import canonical_combo
        return canonical_combo(result)
    if strategy in ['group_experts','macro_f1_calibrated']:
        if result.get('feature_code','').strip(): raise ValueError('Advanced strategies use their versioned preprocessing without generated features')
        if strategy=='group_experts':
            if result.get('family')!='extra_trees': raise ValueError('Group experts require ExtraTrees')
            if result.get('target_transform') not in [None,'log1p']: raise ValueError('Group experts require log1p target')
            result.update(preprocessing='group-experts-v1',scale=False,max_outlier_fraction=0,target_transform='log1p')
            result.setdefault('group_column','building_class'); result.setdefault('minimum_expert_rows',500); result.setdefault('shrinkage',500.)
            if not isinstance(result['group_column'],str) or not result['group_column']: raise ValueError('Group column must be named')
            if isinstance(result['minimum_expert_rows'],bool) or not isinstance(result['minimum_expert_rows'],int) or result['minimum_expert_rows']<500: raise ValueError('Expert sample minimum must be at least 500')
            if isinstance(result['shrinkage'],bool) or not isinstance(result['shrinkage'],(int,float)) or not np.isfinite(result['shrinkage']) or result['shrinkage']<=0: raise ValueError('Invalid expert shrinkage')
        else:
            if result.get('family')!='xgboost': raise ValueError('Macro-F1 calibration requires XGBoost')
            if result.get('target_transform') not in [None,'none']: raise ValueError('Calibration has no target transform')
            result.update(preprocessing='macro-f1-v1',scale=False,max_outlier_fraction=0,target_transform='none')
            result.setdefault('calibration_folds',3)
            if result['calibration_folds']!=3: raise ValueError('Calibration requires the versioned three-fold protocol')
            multipliers=result.get('class_multipliers')
            if multipliers is not None:
                if not isinstance(multipliers,dict) or not multipliers: raise ValueError('Class multipliers must be a nonempty label mapping')
                if any(not isinstance(k,str) or isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for k,v in multipliers.items()): raise ValueError('Class multipliers must be finite and positive')
                result['class_multipliers']={k:float(v) for k,v in sorted(multipliers.items())}
    elif strategy not in [None,'','full']:
        raise ValueError('Unsupported strategy')
    else:
        result.update(preprocessing='full-v1',scale=True,max_outlier_fraction=.05)
    if result.get('target_transform') not in [None,'none','log1p']: raise ValueError('Unsupported target transformation')
    return result


class PreparedModel(BaseEstimator):
    """Mandatory train-only preparation, bounded row filtering and invertible labels."""
    def __init__(self,estimator,kind='regression',native=False,target_transform=None,feature_code=''):
        self.estimator=estimator; self.kind=kind; self.native=native
        self.target_transform=target_transform; self.feature_code=feature_code
    def fit(self,X,y):
        from automl_preprocessing import FullPreprocessor,TrainingOutlierFilter
        target=np.asarray(y)
        if self.kind=='regression' and self.target_transform=='log1p' and (np.any(target<=-1) or not np.isfinite(target).all()): raise ValueError('Invalid log1p target domain')
        self.features_=FeatureCode(self.feature_code).fit(X)
        featured=self.features_.transform(X)
        self.outlier_filter_=TrainingOutlierFilter()
        kept,target,self.keep_mask_=self.outlier_filter_.fit_filter(featured,target,classification=self.kind=='classification')
        self.preprocessor_=FullPreprocessor(native=self.native)
        values=self.preprocessor_.fit_transform(kept,target)
        if self.kind=='classification':
            self.labels_=LabelEncoder().fit(target); self.classes_=self.labels_.classes_; target=self.labels_.transform(target)
        elif self.target_transform=='log1p': target=np.log1p(target)
        self.estimator_=clone(self.estimator).fit(values,target)
        self.removed_positions_=np.flatnonzero(~self.keep_mask_)
        numeric_values=values.select_dtypes(include='number').to_numpy(dtype=float) if isinstance(values,pd.DataFrame) else np.asarray(values[:,:len(self.preprocessor_.numeric_output_columns_)],dtype=float)
        deviations=numeric_values.std(axis=0) if numeric_values.shape[1] else np.empty(0)
        mean_error=float(np.max(np.abs(numeric_values.mean(axis=0)))) if numeric_values.shape[1] else 0.
        std_error=float(np.max(np.abs(deviations[deviations>1e-8]-1))) if np.any(deviations>1e-8) else 0.
        self.audit_={'policy':'full-v1','fit_original_rows':len(X),'fit_retained_rows':int(self.keep_mask_.sum()),
                     'target_transform':self.target_transform,'outliers':self.outlier_filter_.audit_,'preprocessing':self.preprocessor_.audit_,
                     'normalized_training_mean_abs_max':mean_error,'normalized_training_std_error_max':std_error}
        return self
    def _values(self,X): return self.preprocessor_.transform(self.features_.transform(X))
    def predict(self,X):
        prediction=np.asarray(self.estimator_.predict(self._values(X))).reshape(-1)
        if self.kind=='classification': return self.labels_.inverse_transform(prediction.astype(int))
        if self.target_transform=='log1p': prediction=np.expm1(prediction)
        if not np.isfinite(prediction).all(): raise ValueError('Nonfinite predictions')
        return prediction
    def predict_proba(self,X):
        if self.kind!='classification': raise ValueError('Regression has no class probabilities')
        p=np.asarray(self.estimator_.predict_proba(self._values(X)))
        if not np.isfinite(p).all(): raise ValueError('Nonfinite class scores')
        return p


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
    candidate=canonical_candidate(candidate)
    if candidate.get('strategy')=='temporal_profile_memory':
        if kind!='regression':raise ValueError('Temporal profile memory requires regression')
        from automl_temporal_memory import TemporalProfileModel
        return TemporalProfileModel(candidate,threads)
    if candidate.get('strategy') in {'reg_feature_experts','reg_convex_blend','target_tail_mixture'}:
        if kind!='regression':raise ValueError('Regression strategy cannot solve classification')
        from automl_reg_combo import build_reg_combo
        return build_reg_combo(candidate,threads)
    if candidate.get('strategy') in {'smote_macro_f1','frequency_blend_macro_f1','probability_stacking'}:
        if kind!='classification':raise ValueError('Classification strategy cannot solve regression')
        from automl_class_combo import build_class_combo
        return build_class_combo(candidate,threads)
    if candidate.get('strategy') in ['group_experts','macro_f1_calibrated']:
        from automl_advanced import build_advanced
        return build_advanced(kind,candidate,threads)
    family=candidate['family']; regression=kind=='regression'; p=dict(candidate.get('parameters',{}))
    for k in ['random_state','random_seed','seed','n_jobs','thread_count','verbose']: p.pop(k,None)
    allowed_parameters={'n_estimators','iterations','learning_rate','num_leaves','min_child_samples','reg_lambda','reg_alpha','max_depth','min_samples_leaf','max_features','depth','l2_leaf_reg','subsample','colsample_bytree','class_weight','criterion','tree_method','device'}
    if set(p)-allowed_parameters: raise ValueError('Unsupported model parameters')
    for key,upper in [('n_estimators',1200),('iterations',1200),('max_depth',32),('depth',10),('num_leaves',128)]:
        if key in p and (not isinstance(p[key],int) or not 1<=p[key]<=upper): raise ValueError('Parameter exceeds worker resource bounds: '+key)
    if family=='xgboost':
        from xgboost import XGBRegressor,XGBClassifier
        if p.get('device','cpu')!='cpu': raise ValueError('Only CPU training is permitted')
        settings=dict(n_estimators=600,max_depth=6,learning_rate=.05,reg_lambda=1,tree_method='hist',device='cpu')
        settings.update(p); model=(XGBRegressor if regression else XGBClassifier)(**settings,random_state=42,n_jobs=threads)
    elif family=='lightgbm':
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
    return PreparedModel(model,kind=kind,native=family=='lightgbm',target_transform=candidate.get('target_transform'),feature_code=candidate.get('feature_code',''))


def predict_for_metric(model,X,metric):
    if metric in ['roc_auc','log_loss']:
        p=model.predict_proba(X)
        return p[:,-1] if metric=='roc_auc' and p.shape[1]==2 else p
    return np.asarray(model.predict(X)).reshape(-1)
