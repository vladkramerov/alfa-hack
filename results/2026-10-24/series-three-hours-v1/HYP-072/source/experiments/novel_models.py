"""Frozen SPEC-004 candidates and model mechanics, without implicit searches."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from automl_models import Encoder


LGB_PARAMS=dict(n_estimators=500,learning_rate=.05,num_leaves=31,min_child_samples=20,reg_lambda=1,random_state=42,n_jobs=4,verbosity=-1)
ET_PARAMS=dict(n_estimators=200,max_depth=24,min_samples_leaf=2,max_features=.8,random_state=42,n_jobs=4)
XGB_PARAMS=dict(n_estimators=600,max_depth=6,learning_rate=.05,reg_lambda=1,tree_method='hist',device='cpu',random_state=42,n_jobs=4)


def experiment_plan():
    jobs=[]
    for task in ['regression','multiclass']:
        jobs.append(dict(hypothesis='HYP-011',task=task,family='xgboost',transform='none',parameters={**XGB_PARAMS,'objective':'reg:squarederror' if task=='regression' else 'multi:softprob'}))
    for number,family,transform,extra in [(12,'extra_trees','log1p_target',{}),(13,'lightgbm','none',{'objective':'tweedie','tweedie_variance_power':1.5}),
                                          (14,'lightgbm','linear_scaled',{'linear_tree':True}),(15,'extra_trees','target_encode',{})]:
        jobs.append(dict(hypothesis=f'HYP-{number:03d}',task='regression',family=family,transform=transform,parameters={**(ET_PARAMS if family=='extra_trees' else LGB_PARAMS),**extra}))
    for number,transform,extra in [(16,'categorical_cross',{}),(17,'frequency',{}),(18,'none',{'objective':'multiclassova'}),
                                    (19,'hierarchical',{}),(20,'inverse_group_weights',{})]:
        jobs.append(dict(hypothesis=f'HYP-{number:03d}',task='multiclass',family='lightgbm',transform=transform,parameters={**LGB_PARAMS,**extra}))
    return jobs


def semantic_key(candidate):
    identity={k:v for k,v in candidate.items() if k not in ['hypothesis','display_name','description']}
    identity['parameters']={k:v for k,v in candidate['parameters'].items() if k not in ['random_state','random_seed','seed','n_jobs','thread_count','verbosity']}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def inverse_group_weights(X):
    groups=pd.util.hash_pandas_object(X,index=False)
    counts=groups.value_counts()
    return (1/groups.map(counts)).to_numpy(dtype=float)


def transform_target(y,kind):
    values=np.asarray(y,dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Target must be finite')
    if kind=='log1p':
        if np.any(values<=-1): raise ValueError('log1p target outside domain')
        return np.log1p(values)
    return values


def inverse_target(values,kind):
    out=np.expm1(values) if kind=='log1p' else np.asarray(values)
    if not np.isfinite(out).all(): raise ValueError('Inverse target generated nonfinite predictions')
    return out


def hierarchy_probabilities(attack_probability,conditional,attack_classes,global_classes,normal_class):
    p=np.asarray(attack_probability,dtype=float).reshape(-1)
    conditional=np.asarray(conditional,dtype=float)
    global_classes=list(global_classes); attack_classes=list(attack_classes)
    if normal_class not in global_classes or set(attack_classes)!=set(global_classes)-{normal_class}: raise ValueError('Invalid hierarchy class alignment')
    if conditional.shape!=(len(p),len(attack_classes)): raise ValueError('Invalid conditional shape')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isfinite(conditional).all() or np.any(conditional<0): raise ValueError('Invalid hierarchy probabilities')
    if not np.allclose(conditional.sum(axis=1),1,atol=1e-7): raise ValueError('Conditional class probabilities must sum to one')
    result=np.zeros((len(p),len(global_classes)))
    result[:,global_classes.index(normal_class)]=1-p
    for j,label in enumerate(attack_classes): result[:,global_classes.index(label)]=p*conditional[:,j]
    return result


class EncodedXGBClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,parameters=None): self.parameters=parameters
    def fit(self,X,y):
        from xgboost import XGBClassifier
        self.labels_=LabelEncoder().fit(y); self.classes_=self.labels_.classes_
        self.model_=XGBClassifier(**self.parameters).fit(X,self.labels_.transform(y))
        return self
    def predict_proba(self,X): return self.model_.predict_proba(X)
    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


class HierarchyClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,parameters=None,normal_class='Normal'): self.parameters=parameters; self.normal_class=normal_class
    def fit(self,X,y):
        from lightgbm import LGBMClassifier
        y=np.asarray(y); self.classes_=np.unique(y)
        if self.normal_class not in self.classes_ or len(self.classes_)<3: raise ValueError('Hierarchy requires declared normal and at least two attack classes')
        is_attack=y!=self.normal_class
        self.binary_=Pipeline([('encode',Encoder(native=True)),('model',LGBMClassifier(**self.parameters))])
        self.attack_=Pipeline([('encode',Encoder(native=True)),('model',LGBMClassifier(**self.parameters))])
        self.binary_.fit(X,is_attack.astype(int)); self.attack_.fit(X.iloc[np.flatnonzero(is_attack)],y[is_attack])
        self.audit_={'binary_fit_rows':len(y),'attack_fit_rows':int(is_attack.sum()),'normal_class':self.normal_class,
                     'global_classes':self.classes_.tolist(),'attack_classes':self.attack_.classes_.tolist(),'separate_fitted_preprocessing':True}
        return self
    def predict_proba(self,X):
        binary=self.binary_.predict_proba(X)
        attack_column=list(self.binary_.classes_).index(1)
        return hierarchy_probabilities(binary[:,attack_column],self.attack_.predict_proba(X),self.attack_.classes_,self.classes_,self.normal_class)
    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


def build_experiment(candidate):
    from lightgbm import LGBMRegressor,LGBMClassifier
    from sklearn.ensemble import ExtraTreesRegressor
    from experiments.novel_features import GroupTargetEncoder,CrossFeatureTransformer,FrequencyFeatureTransformer
    family,transform,task=candidate['family'],candidate['transform'],candidate['task']
    parameters=dict(candidate['parameters'])
    if transform=='hierarchical': return HierarchyClassifier(parameters=parameters)
    if family=='xgboost':
        if task=='regression':
            from xgboost import XGBRegressor
            model=XGBRegressor(**parameters)
        else: model=EncodedXGBClassifier(parameters=parameters)
    elif family=='extra_trees': model=ExtraTreesRegressor(**parameters)
    else: model=(LGBMRegressor if task=='regression' else LGBMClassifier)(**parameters)
    steps=[]
    if transform=='target_encode': steps.append(('new_features',GroupTargetEncoder(n_splits=5,smoothing=20.)))
    elif transform=='categorical_cross': steps.append(('new_features',CrossFeatureTransformer()))
    elif transform=='frequency': steps.append(('new_features',FrequencyFeatureTransformer()))
    steps.extend([('encode',Encoder(native=family=='lightgbm',scale=transform=='linear_scaled')),('model',model)])
    return Pipeline(steps)


def fit_experiment(candidate,X,y):
    model=build_experiment(candidate); kwargs={}
    target=transform_target(y,'log1p') if candidate['transform']=='log1p_target' else y
    if candidate['parameters'].get('objective')=='tweedie':
        if np.any(np.asarray(y)<0) or not np.any(np.asarray(y)>0): raise ValueError('Tweedie requires nonnegative target with positive observations')
    if candidate['transform']=='inverse_group_weights': kwargs['model__sample_weight']=inverse_group_weights(X)
    model.fit(X,target,**kwargs)
    return model


def predictions(candidate,model,X):
    if candidate['task']=='regression':
        pred=inverse_target(model.predict(X),'log1p') if candidate['transform']=='log1p_target' else np.asarray(model.predict(X))
        return np.asarray(pred).reshape(-1),None,None
    classes=np.asarray(model.classes_); probabilities=np.asarray(model.predict_proba(X))
    if probabilities.shape!=(len(X),len(classes)) or not np.isfinite(probabilities).all(): raise ValueError('Invalid classification scores')
    return classes[np.argmax(probabilities,axis=1)],probabilities,classes
