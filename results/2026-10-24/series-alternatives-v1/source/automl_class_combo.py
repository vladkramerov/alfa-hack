"""Portable, train-only classification combinations (SPEC-008, H46--H48)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from automl_advanced import calibration_folds, _choose_multipliers, calibrate_probabilities

COMPONENT_ORDER = ['raw', 'smote', 'frequency']
STRATEGIES = {'smote_macro_f1': ['smote'],
              'frequency_blend_macro_f1': ['raw', 'frequency'],
              'probability_stacking': COMPONENT_ORDER}


def _fit_once(model, X, y, configuration):
    from automl_combo_memory import fit_once
    return fit_once(model, X, y, configuration)


def aligned_probabilities(probabilities, source, target):
    p = np.asarray(probabilities, dtype=float)
    source, target = list(source), list(target)
    if (p.ndim != 2 or p.shape[1] != len(source) or len(set(source)) != len(source)
            or not set(source).issubset(target) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p.sum(axis=1) <= 0)):
        raise ValueError('Invalid class probabilities or alignment')
    result = np.zeros((len(p), len(target)))
    for i, label in enumerate(source): result[:, target.index(label)] = p[:, i]
    return result / result.sum(axis=1, keepdims=True)


def _keys(series): return ('V:' + series.astype('string')).fillna('M:')


class FrequencyFeatureTransformer(TransformerMixin, BaseEstimator):
    """The exact H17 tagged-key individual and joint count/n_fit features."""
    def __init__(self, columns=('proto', 'service', 'state'), joint_name='proto_service_state'):
        self.columns, self.joint_name = columns, joint_name

    def _all_keys(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or any(c not in X for c in self.columns):
            raise ValueError('Frequency inputs require a DataFrame with source columns')
        keys = {c: _keys(X[c]) for c in self.columns}
        parts = [keys[c].map(lambda value: str(len(value)) + ':' + value) for c in self.columns]
        joint = parts[0]
        for part in parts[1:]: joint = joint + '|' + part
        keys[self.joint_name] = joint.astype(str)
        return keys

    def fit(self, X, y=None):
        if not self.columns or len(set(self.columns)) != len(self.columns) or self.joint_name in self.columns:
            raise ValueError('Frequency source names must be nonempty and distinct')
        keys = self._all_keys(X)
        if not len(X) or any(c + '__frequency' in X for c in keys):
            raise ValueError('Empty frequency input or feature name collision')
        self.n_fit_ = len(X)
        self.frequency_maps_ = {c: (key.value_counts(dropna=False) / len(X)).to_dict() for c, key in keys.items()}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.output_columns_ = list(X.columns) + [c + '__frequency' for c in keys]
        self.audit_ = {'n_fit': len(X), 'source_columns': list(self.columns),
                       'features': [c + '__frequency' for c in keys], 'uses_target': False}
        return self

    def transform(self, X):
        if any(c not in X for c in self.feature_names_in_): raise ValueError('Missing fitted column')
        out = X.copy()
        for c, key in self._all_keys(X).items():
            out[c + '__frequency'] = key.map(self.frequency_maps_[c]).fillna(0.).to_numpy(float)
        return out

    def get_feature_names_out(self, input_features=None): return np.asarray(self.output_columns_, dtype=object)


def constrain_synthetic(X, generated, numeric):
    generated = generated.copy()
    valid = np.ones(len(generated), dtype=bool)
    audit = {'rounded_integer_fields': [], 'tcp_zero_support_constraints': []}
    for column in numeric:
        original = pd.to_numeric(X[column], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(original) and np.allclose(original, np.rint(original), atol=1e-8, rtol=0):
            generated[column] = np.rint(generated[column].astype(float))
            audit['rounded_integer_fields'].append(column)
        if len(original) and (original >= 0).all(): valid &= generated[column].to_numpy(dtype=float) >= -1e-8
    joint = [c for c in ('proto', 'service', 'state') if c in X]
    if len(joint) >= 2:
        observed = set(map(tuple, X[joint].astype('string').fillna('__MISSING__').astype(str).to_numpy()))
        keys = generated[joint].astype('string').fillna('__MISSING__').astype(str).to_numpy()
        valid &= np.array([tuple(row) in observed for row in keys], dtype=bool)
    if 'proto' in X:
        original_protocol = X['proto'].astype('string').str.lower()
        generated_protocol = generated['proto'].astype('string').str.lower()
        fields = [c for c in ('stcpb','dtcpb','tcprtt','synack','ackdat','swin','dwin') if c in X]
        for protocol in original_protocol.dropna().unique():
            if protocol in {'tcp','6'}: continue
            for column in fields:
                observed = pd.to_numeric(X.loc[original_protocol.eq(protocol).fillna(False), column], errors='coerce').dropna()
                if len(observed) and np.allclose(observed, 0., atol=1e-12):
                    mask = generated_protocol.eq(protocol).fillna(False).to_numpy(bool)
                    values = pd.to_numeric(generated[column], errors='coerce').to_numpy()
                    valid &= ~mask | (np.isfinite(values) & (np.abs(values) <= 1e-8))
                    audit['tcp_zero_support_constraints'].append([str(protocol),column])
    return generated, valid, audit


def bounded_smote(X, y):
    """H38 SMOTENC, retaining original raw rows and rejecting invalid synthesis."""
    from imblearn.over_sampling import SMOTENC
    categorical = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])
                   or str(c).lower() in {'proto','protocol'} or set(X[c].dropna().unique()).issubset({0,1})]
    numeric = [c for c in X if c not in categorical]
    labels, counts = np.unique(y, return_counts=True)
    strategy = {label: int(count + min(count,500)) for label,count in zip(labels,counts) if 6 <= count < counts.max()}
    audit = {'categorical_columns':categorical,'numeric_columns':numeric,'strategy':{str(k):v for k,v in strategy.items()},
             'k_neighbors':3,'generated_before_constraints':0,'generated_accepted':0,'constraint_rejections':0}
    if not categorical or not numeric or not strategy:
        audit['diagnosis'] = 'no_eligible_mixed_type_sampling'
        audit['accepted'] = []
        return X.copy(),np.asarray(y).copy(),audit,X.iloc[:0].copy()
    frame = X.copy()
    for c in categorical: frame[c] = frame[c].astype('string').fillna('__MISSING__').astype(object)
    values = frame[numeric].replace([np.inf,-np.inf],np.nan)
    scaler = StandardScaler()
    frame[numeric] = scaler.fit_transform(values.fillna(values.median().fillna(0)))
    sampler = SMOTENC(categorical_features=[X.columns.get_loc(c) for c in categorical],
                       sampling_strategy=strategy,k_neighbors=3,random_state=42)
    sampled,target = sampler.fit_resample(frame,np.asarray(y))
    sampled = pd.DataFrame(sampled,columns=X.columns)
    sampled[numeric] = scaler.inverse_transform(sampled[numeric])
    generated = sampled.iloc[len(X):].copy()
    for c in categorical:
        generated[c] = (pd.to_numeric(generated[c].replace('__MISSING__',np.nan))
                        if pd.api.types.is_numeric_dtype(X[c]) else generated[c].replace('__MISSING__',np.nan))
    generated,valid,domain = constrain_synthetic(X,generated,numeric)
    accepted = generated.loc[valid].copy()
    accepted_y = np.asarray(target)[len(X):][valid]
    audit.update(generated_before_constraints=len(generated),generated_accepted=len(accepted),
                 constraint_rejections=int((~valid).sum()),domains=domain,accepted=valid.tolist(),
                 generated_target=np.asarray(target)[len(X):].astype(str).tolist())
    synthetic = accepted.copy(); synthetic['__synthetic_target__'] = accepted_y
    return pd.concat([X,accepted],ignore_index=True),np.concatenate([np.asarray(y),accepted_y]),audit,synthetic


class ClassBranch:
    def __init__(self, component, parameters=None, threads=4):
        self.component,self.parameters,self.threads = component,dict(parameters or {}),threads

    def fit(self, X, y):
        from automl_models import Encoder
        y = np.asarray(y); self.labels_ = LabelEncoder().fit(y); self.classes_ = self.labels_.classes_
        frame = X.copy(); self.synthetic_ = X.iloc[:0].copy(); self.audit_ = {}
        if self.component == 'smote': frame,y,self.audit_,self.synthetic_ = bounded_smote(X,y)
        self.feature_ = FrequencyFeatureTransformer() if self.component == 'frequency' else None
        if self.feature_ is not None: frame = self.feature_.fit_transform(frame)
        scale = self.component=='smote' and 'diagnosis' not in self.audit_
        self.encoder_ = Encoder(native=self.component=='frequency',scale=scale)
        values = self.encoder_.fit_transform(frame,y)
        if len(self.classes_) == 1: self.model_ = None; return self
        if self.component == 'frequency':
            from lightgbm import LGBMClassifier
            settings = dict(n_estimators=500,learning_rate=.05,num_leaves=31,min_child_samples=20,reg_lambda=1)
            settings.update(self.parameters)
            self.model_ = LGBMClassifier(**settings,random_state=42,n_jobs=self.threads,verbosity=-1)
            family = 'lightgbm'
        else:
            from xgboost import XGBClassifier
            settings = dict(n_estimators=600,max_depth=6,learning_rate=.05,reg_lambda=1,tree_method='hist',device='cpu')
            settings.update(self.parameters)
            self.model_ = XGBClassifier(**settings,random_state=42,n_jobs=self.threads)
            family = 'xgboost'
        config = {'task':'multiclass','family':family,'parameters':settings,
                  'policy':'light' if scale else 'legacy',
                  'features':'tagged_frequency_H17' if self.component=='frequency' else None,
                  'encoder':{'native':self.component=='frequency','scale':scale}}
        self.model_ = _fit_once(self.model_,values,self.labels_.transform(y),config)
        self.audit_.update(fit_original_rows=len(X),fit_model_rows=len(frame),configuration=config)
        return self

    def predict_proba(self, X):
        if self.model_ is None: return np.ones((len(X),1))
        frame = self.feature_.transform(X) if self.feature_ is not None else X
        return aligned_probabilities(self.model_.predict_proba(self.encoder_.transform(frame)),self.classes_,self.classes_)


def stack_features(probabilities):
    parts = [np.asarray(p,dtype=float) for p in probabilities]
    if not parts or any(p.ndim!=2 or p.shape!=parts[0].shape or not np.isfinite(p).all() or np.any(p<0) for p in parts):
        raise ValueError('Invalid stacking probabilities')
    return np.concatenate([np.log(np.clip(p,1e-6,1.)) for p in parts],axis=1)


def fit_meta_state(probabilities, y, classes):
    from sklearn.linear_model import LogisticRegression
    y,classes = np.asarray(y),np.asarray(classes)
    values = stack_features(probabilities)
    labels,counts = np.unique(y,return_counts=True)
    weights = {label:float(1/np.sqrt(count)) for label,count in zip(labels,counts)}
    config = {'task':'multiclass','family':'logistic_regression','features':'logclip1e-6_raw_smote_frequency',
              'parameters':{'C':.1,'solver':'lbfgs','max_iter':1000,'class_weight':{str(k):v for k,v in weights.items()}}}
    if len(classes) == 1:
        coef,intercept = np.zeros((1,values.shape[1])),np.zeros(1)
    else:
        model = LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,class_weight=weights)
        model = _fit_once(model,values,y,config)
        if not np.array_equal(model.classes_,classes): raise ValueError('Stack classes mismatch')
        coef,intercept = model.coef_,model.intercept_
    state = {'coef':coef.tolist(),'intercept':intercept.tolist(),'classes':classes.tolist(),
             'component_order':list(COMPONENT_ORDER),'clip':1e-6}
    return state,{'class_weight':{str(k):v for k,v in weights.items()},'configuration':config,
                  'selection_source':'training_grouped_oof','meta_fit_on_oof':True,
                  'meta_training_is_independent_oof_evaluation':False}


def predict_meta_state(probabilities, state, classes):
    if state.get('component_order') != COMPONENT_ORDER or state.get('clip',1e-6)!=1e-6:
        raise ValueError('Unsupported frozen stacking protocol')
    if not np.array_equal(state['classes'],classes): raise ValueError('Frozen meta classes mismatch')
    values = stack_features(probabilities)
    coef,intercept = np.asarray(state['coef'],dtype=float),np.asarray(state['intercept'],dtype=float)
    expected_rows = 1 if len(classes)<=2 else len(classes)
    if coef.shape!=(expected_rows,values.shape[1]) or intercept.shape!=(expected_rows,) or not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError('Invalid frozen meta coefficients')
    if len(classes)==1: return np.ones((len(values),1))
    logits = values@coef.T+intercept
    if len(classes)==2:
        logits = np.column_stack([np.zeros(len(values)),logits[:,0]])
    logits -= logits.max(axis=1,keepdims=True)
    exp = np.exp(logits)
    return exp/exp.sum(axis=1,keepdims=True)


class ClassCombination:
    def __init__(self,candidate,threads=4): self.candidate,self.threads = dict(candidate),threads

    def fit(self,X,y):
        y = np.asarray(y); self.classes_ = np.unique(y)
        self.strategy_ = self.candidate['strategy']; components = STRATEGIES[self.strategy_]
        self.audit_ = {'strategy':self.strategy_,'fit_original_rows':len(X),'fit_retained_rows':len(X),'folds':[]}
        self.keep_mask_ = np.ones(len(X),dtype=bool); self.removed_positions_ = np.array([],dtype=int)
        supplied = self.candidate.get('meta_state' if self.strategy_=='probability_stacking' else 'class_multipliers')
        frozen_equal_stack=self.strategy_=='probability_stacking' and 'meta_state' in self.candidate and supplied is None
        if supplied is not None or frozen_equal_stack:
            if self.strategy_=='probability_stacking' and supplied is not None: predict_meta_state([np.ones((1,len(self.classes_)))/len(self.classes_)]*3,supplied,self.classes_)
            elif frozen_equal_stack: pass
            else: calibrate_probabilities(np.ones((1,len(self.classes_))),self.classes_,supplied)
            self.learned_state_ = {'meta_state' if self.strategy_=='probability_stacking' else 'class_multipliers':supplied}
            self.audit_['state_source'] = 'frozen_candidate'
        else:
            folds,reason = calibration_folds(X,y)
            oof = {c:np.zeros((len(X),len(self.classes_))) for c in components}
            if folds:
                for number,(fit,valid) in enumerate(folds):
                    for c in components:
                        branch = ClassBranch(c,self._parameters(c),self.threads).fit(X.iloc[fit],y[fit])
                        oof[c][valid] = aligned_probabilities(branch.predict_proba(X.iloc[valid]),branch.classes_,self.classes_)
                    self.audit_['folds'].append({'fold':number,'fit_positions':fit.tolist(),'valid_positions':valid.tolist(),'raw_groups_disjoint':True})
                if self.strategy_=='probability_stacking':
                    state,meta_audit = fit_meta_state([oof[c] for c in COMPONENT_ORDER],y,self.classes_)
                    self.learned_state_ = {'meta_state':state}; self.audit_['meta'] = meta_audit
                else:
                    p = sum(oof.values())/len(components)
                    mapping,score = _choose_multipliers(y,p,self.classes_)
                    self.learned_state_ = {'class_multipliers':mapping}; self.audit_['inner_macro_f1'] = score
                self.audit_['state_source'] = 'training_grouped_oof'
            else:
                self.audit_.update(state_source='identity_insufficient_data',skip_reason=reason)
                self.learned_state_ = {'class_multipliers':{str(c):1. for c in self.classes_}}
                if self.strategy_=='probability_stacking':
                    # With insufficient groups use a fixed equal-probability mixture.
                    self.learned_state_ = {'meta_state':None}
        self.branches_ = {c:ClassBranch(c,self._parameters(c),self.threads).fit(X,y) for c in components}
        self.audit_.update(self.learned_state_)
        self.audit_['branches'] = {c:b.audit_ for c,b in self.branches_.items()}
        return self

    def _parameters(self,component):
        parameters = dict(self.candidate.get('parameters',{}))
        parameters.update(self.candidate.get('component_parameters',{}).get(component,{}))
        return parameters

    def predict_proba(self,X):
        parts = {c:aligned_probabilities(b.predict_proba(X),b.classes_,self.classes_) for c,b in self.branches_.items()}
        if self.strategy_=='probability_stacking':
            state = self.learned_state_['meta_state']
            return predict_meta_state([parts[c] for c in COMPONENT_ORDER],state,self.classes_) if state is not None else sum(parts.values())/len(parts)
        return calibrate_probabilities(sum(parts.values())/len(parts),self.classes_,self.learned_state_['class_multipliers'])

    def predict(self,X): return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


def build_class_combo(candidate,threads=4):
    if candidate.get('strategy') not in STRATEGIES: raise ValueError('Unknown class combination strategy')
    return ClassCombination(candidate,threads)
