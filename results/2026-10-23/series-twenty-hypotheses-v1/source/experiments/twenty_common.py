"""SPEC-006 model adapter and read-before-fit, non-repeating fit registrar."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from automl_models import Encoder
from experiments.novel_models import ET_PARAMS, XGB_PARAMS, LGB_PARAMS

ROOT=Path(__file__).resolve().parents[1]

def digest_frame(X):
    meta=json.dumps([(str(c),str(t)) for c,t in zip(X.columns,X.dtypes)],ensure_ascii=False)
    return hashlib.sha256(meta.encode()+pd.util.hash_pandas_object(X,index=True).values.tobytes()).hexdigest()

def feature_identity(feature):
    return None if feature is None else {'class':type(feature).__module__+'.'+type(feature).__name__,
        'parameters':feature.get_params(deep=False) if hasattr(feature,'get_params') else repr(feature)}

def fit_identity(X,y,configuration,weights=None):
    identity={'X':digest_frame(X),'y':hashlib.sha256(pd.util.hash_pandas_object(pd.Series(np.asarray(y)),index=False).values.tobytes()).hexdigest(),
              'configuration':configuration,'weights':None if weights is None else hashlib.sha256(np.asarray(weights,dtype='<f8').tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest(),identity

def register_fit(X,y,configuration,weights=None):
    run=os.environ.get('SPEC006_RUN_DIR')
    if not run: return None
    ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf-8')
    fingerprint,identity=fit_identity(X,y,configuration,weights)
    directory=Path(run); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'model-fit-attempts.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if any(r['fingerprint']==fingerprint for r in rows):
        raise RuntimeError('Repeated model fit prohibited; reuse cached predictions: '+fingerprint)
    record={'fingerprint':fingerprint,'status':'started','at':datetime.now(timezone(timedelta(hours=3))).isoformat(),
            'rows':len(X),'configuration':configuration,'identity':identity,
            'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'hypothesis':os.environ.get('SPEC006_HYPOTHESIS')}
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
    return fingerprint

def finish_fit(fingerprint,status,error=None):
    if fingerprint is None: return
    path=Path(os.environ['SPEC006_RUN_DIR'])/'model-fit-outcomes.jsonl'
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'fingerprint':fingerprint,'status':status,'error':error},ensure_ascii=False)+'\n')

class ExperimentModel:
    def __init__(self,task,policy='light',feature=None,family=None,target_transform='auto',parameters=None):
        self.task=task; self.policy=policy; self.feature=feature
        self.family=family or ('extra_trees' if task=='regression' else 'xgboost')
        self.target_transform=('log1p' if task=='regression' else None) if target_transform=='auto' else target_transform
        self.parameters=dict(parameters or {})
    def fit(self,X,y,sample_weight=None):
        target=np.asarray(y); regression=self.task=='regression'
        p=dict(ET_PARAMS if self.family=='extra_trees' else XGB_PARAMS if self.family=='xgboost' else LGB_PARAMS)
        p.update(self.parameters)
        # Seeds and parallelism cannot turn a repeated configuration into a new fit.
        identity_p={k:v for k,v in p.items() if k not in ['random_state','seed','random_seed','n_jobs','verbosity','device']}
        configuration={'task':self.task,'family':self.family,'policy':self.policy,'feature':feature_identity(self.feature),
                       'target_transform':self.target_transform,'parameters':identity_p}
        if os.environ.get('SPEC006_ENFORCE_BASELINE_GUARD')=='1' and self.feature is None and sample_weight is None:
            old_p={k:v for k,v in (ET_PARAMS if regression else XGB_PARAMS).items() if k not in ['random_state','seed','random_seed','n_jobs','verbosity','device']}
            if len(X)==(35000 if regression else 70529) and self.family==('extra_trees' if regression else 'xgboost') and identity_p==old_p:
                raise RuntimeError('Known outer baseline fit prohibited; use saved predictions')
        fingerprint=register_fit(X,target,configuration,sample_weight)
        try:
            self.feature_=clone(self.feature) if self.feature is not None else None
            frame=self.feature_.fit_transform(X,target) if self.feature_ is not None else X.copy()
            self.keep_mask_=np.ones(len(X),dtype=bool)
            if self.policy=='full':
                from automl_preprocessing import TrainingOutlierFilter
                detector=TrainingOutlierFilter()
                frame,target,self.keep_mask_=detector.fit_filter(frame,target,classification=not regression)
                outlier_audit=detector.audit_
            else: outlier_audit={'removed':0}
            if self.policy in ['full','gaussian']:
                from automl_preprocessing import FullPreprocessor
                self.encoder_=FullPreprocessor(native=self.family=='lightgbm')
            elif self.policy in ['light','legacy']: self.encoder_=Encoder(native=self.family=='lightgbm',scale=self.policy=='light')
            else: raise ValueError('Unknown preprocessing policy '+self.policy)
            values=self.encoder_.fit_transform(frame,target)
            if not regression:
                self.labels_=LabelEncoder().fit(target); self.classes_=self.labels_.classes_; target=self.labels_.transform(target)
            elif self.target_transform=='log1p':
                if np.any(target<=-1): raise ValueError('log1p target domain')
                target=np.log1p(target)
            if self.family=='extra_trees':
                from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
                self.model_=(ExtraTreesRegressor if regression else ExtraTreesClassifier)(**p)
            elif self.family=='xgboost':
                from xgboost import XGBRegressor,XGBClassifier
                self.model_=(XGBRegressor if regression else XGBClassifier)(**p)
            elif self.family=='lightgbm':
                from lightgbm import LGBMRegressor,LGBMClassifier
                self.model_=(LGBMRegressor if regression else LGBMClassifier)(**p)
            else: raise ValueError('Unknown family')
            kwargs={} if sample_weight is None else {'sample_weight':np.asarray(sample_weight)[self.keep_mask_]}
            self.model_.fit(values,target,**kwargs)
            self.audit_={'policy':self.policy,'fit_rows':len(X),'retained_rows':len(frame),'removed_positions':np.flatnonzero(~self.keep_mask_).tolist(),
                         'configuration':configuration,'fingerprint':fingerprint,'outliers':outlier_audit,
                         'feature':getattr(self.feature_,'audit_',feature_identity(self.feature_))}
            finish_fit(fingerprint,'completed')
        except Exception as exc:
            finish_fit(fingerprint,'failed',type(exc).__name__+': '+str(exc)); raise
        return self
    def _values(self,X):
        return self.encoder_.transform(self.feature_.transform(X) if self.feature_ is not None else X)
    def predict(self,X):
        if self.task!='regression': return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
        out=np.asarray(self.model_.predict(self._values(X))).reshape(-1)
        if self.target_transform=='log1p': out=np.expm1(out)
        if not np.isfinite(out).all(): raise ValueError('Nonfinite regression prediction')
        return out
    def predict_proba(self,X): return np.asarray(self.model_.predict_proba(self._values(X)))

def make_model(task,policy='light',feature=None,family=None,target_transform='auto',parameters=None):
    return ExperimentModel(task,policy,feature,family,target_transform,parameters)
