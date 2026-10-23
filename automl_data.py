"""Task/data contract shared by the entry point and isolated workers."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


LOWER_IS_BETTER = {'rmse','mae','mse','rmsle','mape','log_loss'}
CLASSIFICATION_METRICS = {'macro_f1','weighted_f1','accuracy','balanced_accuracy','roc_auc','log_loss'}
REGRESSION_METRICS = {'rmse','mae','mse','rmsle','mape','r2'}
_METRIC_PATTERNS = {
    'balanced_accuracy': r'\bbalanced[\s_-]*accuracy\b',
    'weighted_f1': r'\b(?:weighted[\s_-]*f1|f1[\s_-]*weighted)\b',
    'macro_f1': r'\b(?:macro[\s_-]*f1|f1[\s_-]*macro)\b',
    'roc_auc': r'\broc[\s_-]*auc\b',
    'log_loss': r'\blog[\s_-]*loss\b',
    'rmse': r'\brmse\b', 'mae': r'\bmae\b', 'mse': r'\bmse\b',
    'rmsle': r'\brmsle\b', 'mape': r'\bmape\b', 'r2': r'\br(?:2|²|\^2)\b',
    'accuracy': r'\b(?:accuracy|точност[ьи])\b',
}
_DECLARATION = re.compile(
    r'\b(?:(?P<primary>primary|main|основн\w*)\s+)?'
    r'(?:(?:evaluation|quality)\s+)?(?:metric|метрик\w*)\b(?:\s+(?:оценк\w*|качеств\w*))?', re.I)
_SECONDARY = re.compile(r'\b(?:secondary|auxiliary|example\w*|diagnostic\w*|'
                        r'дополнител\w*|вторичн\w*|справочн\w*|пример\w*)\b', re.I)


def _metric_mentions(text):
    # Keep the longest overlapping match: balanced accuracy is one metric,
    # not a simultaneous mention of ordinary accuracy.
    matches = [(m.start(),m.end(),key) for key,pattern in _METRIC_PATTERNS.items()
               for m in re.finditer(pattern,text,re.I)]
    accepted=[]
    for start,end,key in sorted(matches,key=lambda m:(m[0],-(m[1]-m[0]))):
        if not any(start<old_end and end>old_start for old_start,old_end,_ in accepted):
            accepted.append((start,end,key))
    return {key for _,_,key in accepted}


def infer_metric(text,default='rmse'):
    """Prefer explicit main metric, then declaration, then sole mention.

    Unknown explicit metrics and conflicting declarations raise before fit;
    a default applies only when no metric has been specified or mentioned.
    """
    lines=[re.sub(r'[*`]', '',line).strip() for line in text.splitlines()]
    declarations=[]
    for i,line in enumerate(lines):
        for match in _DECLARATION.finditer(line):
            primary=bool(match.group('primary'))
            if _SECONDARY.search(line[:match.start()]) and not primary: continue
            fragment=line[match.end():].strip()
            heading=bool(re.match(r'^#{1,6}\s',line))
            if not (primary or heading or re.match(r'^(?:[:—–=\-]|is\b|equals\b|выбран\w*\b|является\b)',fragment,re.I)):
                continue
            fragment=_SECONDARY.split(fragment,maxsplit=1)[0].strip()
            if not fragment.strip(' :—–=-.'): 
                fragment=next((following for following in lines[i+1:] if following), '')
                if fragment.startswith('#'): fragment=''
            value=re.sub(r'^[\s:—–=.,\-]*(?:(?:is|equals|the|will\s+be|выбран\w*|является)\b[\s:—–=.,\-]*)*',
                         '',fragment,flags=re.I)
            # A known diagnostic mentioned after an unknown explicit metric
            # cannot silently replace it (e.g. SMAPE, with RMSE as comparison).
            known_start=any(re.match(pattern,value,re.I) for pattern in _METRIC_PATTERNS.values())
            declarations.append((2 if primary else 1,fragment,_metric_mentions(fragment) if known_start else set()))
    if declarations:
        rank=max(item[0] for item in declarations)
        chosen=[item for item in declarations if item[0]==rank]
        unsupported=[fragment for _,fragment,names in chosen if not names]
        if unsupported:
            raise ValueError('Unsupported explicit metric: '+repr(unsupported[0]))
        names=set().union(*(names for _,_,names in chosen))
    else:
        names=_metric_mentions(text)
    if len(names)>1: raise ValueError('Ambiguous metric: '+', '.join(sorted(names)))
    if names: return next(iter(names))
    if default not in CLASSIFICATION_METRICS|REGRESSION_METRICS:
        raise ValueError('Unsupported default metric: '+str(default))
    return default


@dataclass
class TaskData:
    task_path: Path
    text: str
    paths: dict
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    dictionary: pd.DataFrame
    target: str
    id_column: str | None
    task_type: str
    metric: str
    @property
    def X(self):
        return self.train.drop(columns=[self.target]+([self.id_column] if self.id_column else []))
    @property
    def y(self):
        return self.train[self.target]
    @property
    def X_test(self):
        return self.test[self.X.columns]
    @property
    def hashes(self):
        return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in self.paths.items()}


def resolve_paths(task_path,text):
    base=task_path.parent; result={}
    tokens=re.findall(r'[^\s`<>"\[\]()]+\.csv',text,flags=re.I)
    for name in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv']:
        explicit=[]
        for token in tokens:
            token=token.strip("'.,;:").replace('\\','/')
            if token.lower().endswith('/'+name) or token.lower()==name:
                p=Path(token); p=p if p.is_absolute() else base/p
                if p.is_file(): explicit.append(p.resolve())
        options=list(dict.fromkeys(explicit))
        if not options:
            options=[p.resolve() for p in base.rglob(name) if 'agent_runs' not in p.parts]
        if len(options)!=1:
            raise ValueError(f'Expected one unambiguous {name}; found {len(options)}')
        result[name]=options[0]
    return result


def load_task(path):
    path=Path(path).resolve(); text=path.read_text(encoding='utf-8-sig')
    paths=resolve_paths(path,text)
    train,test,sample,dictionary=[pd.read_csv(paths[k]) for k in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv']]
    extra=[c for c in train if c not in test]
    if len(extra)!=1: raise ValueError('Expected one training target absent from test')
    target=extra[0]
    common=[c for c in sample if c in test and c in train]
    ids=[c for c in common if sample[c].equals(test[c]) and test[c].is_unique]
    if 'role' in dictionary and 'feature' in dictionary:
        declared=dictionary.loc[dictionary.role.astype(str).str.lower().isin(['id','identifier']), 'feature'].tolist()
        ids=[c for c in ids if c in declared] or ids
    if len(ids)>1: raise ValueError('Ambiguous identifier')
    identifier=ids[0] if ids else None
    lower=text.lower(); y=train[target]
    classification=bool(re.search(r'classif|классифик',lower)) or not pd.api.types.is_numeric_dtype(y)
    if re.search(r'regression|регресс',lower) and pd.api.types.is_numeric_dtype(y): classification=False
    metric=infer_metric(text,default='macro_f1' if classification else 'rmse')
    kind='classification' if metric in CLASSIFICATION_METRICS else 'regression'
    if y.isna().any(): raise ValueError('Training target contains missing values')
    expected=[c for c in train if c not in [target,identifier]]
    if set(expected)!=set(test.columns)-({identifier} if identifier else set()):
        raise ValueError('Train/test predictor schemas differ')
    if len(sample)!=len(test): raise ValueError('Sample/test row count mismatch')
    return TaskData(path,text,paths,train,test,sample,dictionary,target,identifier,kind,metric)


def validation_split(bundle):
    X,y=bundle.X,bundle.y; pos=np.arange(len(y))
    if bundle.task_type=='regression':
        for c in X:
            if not re.search(r'year|period|date|год|период',c,re.I) or re.search(r'built|birth|построй',c,re.I): continue
            if not pd.api.types.is_numeric_dtype(X[c]): continue
            unique=sorted(X[c].dropna().unique())
            if 3<=len(unique)<=30 and bundle.X_test[c].min()>unique[-1]:
                fit=pos[X[c].to_numpy()<unique[-1]]; valid=pos[X[c].to_numpy()==unique[-1]]
                return fit,valid,f'temporal:{c}:last={unique[-1]}'
        a,b=train_test_split(pos,test_size=.2,random_state=42)
        return a,b,'random80/20:seed42'
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    if len(np.unique(groups))<len(groups):
        a,b=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,groups))
        desc='stratified-group5:first:seed42'
    else:
        a,b=train_test_split(pos,test_size=.2,stratify=y,random_state=42); desc='stratified80/20:seed42'
    if set(y.iloc[a])!=set(y) or set(y.iloc[b])!=set(y): raise ValueError('Validation split lacks a class')
    if set(groups[a]) & set(groups[b]): raise ValueError('Duplicate feature rows across validation')
    return a,b,desc


def split_key(fit,valid):
    return hashlib.sha256(np.asarray(fit,dtype='<i8').tobytes()+b'|'+np.asarray(valid,dtype='<i8').tobytes()).hexdigest()


def score(metric,y,prediction):
    if metric=='rmse': return float(np.sqrt(metrics.mean_squared_error(y,prediction)))
    if metric=='mse': return float(metrics.mean_squared_error(y,prediction))
    if metric=='mae': return float(metrics.mean_absolute_error(y,prediction))
    if metric=='mape': return float(metrics.mean_absolute_percentage_error(y,prediction))
    if metric=='rmsle':
        if np.any(np.asarray(y)<0): raise ValueError('RMSLE requires nonnegative actual targets')
        return float(np.sqrt(metrics.mean_squared_log_error(y,np.maximum(np.asarray(prediction),0))))
    if metric=='r2': return float(metrics.r2_score(y,prediction))
    if metric=='accuracy': return float(metrics.accuracy_score(y,prediction))
    if metric=='balanced_accuracy': return float(metrics.balanced_accuracy_score(y,prediction))
    if metric=='roc_auc': return float(metrics.roc_auc_score(y,prediction,multi_class='ovr' if np.ndim(prediction)>1 else 'raise'))
    if metric=='log_loss': return float(metrics.log_loss(y,prediction))
    if metric in ['macro_f1','weighted_f1']:
        return float(metrics.f1_score(y,prediction,labels=sorted(pd.unique(np.asarray(y))),average='weighted' if metric=='weighted_f1' else 'macro',zero_division=0))
    raise ValueError('Unsupported metric: '+str(metric))


def better(metric,a,b):
    if b is None: return np.isfinite(a)
    return np.isfinite(a) and (a<b if metric in LOWER_IS_BETTER else a>b)


def make_submission(bundle,prediction):
    frame=bundle.sample.copy(); outputs=[c for c in frame if c!=bundle.id_column]
    p=np.asarray(prediction)
    if len(outputs)==1:
        if p.ndim>1 and p.shape[1]>1: p=p[:,-1]
        frame[outputs[0]]=p.reshape(-1)
    elif p.ndim==2 and p.shape[1]==len(outputs): frame.loc[:,outputs]=p
    else: raise ValueError('Prediction shape does not match sample columns')
    if bundle.id_column: frame[bundle.id_column]=bundle.test[bundle.id_column].to_numpy()
    validate_submission(bundle,frame); return frame


def validate_submission(bundle,frame):
    if list(frame)!=list(bundle.sample) or len(frame)!=len(bundle.test) or not len(frame): raise ValueError('Invalid submission shape')
    if frame.isna().any().any(): raise ValueError('Missing submission value')
    if bundle.id_column:
        c=bundle.id_column
        if not frame[c].is_unique or not frame[c].reset_index(drop=True).equals(bundle.test[c].reset_index(drop=True)):
            raise ValueError('Invalid ID order')
    outputs=[c for c in frame if c!=bundle.id_column]
    if bundle.task_type=='regression' or bundle.metric in ['roc_auc','log_loss'] or len(outputs)>1:
        if not np.isfinite(frame[outputs].to_numpy(dtype=float)).all(): raise ValueError('Nonfinite predictions')
    elif not set(frame[outputs[0]]).issubset(set(bundle.y)): raise ValueError('Unknown class label')


def fallback_submission(bundle):
    if bundle.task_type=='regression': pred=np.full(len(bundle.test),float(bundle.y.mean()))
    elif bundle.metric in ['roc_auc','log_loss']:
        priors=bundle.y.value_counts(normalize=True).reindex(sorted(bundle.y.unique())).to_numpy()
        outputs=[c for c in bundle.sample if c!=bundle.id_column]
        classes=sorted(bundle.y.unique())
        if len(outputs)>1:
            mapping={str(c):i for i,c in enumerate(classes)}
            if set(outputs)!=set(mapping): raise ValueError('Probability output names must match class labels')
            priors=priors[[mapping[c] for c in outputs]]
        pred=np.tile(priors,(len(bundle.test),1))
    else: pred=np.repeat(bundle.y.mode().iloc[0],len(bundle.test))
    return make_submission(bundle,pred)


def atomic_submission(bundle,frame,destination=None):
    validate_submission(bundle,frame)
    target=Path(destination) if destination else bundle.task_path.parent/'submission.csv'
    temporary=target.with_name(target.name+'.tmp')
    frame.to_csv(temporary,index=False)
    validate_submission(bundle,pd.read_csv(temporary))
    os.replace(temporary,target)
