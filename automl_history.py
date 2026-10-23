"""Canonical experiment identities and portable, embedded previous evidence."""
import ast
import hashlib
import json
import os
from pathlib import Path

try:
    from automl_known import KNOWN_TASKS
except ImportError:
    KNOWN_TASKS=[]


def candidate_key(candidate):
    defaults={'extra_trees':dict(n_estimators=200,max_depth=24,min_samples_leaf=2,max_features=.8),
              'random_forest':dict(n_estimators=150,max_depth=20,min_samples_leaf=2,max_features=.8),
              'lightgbm':dict(n_estimators=500,learning_rate=.05,num_leaves=31,min_child_samples=20,reg_lambda=1),
              'catboost':dict(iterations=500,depth=6,learning_rate=.05,l2_leaf_reg=3)}
    parameters=dict(defaults.get(candidate['family'],{}))
    parameters.update({k:v for k,v in candidate.get('parameters',{}).items()
                if k not in ['random_state','random_seed','seed','n_jobs','thread_count','verbose']}
    )
    code=candidate.get('feature_code','').strip()
    normalized=ast.dump(ast.parse(code),include_attributes=False) if code else ''
    identity={'family':candidate['family'],'parameters':parameters,'code':normalized,
              'scale':bool(candidate.get('scale',False)),'preprocessing':candidate.get('preprocessing','default')}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def known_task(hashes,validation_key,metric):
    for task in KNOWN_TASKS:
        if all(hashes.get(k)==v for k,v in task['hashes'].items()) and task['split_key']==validation_key and task['metric']==metric:
            return task
    return None


def already_known(candidate,records,prior=None):
    key=candidate_key(candidate)
    if any(r.get('candidate_key')==key for r in records): return True
    if prior and any(candidate_key(r['candidate'])==key for r in prior['records']): return True
    # A renamed ordinary baseline or another strength is not a new direction.
    if prior and candidate['family'] in ['catboost','random_forest','ridge','logistic']:
        return True
    return False


def ledger_case(hashes,validation_key,metric):
    return hashlib.sha256(json.dumps({'hashes':hashes,'split':validation_key,'metric':metric},sort_keys=True).encode()).hexdigest()


def read_ledger(path,case):
    path=Path(path)
    if not path.exists(): return {'records':[]}
    content=json.loads(path.read_text(encoding='utf-8'))
    return content.get('cases',{}).get(case,{'records':[]})


def write_ledger(path,case,entry):
    path=Path(path)
    content=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'version':1,'cases':{}}
    content['cases'][case]=entry
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(content,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(temporary,path)
