"""Canonical identities and task-local evidence; no embedded pretrained dataset state."""
import ast
import hashlib
import json
import os
import re
from pathlib import Path

try:
    from automl_known import KNOWN_TASKS
except ImportError:
    KNOWN_TASKS=[]


def candidate_key(candidate):
    defaults={'extra_trees':dict(n_estimators=200,max_depth=24,min_samples_leaf=2,max_features=.8),
              'random_forest':dict(n_estimators=150,max_depth=20,min_samples_leaf=2,max_features=.8),
              'lightgbm':dict(n_estimators=500,learning_rate=.05,num_leaves=31,min_child_samples=20,reg_lambda=1),
              'xgboost':dict(n_estimators=600,max_depth=6,learning_rate=.05,reg_lambda=1,tree_method='hist',device='cpu'),
              'catboost':dict(iterations=500,depth=6,learning_rate=.05,l2_leaf_reg=3)}
    parameters=dict(defaults.get(candidate['family'],{}))
    parameters.update({k:v for k,v in candidate.get('parameters',{}).items()
                if k not in ['random_state','random_seed','seed','n_jobs','thread_count','verbose']}
    )
    code=candidate.get('feature_code','').strip()
    normalized=ast.dump(ast.parse(code),include_attributes=False) if code else ''
    identity={'family':candidate['family'],'parameters':parameters,'code':normalized,
              'scale':bool(candidate.get('scale',False)),'preprocessing':candidate.get('preprocessing','default'),
              'target_transform':candidate.get('target_transform') or 'none','max_outlier_fraction':candidate.get('max_outlier_fraction',0)}
    # Leave historical ordinary identities unchanged; advanced behavior is explicit.
    if candidate.get('strategy'):
        identity.update(strategy=candidate['strategy'],group_column=candidate.get('group_column'),
                        minimum_expert_rows=candidate.get('minimum_expert_rows'),shrinkage=candidate.get('shrinkage'),
                        calibration_folds=candidate.get('calibration_folds'),class_multipliers=candidate.get('class_multipliers'))
        if candidate['strategy'] in {'reg_feature_experts','reg_convex_blend','target_tail_mixture','smote_macro_f1','frequency_blend_macro_f1','probability_stacking','temporal_profile_memory'}:
            identity.update({k:candidate.get(k) for k in ['feature_numbers','blend_weights','tail_quantile','meta_state','component_parameters','gate_parameters']})
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
    if prior:
        # A changed coefficient, seed or title does not authorize repeating a
        # measured strategy on the same data/split/metric.
        strategy=candidate.get('strategy')
        if strategy and any(r.get('candidate',{}).get('strategy')==strategy for r in prior.get('spec008_attempts',[])+prior.get('spec009_attempts',[])):return True
        if strategy in ['group_experts','macro_f1_calibrated'] and any(r['candidate'].get('strategy')==strategy for r in prior['records']): return True
        mentioned={int(v) for v in re.findall(r'\bHYP[-_ ]?0*(\d+)\b',str(candidate.get('hypothesis','')),flags=re.IGNORECASE)}
        if candidate.get('number') is not None:
            try: mentioned.add(int(candidate['number']))
            except (ValueError,TypeError): pass
        tested={int(r['number']) for r in prior.get('spec006_attempts',[])+prior.get('spec008_attempts',[])+prior.get('spec009_attempts',[])}
        if mentioned & tested: return True
    # A renamed ordinary baseline or another strength is not a new direction.
    if prior and candidate['family'] in ['catboost','random_forest','ridge','logistic']:
        return True
    return False


def compatible_policy(candidate):
    if candidate.get('strategy') in {'panel_hybrid_memory','sergey_ivan_fusion'}:
        try:
            from automl_models import canonical_candidate
            expected=canonical_candidate(candidate)
            return all(candidate.get(k)==expected[k] for k in ['preprocessing','scale','max_outlier_fraction','target_transform'])
        except (TypeError,ValueError,KeyError): return False
    if candidate.get('preprocessing')=='full-v1' and bool(candidate.get('scale')) and candidate.get('max_outlier_fraction')==.05 and candidate.get('strategy') in [None,'','full']: return True
    if candidate.get('strategy') in {'reg_feature_experts','reg_convex_blend','target_tail_mixture','smote_macro_f1','frequency_blend_macro_f1','probability_stacking','temporal_profile_memory'}:
        try:
            from automl_models import canonical_candidate
            expected=canonical_candidate(candidate)
            return all(candidate.get(k)==expected.get(k) for k in expected if k not in ['hypothesis','parameters'])
        except (TypeError,ValueError,KeyError):return False
    if candidate.get('strategy') not in ['group_experts','macro_f1_calibrated']: return False
    try:
        from automl_models import canonical_candidate
        expected=canonical_candidate(candidate)
    except (TypeError,ValueError): return False
    keys=['preprocessing','scale','max_outlier_fraction','target_transform']
    keys+=['group_column','minimum_expert_rows','shrinkage'] if candidate['strategy']=='group_experts' else ['calibration_folds']
    return all(candidate.get(k)==expected[k] for k in keys)


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
