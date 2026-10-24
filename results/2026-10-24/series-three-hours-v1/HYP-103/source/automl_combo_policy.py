"""Versioned candidate contracts for measured combination architectures."""
import copy
import numpy as np
REG={'reg_feature_experts','reg_convex_blend','target_tail_mixture','temporal_profile_memory'}
CLASS={'smote_macro_f1','frequency_blend_macro_f1','probability_stacking'}
STRATEGIES=REG|CLASS

def canonical_combo(candidate):
    c=copy.deepcopy(candidate);strategy=c['strategy']
    if strategy not in STRATEGIES:raise ValueError('Unknown combination')
    if c.get('feature_code','').strip():raise ValueError('Versioned combinations have fixed feature processing')
    reg=strategy in REG;family='extra_trees' if strategy in ['reg_feature_experts','reg_convex_blend','temporal_profile_memory'] else 'xgboost'
    if c.get('family')!=family:raise ValueError('Combination family mismatch')
    p=dict(c.get('parameters',{}))
    for k in ['random_state','random_seed','seed','n_jobs','thread_count','verbose']:p.pop(k,None)
    allowed={'n_estimators','max_depth','min_samples_leaf','max_features','criterion'} if reg else {'n_estimators','max_depth','learning_rate','reg_lambda','reg_alpha','subsample','colsample_bytree','tree_method','device'}
    if set(p)-allowed:raise ValueError('Unsupported combination parameters')
    for k,upper in [('n_estimators',1200),('max_depth',32)]:
        if k in p and (isinstance(p[k],bool) or not isinstance(p[k],int) or not 1<=p[k]<=upper):raise ValueError('Combination resource bounds exceeded')
    if p.get('device','cpu')!='cpu':raise ValueError('Only CPU training permitted')
    c['parameters']=p
    if c.get('component_parameters') or c.get('gate_parameters'):raise ValueError('Versioned combinations use fixed component/gate settings')
    active={'reg_feature_experts':{'feature_numbers','group_column','minimum_expert_rows','shrinkage'},
            'reg_convex_blend':{'blend_weights'},'target_tail_mixture':{'tail_quantile'},
            'smote_macro_f1':{'class_multipliers'},'frequency_blend_macro_f1':{'class_multipliers'},
            'probability_stacking':{'meta_state'},'temporal_profile_memory':set()}[strategy]
    behavior_fields={'feature_numbers','group_column','minimum_expert_rows','shrinkage','blend_weights','tail_quantile','class_multipliers','meta_state'}
    if any(k in c for k in behavior_fields-active):raise ValueError('Inactive combination fields are forbidden')
    if c.get('target_transform') not in [None,'log1p' if reg else 'none']:raise ValueError('Versioned target transform mismatch')
    c.update(preprocessing=strategy.replace('_','-')+'-v1',scale=False,max_outlier_fraction=0,target_transform='log1p' if reg else 'none')
    if strategy=='reg_feature_experts':
        c.setdefault('feature_numbers',[29,30,31]);c.setdefault('group_column','building_class');c.setdefault('minimum_expert_rows',500);c.setdefault('shrinkage',500.)
        if c['feature_numbers'] not in [[30],[29,30,31]]:raise ValueError('Unsupported feature composition')
        if not isinstance(c['group_column'],str) or not c['group_column']:raise ValueError('Named grouping column required')
        if isinstance(c['minimum_expert_rows'],bool) or not isinstance(c['minimum_expert_rows'],int) or c['minimum_expert_rows']<500:raise ValueError('Expert minimum must be at least500')
        if isinstance(c['shrinkage'],bool) or not isinstance(c['shrinkage'],(int,float)) or not np.isfinite(c['shrinkage']) or c['shrinkage']<=0:raise ValueError('Positive finite shrinkage required')
    if strategy=='reg_convex_blend' and c.get('blend_weights') is not None:
        from automl_reg_combo import validate_weights
        c['blend_weights']=validate_weights(c['blend_weights']).tolist()
    if strategy=='target_tail_mixture':
        c.setdefault('tail_quantile',.8)
        if not isinstance(c['tail_quantile'],(float,int)) or not 0<c['tail_quantile']<1:raise ValueError('Invalid tail quantile')
    if strategy in CLASS and c.get('class_multipliers') is not None:
        m=c['class_multipliers']
        if not isinstance(m,dict) or not m or any(not isinstance(k,str) or isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or v<=0 for k,v in m.items()):raise ValueError('Invalid decision multipliers')
        c['class_multipliers']={k:float(v) for k,v in sorted(m.items())}
    if strategy=='probability_stacking' and c.get('meta_state') is not None:
        from automl_class_combo import predict_meta_state
        state=c['meta_state'];classes=np.asarray(state['classes'])
        if not len(classes) or len(np.unique(classes))!=len(classes):raise ValueError('Invalid meta classes')
        predict_meta_state([np.ones((1,len(classes)))/len(classes)]*3,state,classes)
    return c
