"""Bounded OpenAI-compatible tool client; uses platform environment only."""
import json
import os
import urllib.request


def research_context(profile,observations):
    """Keep measured feedback in a bounded, valid JSON prompt (never cut JSON)."""
    compact={k:v for k,v in profile.items() if k not in ['numeric_summary','known','hashes','split_key']}
    compact['task']=str(compact.get('task',''))[:9000]
    numeric=profile.get('numeric_summary',{})
    compact['numeric_summary']={k:numeric[k] for k in list(numeric)[:12]}
    known=profile.get('known')
    if known:
        compact['known']={'records':[{'candidate':r.get('candidate'),'score':r.get('score')} for r in known.get('records',[])][-10:],
                          'previous_directions':known.get('previous_directions',[])}
    feedback=[{k:r[k] for k in ['candidate','status','score','metric','error','reason'] if k in r} for r in observations[-12:]]
    for entry in feedback:
        if isinstance(entry.get('candidate'),dict):
            entry['candidate']=dict(entry['candidate'])
            if 'feature_code' in entry['candidate']: entry['candidate']['feature_code']=entry['candidate']['feature_code'][:3000]
    for entry,record in zip(feedback[-3:],observations[-3:]):
        diagnostic=record.get('diagnostics')
        if isinstance(diagnostic,dict):
            # Only aggregates produced by the worker; no row-level artifacts.
            diagnostic={k:v for k,v in diagnostic.items() if k in {'source','type','rows','class_count',
                'per_class','top_confusions','classes_truncated','overall','target_buckets','bucket_definition','period_slices','status','error'}}
            diagnostic=dict(diagnostic)
            for name,cap in [('per_class',24),('top_confusions',12),('target_buckets',5),('period_slices',8)]:
                if isinstance(diagnostic.get(name),list):diagnostic[name]=diagnostic[name][:cap]
            entry['diagnostics']=diagnostic
    payload={'profile':compact,'observations':feedback}
    encoded=json.dumps(payload,default=str)
    if len(encoded)>30000:
        compact.pop('numeric_summary',None); compact['task']=compact['task'][:5000]
        for k in ['dtypes','missing','categories']:
            if isinstance(compact.get(k),dict): compact[k]=dict(list(compact[k].items())[:40])
        encoded=json.dumps(payload,default=str)
    while len(encoded)>30000 and len(feedback)>1:
        feedback.pop(0);encoded=json.dumps(payload,default=str)
    if len(encoded)>30000 and feedback:
        feedback[-1].pop('diagnostics',None)
        encoded=json.dumps(payload,default=str)
    return encoded


def parse_action(payload):
    choices=payload.get('choices') if isinstance(payload,dict) else None
    if not choices or not isinstance(choices,list): return None
    first=choices[0]
    if first.get('finish_reason') in ['length','content_filter']: return None
    message=first.get('message') or {}
    try:
        calls=message.get('tool_calls') or []
        if calls:
            function=calls[0].get('function',{})
            if function.get('name')!='run_experiment': return None
            action=json.loads(function.get('arguments',''))
        else:
            content=message.get('content')
            if not isinstance(content,str): return None
            action=json.loads(content)
        return action if isinstance(action,dict) and (action.get('family') or action.get('stop') is True) else None
    except (ValueError,TypeError,KeyError,IndexError): return None


def propose(profile,observations,timeout=45):
    base=os.getenv('OPENAI_BASE_URL'); key=os.getenv('OPENAI_API_KEY'); model=os.getenv('OPENAI_MODEL')
    if not all([base,key,model]): return None,{'status':'unconfigured','required':['OPENAI_BASE_URL','OPENAI_API_KEY','OPENAI_MODEL']}
    url=base.rstrip('/')
    if not url.endswith('/chat/completions'): url+='/chat/completions'
    schema={'type':'function','function':{'name':'run_experiment','description':'Evaluate one genuinely new hypothesis on a fixed validation split.',
        'parameters':{'type':'object','properties':{'hypothesis':{'type':'string'},'family':{'enum':['xgboost','lightgbm','extra_trees','catboost','random_forest']},
        'preprocessing':{'enum':['light-v1','native-v1','full-v1'],'description':'Ordinary models only: light-v1 uses training medians and bounded categories without row deletion or quantile transforms; native-v1 retains native LightGBM/CatBoost categories. Omitted policy preserves historical full-v1.'},
        'parameters':{'type':'object'},'target_transform':{'enum':['none','log1p']},
        'strategy':{'enum':['group_experts','macro_f1_calibrated','reg_feature_experts','probability_stacking','temporal_profile_memory','panel_hybrid_memory','panel_change_hybrid'],'description':'Versioned strategies: regression experts, combined weather/group feature experts, or strictly past profile target memory with expert fallback (requires Year_Factor and all physical profile fields). Panel hybrid uses stable profile history plus learned causal annual residuals with fixed 50/50 weights; requires repeated physical profiles and Year_Factor. Panel change hybrid averages that model with an annual first-difference forecast using strictly past target and covariate history, also at fixed 50/50 weights. Macro-F1 calibration or grouped OOF probability stacking needs suitable class/schema support. Learn all state only on training folds; never invent coefficients or use future/validation targets.'},
        'group_column':{'type':'string','description':'Optional training categorical column for regression experts; target/id prohibited.'},
        'feature_code':{'type':'string','description':'Optional def engineer(X), pandas/numpy available. Row-wise only, preserve index. No imports, I/O or fitted statistics.'}},'required':['hypothesis','family','parameters']}}}
    messages=[{'role':'system','content':'You are a tabular AutoResearch scientist. Inspect the declared primary metric, fit-only profile and measured previous evidence. Propose ONE genuinely new justified experiment using run_experiment. Choose ordinary preprocessing explicitly by data and model: light-v1 retains all rows with training median numeric values and bounded one-hot categories; native-v1 supports native LightGBM/CatBoost categories; full-v1 additionally uses capped training-only outlier removal, quantile-normal continuous features and scaling. Omitted preprocessing retains the historical full-v1 default. Normalization and outlier deletion are optional alternatives, never assumed beneficial. Explicit group_experts and macro_f1_calibrated strategies preserve the measured legacy median/one-hot representation without mandatory Gaussian transforms or row deletion. Choose strategies by applicability and primary metric; normal-looking features are not evidence of better predictions. Do not invent calibration multipliers or use validation targets; the worker learns calibration on grouped training OOF. No repeat even after seed/name change. No target leakage. Use the latest aggregate per-class errors, confusions, regression target buckets and period slices to justify the next mechanism. These validation aggregates inform proposals, never row features, fitting, calibration or hand-coded rules. Use useful domain interactions; prefer new mechanisms over blind tuning. log1p target applies only to regression with valid domain. Families and bounds are enforced. If no useful new direction remains, return JSON {"stop":true}. The platform metric cannot be inferred from local score scaling.'},
              {'role':'user','content':research_context(profile,observations)}]
    body=json.dumps({'model':model,'messages':messages,'tools':[schema],'tool_choice':'auto','max_tokens':2000,'temperature':.2}).encode()
    errors=[]
    for attempt in range(2):
        try:
            request=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=timeout) as response: payload=json.loads(response.read(2_000_000))
            action=parse_action(payload)
            if action is not None: return action,{'status':'ok','attempts':attempt+1,'usage':payload.get('usage',{}),'estimated_tokens':(len(body)//4+2000)*(attempt+1)}
            errors.append('empty/truncated/invalid response')
        except Exception as exc: errors.append(type(exc).__name__)
    return None,{'status':'failed','errors':errors,'attempts':2,'estimated_tokens':2*(len(body)//4+2000)}
