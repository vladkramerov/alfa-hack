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
    payload={'profile':compact,'observations':feedback}
    encoded=json.dumps(payload,default=str)
    if len(encoded)>30000:
        compact.pop('numeric_summary',None); compact['task']=compact['task'][:5000]
        for k in ['dtypes','missing','categories']:
            if isinstance(compact.get(k),dict): compact[k]=dict(list(compact[k].items())[:40])
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
        'parameters':{'type':'object'},'target_transform':{'enum':['none','log1p']},
        'strategy':{'enum':['group_experts','macro_f1_calibrated','reg_feature_experts','probability_stacking'],'description':'Versioned strategies: regression group_experts or reg_feature_experts (ExtraTrees, combined wind/monthly/group features); Macro-F1 macro_f1_calibrated or probability_stacking (XGBoost, raw/SMOTE/frequency branches, needs proto/service/state). All decision or meta state is learned on grouped training OOF; never invent it.'},
        'group_column':{'type':'string','description':'Optional training categorical column for regression experts; target/id prohibited.'},
        'feature_code':{'type':'string','description':'Optional def engineer(X), pandas/numpy available. Row-wise only, preserve index. No imports, I/O or fitted statistics.'}},'required':['hypothesis','family','parameters']}}}
    messages=[{'role':'system','content':'You are a tabular AutoResearch scientist. Inspect the declared primary metric, fit-only profile and measured previous evidence. Propose ONE genuinely new justified experiment using run_experiment. Ordinary candidates use default full-v1 cleaning, capped training-only outlier removal, quantile-normal continuous features and scaling. Explicit group_experts and macro_f1_calibrated strategies preserve the measured legacy median/one-hot representation without mandatory Gaussian transforms or row deletion. Choose strategies by applicability and primary metric; normal-looking features are not evidence of better predictions. Do not invent calibration multipliers or use validation targets; the worker learns calibration on grouped training OOF. No repeat even after seed/name change. No target leakage. Use previous error diagnostics and useful domain interactions; prefer new mechanisms over blind tuning. log1p target applies only to regression with valid domain. Families and bounds are enforced. If no useful new direction remains, return JSON {"stop":true}. The platform metric cannot be inferred from local score scaling.'},
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
