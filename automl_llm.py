"""Bounded OpenAI-compatible tool client; uses platform environment only."""
import json
import os
import urllib.request


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
        return action if isinstance(action,dict) and action.get('family') else None
    except (ValueError,TypeError,KeyError,IndexError): return None


def propose(profile,observations,timeout=45):
    base=os.getenv('OPENAI_BASE_URL'); key=os.getenv('OPENAI_API_KEY'); model=os.getenv('OPENAI_MODEL')
    if not all([base,key,model]): return None,{'status':'unconfigured','required':['OPENAI_BASE_URL','OPENAI_API_KEY','OPENAI_MODEL']}
    url=base.rstrip('/')
    if not url.endswith('/chat/completions'): url+='/chat/completions'
    schema={'type':'function','function':{'name':'run_experiment','description':'Evaluate one genuinely new hypothesis on a fixed validation split.',
        'parameters':{'type':'object','properties':{'hypothesis':{'type':'string'},'family':{'enum':['lightgbm','extra_trees','catboost','random_forest']},
        'parameters':{'type':'object'},'feature_code':{'type':'string','description':'Optional def engineer(X), pandas/numpy available. Row-wise only, preserve index. No imports, I/O or fitted statistics.'}},'required':['hypothesis','family','parameters']}}}
    messages=[{'role':'system','content':'You are a tabular AutoResearch scientist. Inspect task, profile and previous evidence. Propose ONE new justified experiment using run_experiment. Never repeat known experiments or disguise a repeat by changing seeds. No data access outside X; no target leakage. Prefer useful domain interactions. Families and resource bounds are enforced. Return JSON tool arguments.'},
              {'role':'user','content':json.dumps({'profile':profile,'observations':observations},default=str)[:36000]}]
    body=json.dumps({'model':model,'messages':messages,'tools':[schema],'tool_choice':'auto','max_tokens':3000,'temperature':.2}).encode()
    errors=[]
    for attempt in range(2):
        try:
            request=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=timeout) as response: payload=json.loads(response.read(2_000_000))
            action=parse_action(payload)
            if action is not None: return action,{'status':'ok','attempts':attempt+1,'usage':payload.get('usage',{})}
            errors.append('empty/truncated/invalid response')
        except Exception as exc: errors.append(type(exc).__name__)
    return None,{'status':'failed','errors':errors}
