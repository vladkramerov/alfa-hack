"""Optional offline read-before-fit journal; portable models need no local files."""
import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from scipy import sparse

IGNORED={'hypothesis','name','display_name','random_state','random_seed','seed','n_jobs','thread_count','verbosity','verbose'}

def _semantic(value):
    if isinstance(value,dict): return {str(k):_semantic(v) for k,v in value.items() if k not in IGNORED}
    if isinstance(value,(list,tuple)): return [_semantic(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value

def _data_hash(X):
    h=hashlib.sha256()
    if sparse.issparse(X):
        a=X.tocsr();h.update(str((a.shape,str(a.dtype))).encode())
        for v in [a.indptr,a.indices,a.data]:h.update(np.ascontiguousarray(v).tobytes())
    elif isinstance(X,pd.DataFrame):
        h.update(json.dumps([(str(c),str(t)) for c,t in zip(X.columns,X.dtypes)],ensure_ascii=False).encode())
        h.update(pd.util.hash_pandas_object(X,index=False).values.tobytes())
    else:
        a=np.asarray(X);h.update(str((a.shape,str(a.dtype))).encode())
        if a.dtype.hasobject or a.dtype.kind in 'US':h.update(pd.util.hash_pandas_object(pd.DataFrame(a),index=False).values.tobytes())
        else:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def fit_fingerprint(X,y,configuration):
    identity={'X':_data_hash(X),'y':_data_hash(np.asarray(y).reshape(-1,1)),'configuration':_semantic(configuration)}
    return hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()

def fit_once(model,X,y,configuration):
    directory=os.environ.get('COMBO_RUN_DIR')
    if not directory:return model.fit(X,y)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    ledger=Path(os.environ['COMBO_LEDGER_PATH']).read_text(encoding='utf-8')
    fingerprint=fit_fingerprint(X,y,configuration)
    attempts=directory/'model-fit-attempts.jsonl'
    previous=[json.loads(line) for line in attempts.read_text(encoding='utf-8').splitlines()] if attempts.exists() else []
    if any(r['fingerprint']==fingerprint for r in previous):raise RuntimeError('Fit already attempted; cached artifacts must be reused: '+fingerprint)
    event={'fingerprint':fingerprint,'status':'started','at':datetime.now(timezone(timedelta(hours=3))).isoformat(),
           'configuration':_semantic(configuration),'X_sha256':_data_hash(X),'y_sha256':_data_hash(np.asarray(y).reshape(-1,1)),
           'rows':len(y),'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'hypothesis':os.environ.get('COMBO_HYPOTHESIS')}
    with attempts.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False,default=str)+'\n')
    try:
        fitted=model.fit(X,y);outcome={'fingerprint':fingerprint,'status':'completed'}
    except Exception as exc:
        outcome={'fingerprint':fingerprint,'status':'failed','error':type(exc).__name__+': '+str(exc)}
        with (directory/'model-fit-outcomes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(outcome,ensure_ascii=False)+'\n')
        raise
    with (directory/'model-fit-outcomes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(outcome,ensure_ascii=False)+'\n')
    return fitted
