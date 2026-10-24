from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
"""Read-only independent metric/data/source audit, no model training."""
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R))
import numpy as np,joblib
from sklearn.metrics import f1_score,accuracy_score
from automl_data import load_task

b=load_task(R/'data/multiclass/task.md');records=[]
for number,stage in enumerate(['primary','secondary'],1):
    out=R/f'results/2026-10-24/HYP-050-{number:02d}-focal-{stage}'
    result=json.loads((out/'metrics.json').read_text())
    if result['status']!='completed':
        records.append({'stage':stage,'status':result['status'],'numerical_score_verified':False});continue
    evidence=json.loads((out/'evidence.json').read_text());audit=json.loads((out/'audit.json').read_text())
    with np.load(out/'validation_predictions.npz') as f:z={k:f[k] for k in f.files}
    with np.load(out/'splits.npz') as f:s={k:f[k] for k in f.files}
    p,classes=z['probabilities'],z['classes'];checks={}
    checks['data_sha']=evidence['data_hashes']==b.hashes
    checks['source_sha']=all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
                               for name,digest in evidence['source_sha256'].items())
    checks['positions']=np.array_equal(z['positions'],s['valid'])
    checks['targets']=np.array_equal(z['y'],np.asarray(b.y.iloc[s['valid']],dtype=str))
    checks['disjoint_rows']=not np.intersect1d(s['fit'],s['valid']).size
    checks['class_codec']=np.array_equal(classes,np.unique(np.asarray(b.y.iloc[s['fit']],dtype=str)))
    checks['finite_shape']=p.shape==(len(s['valid']),len(classes)) and bool(np.isfinite(p).all() and (p>=0).all())
    checks['probability_normalization']=bool(np.allclose(p.sum(1),1,atol=1e-7))
    checks['prediction_argmax']=np.array_equal(z['prediction'],classes[p.argmax(1)])
    score=float(f1_score(z['y'],z['prediction'],labels=classes,average='macro',zero_division=0))
    checks['macro_f1_exact']=score==result['score']
    checks['accuracy_exact']=float(accuracy_score(z['y'],z['prediction']))==result['accuracy']
    checks['fixed600']=audit['objective_calls']==audit['estimators']==600
    checks['actual_multi_softprob']=audit['prediction_objective']=='multi:softprob'
    checks['weight_normalization']=abs(audit['mean_sample_alpha']-1)<1e-12
    checks['one_fit']=audit['model_fit_count']==result['model_fit_count']==1
    checks['budget']=result['total_seconds']<=900
    model=joblib.load(out/'model.joblib')
    checks['serialization_reproduces_predictions']=bool(np.allclose(model.predict_proba(b.X.iloc[s['valid'][:16]]),p[:16],atol=1e-7))
    record={'stage':stage,'recomputed_macro_f1':score,'checks':checks,'all_passed':all(checks.values())}
    assert record['all_passed'],record
    records.append(record)
print(json.dumps({'hypothesis':50,'public_fits_in_verification':0,'records':records,
                  'all_passed':all(r.get('all_passed',False) for r in records)},ensure_ascii=False,indent=2))
