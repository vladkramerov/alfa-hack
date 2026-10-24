"""H91 frozen-inner convex pool of complementary NEW regression mechanisms."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from research_memory import Memory, atomic_json, timestamp

ROOT=Path(__file__).resolve().parents[1]
COMPONENTS={'inner':['HYP-070-03-inner-high-sprint','HYP-083-01-inner-default-sprint','HYP-085-01-inner-default-sprint'],
            'outer':['HYP-070-04-outer-high-sprint','HYP-083-02-outer-default-sprint','HYP-085-02-outer-default-sprint']}

def fit_weights(p,y):
    p,y=np.asarray(p,float),np.asarray(y,float)
    scale=max(float(np.std(y)),1.);x,t=p/scale,y/scale
    objective=lambda w:(float(np.mean((x@w-t)**2)),2*x.T@(x@w-t)/len(t))
    k=p.shape[1]
    result=minimize(objective,np.ones(k)/k,jac=True,method='SLSQP',bounds=[(0,1)]*k,
                    constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1,'jac':lambda w:np.ones(k)}],
                    options={'maxiter':200,'ftol':1e-12})
    choices=list(np.eye(k))
    if result.success and np.isfinite(result.x).all() and abs(result.x.sum()-1)<1e-6:choices.append(result.x)
    w=min(choices,key=lambda w:objective(w)[0]).copy()
    w[w<1e-8]=0;return w/w.sum()

def main(stage='inner'):
    ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    assert 'HYP-091' in ledger
    parent=ROOT/'results/2026-10-24'
    out=parent/f'HYP-091-{1 if stage=="inner" else 2:02d}-{stage}-multiblend-sprint';out.mkdir(exist_ok=False)
    source=[parent/name/'validation_predictions.npz' for name in COMPONENTS[stage]]
    configuration={'strategy':'new_temporal_transport_conditional_uncertainty_pool','components':['H70','H83','H85'],
                   'optimizer':'inner_only_SLSQP_original_MSE_simplex','maxiter':200,'ftol':1e-12,'weight_pruning':1e-8}
    evidence={'status':'started','started':timestamp(),'configuration':configuration,'stage':stage,
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest()}
    atomic_json(out/'evidence.json',evidence);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    saved=[]
    for path in source:
        with np.load(path,allow_pickle=False) as z:saved.append({k:z[k] for k in z.files})
    assert all(np.array_equal(z['positions'],saved[0]['positions']) and np.array_equal(z['y'],saved[0]['y']) for z in saved)
    p=np.column_stack([z['prediction'] for z in saved]);y=saved[0]['y'];history=Memory(out/'model-fit-history.json',ROOT/'HYPOTHESES.md')
    if stage=='inner':
        claim=history.claim(configuration,pd.DataFrame(p,columns=configuration['components']),y,'inner')
        weights=fit_weights(p,y)
    else:
        claim=None
        selected=json.loads((parent/'HYP-091-01-inner-multiblend-sprint/metrics.json').read_text(encoding='utf8'))
        if not selected['eligible_for_outer']:raise ValueError('Rejected inner combination cannot be evaluated on outer')
        weights=np.array(selected['weights'])
    prediction=p@weights;score=float(np.sqrt(np.mean((prediction-y)**2)));control=float(np.sqrt(np.mean((p[:,0]-y)**2)))
    result={'status':'completed','hypothesis':'HYP-091','stage':stage,'metric':'rmse','score':score,'reference_score':control,
            'weights':weights.tolist(),'eligible_for_outer':bool(np.count_nonzero(weights)>1 and score<control),
            'old_model_refits':0,'learner_fits':0,'parameter_fits':int(stage=='inner'),'fingerprint':claim,'finished':timestamp()}
    atomic_json(out/'metrics.json',result)
    if claim:history.finish(claim,result)
    np.savez_compressed(out/'validation_predictions.npz',positions=saved[0]['positions'],prediction=prediction,y=y,reference=p[:,0])
    (out/'report.md').write_text('# H91 '+stage+'\n\n'+json.dumps(result,indent=2)+'\n\nInner weights frozen before any H85 outer fit; reused public outer is exploratory. No old model refits.\n',encoding='utf8')
    for filename in ['HYPOTHESES.md','results/README.md']:
        with (ROOT/filename).open('a',encoding='utf8') as f:f.write('\nHYP-091/'+stage+' completed RMSE='+str(score)+' vsH70='+str(control)+' weights='+str(weights.tolist())+' [report]('+('results/' if filename=='HYPOTHESES.md' else '')+str(out.relative_to(ROOT/'results')).replace('\\','/')+'/report.md). Повтор не запускать.\n')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['inner','outer'],default='inner');main(parser.parse_args().stage)
