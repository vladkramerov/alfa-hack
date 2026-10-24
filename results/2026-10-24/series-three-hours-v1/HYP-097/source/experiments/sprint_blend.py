"""Predeclared H79 inner-only selection of new temporal component weights."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_memory import Memory, atomic_json, timestamp

ROOT=Path(__file__).resolve().parents[1]

def convex_weight(a,b,y):
    a,b,y=(np.asarray(v,float) for v in (a,b,y))
    delta=a-b;denominator=delta@delta
    return float(np.clip(((y-b)@delta)/denominator,0,1)) if denominator>0 else 1.

def main():
    ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    out=ROOT/'results/2026-10-24/HYP-079-01-inner-combination-sprint'
    out.mkdir(exist_ok=False)
    source=['results/2026-10-24/HYP-070-03-inner-high-sprint','results/2026-10-24/HYP-076-01-inner-default-sprint']
    configuration={'strategy':'H70_H76_convex_temporal_combination','selection':'analytic_least_squares_inner_only','sources':source}
    hashes={p:hashlib.sha256((ROOT/p/'validation_predictions.npz').read_bytes()).hexdigest() for p in source}
    atomic_json(out/'evidence.json',{'started':timestamp(),'status':'started','configuration':configuration,'source_prediction_hashes':hashes,
                                 'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    arrays=[]
    for p in source:
        with np.load(ROOT/p/'validation_predictions.npz',allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
    a,b=arrays
    assert np.array_equal(a['positions'],b['positions']) and np.array_equal(a['y'],b['y'])
    memory=Memory(out/'model-fit-history.json',ROOT/'HYPOTHESES.md')
    X=pd.DataFrame({'h70':a['prediction'],'h76':b['prediction']})
    claim=memory.claim(configuration,X,a['y'],'inner')
    weight=convex_weight(a['prediction'],b['prediction'],a['y'])
    prediction=weight*a['prediction']+(1-weight)*b['prediction']
    score=float(np.sqrt(np.mean((prediction-a['y'])**2)));control=float(np.sqrt(np.mean((a['prediction']-a['y'])**2)))
    result={'status':'completed','hypothesis':'HYP-079','stage':'inner','metric':'rmse','score':score,'reference_score':control,
            'weight_h70':weight,'weight_h76':1-weight,'eligible_for_outer':0<weight<1 and score<control,
            'old_model_refits':0,'source_prediction_hashes':hashes,'fingerprint':claim,'finished':timestamp()}
    atomic_json(out/'metrics.json',result);memory.finish(claim,result)
    np.savez_compressed(out/'validation_predictions.npz',positions=a['positions'],prediction=prediction,y=a['y'],reference=a['prediction'])
    (out/'report.md').write_text('# H79 inner new temporal combination\n\n'+json.dumps(result,indent=2)+'\n\nOnly inner Year4 used to select weight; outer scores not used. No old learner refits.\n',encoding='utf8')
    for p in ['HYPOTHESES.md','results/README.md']:
        with (ROOT/p).open('a',encoding='utf8') as f:f.write('\nHYP-079 inner completed: RMSE='+str(score)+' vsH70='+str(control)+'; w70='+str(weight)+'; [report]('+('results/' if p=='HYPOTHESES.md' else '')+'2026-10-24/HYP-079-01-inner-combination-sprint/report.md). Повтор не запускать.\n')
    print(json.dumps(result))

if __name__=='__main__':main()
