"""H102 frozen-inner amplitude of cached H91 minus cached H69; zero learner fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from research_memory import Memory, atomic_json, data_digest, timestamp


ROOT=Path(__file__).resolve().parents[1]
RUNS={'inner':('HYP-091-01-inner-multiblend-sprint','HYP-070-03-inner-high-sprint'),
      'outer':('HYP-091-02-outer-multiblend-sprint','HYP-070-04-outer-high-sprint')}


def _vectors(*values):
    result=[np.asarray(value,float) for value in values]
    if (not result or result[0].ndim!=1 or not len(result[0])
            or any(value.shape!=result[0].shape or not np.isfinite(value).all() for value in result)):
        raise ValueError('Finite nonempty aligned prediction/target vectors required')
    return result


def fit_amplitude(pool,base,y):
    pool,base,y=_vectors(pool,base,y)
    delta=pool-base
    denominator=float(np.dot(delta,delta))
    numerator=float(np.dot(delta,y-base))
    if not np.isfinite([denominator,numerator]).all():
        raise ValueError('Amplitude sufficient statistics outside finite numerical range')
    return 1. if denominator==0 else float(np.clip(numerator/denominator,0.,2.))


def apply_amplitude(pool,base,amplitude):
    pool,base=_vectors(pool,base)
    if not np.isfinite(amplitude) or not 0<=amplitude<=2:
        raise ValueError('Frozen amplitude must be finite and in [0,2]')
    prediction=np.maximum(0.,base+float(amplitude)*(pool-base))
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite amplitude predictions')
    return prediction


def select_amplitude(stage,pool,base,y,configuration=None,history=None,frozen=None):
    if stage=='inner':
        if history is None or configuration is None:
            raise ValueError('Inner scalar optimization requires a pre-fit Memory claim')
        pool,base,y=_vectors(pool,base,y)
        claim=history.claim(configuration,pd.DataFrame({'base_H69':base,'pool_H91':pool}),y,'inner')
        try:
            return fit_amplitude(pool,base,y),claim
        except Exception as exc:
            history.finish(claim,{'status':'failed','error':type(exc).__name__+': '+str(exc)})
            raise
    if stage!='outer' or not isinstance(frozen,dict) or frozen.get('status')!='completed' or frozen.get('eligible_for_outer') is not True:
        raise ValueError('Outer requires completed, incrementally successful frozen inner result')
    amplitude=float(frozen['amplitude'])
    if not np.isfinite(amplitude) or not 0<=amplitude<=2:
        raise ValueError('Invalid frozen inner amplitude')
    return amplitude,None


def validate_cached(pool,transport,split):
    positions=np.asarray(pool['positions'])
    if (positions.ndim!=1 or positions.dtype.kind not in 'iu' or not len(positions)
            or len(np.unique(positions))!=len(positions)
            or not np.array_equal(positions,transport['positions'])
            or not np.array_equal(positions,split['valid'])
            or len(np.intersect1d(split['fit'],split['valid']))
            or not np.array_equal(pool['y'],transport['y'])):
        raise ValueError('Cache positions, targets and original split must align exactly')
    vectors=_vectors(pool['prediction'],transport['reference'],pool['y'])
    if len(vectors[0])!=len(positions):
        raise ValueError('Cache prediction length mismatch')


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _arrays(path):
    with np.load(path,allow_pickle=False) as saved:return {key:saved[key].copy() for key in saved.files}


def main(stage='inner'):
    if stage not in RUNS:raise ValueError('Unknown amplitude stage')
    ledger_path=ROOT/'HYPOTHESES.md';ledger=ledger_path.read_text(encoding='utf8')
    if 'HYP-102' not in ledger:raise ValueError('HYP-102 must be registered before any parameter fit')
    parent=ROOT/'results/2026-10-24'
    out=parent/f'HYP-102-{1 if stage=="inner" else 2:02d}-{stage}-amplitude-sprint'
    out.mkdir(exist_ok=False)
    pool_run,transport_run=[parent/name for name in RUNS[stage]]
    configuration={'strategy':'cached_H91_minus_H69_correction_amplitude','version':1,
        'optimizer':'inner_no_intercept_analytic_least_squares','amplitude_bounds':[0.,2.],
        'zero_delta_amplitude':1.,'prediction':'max(0,H69+amplitude*(H91-H69))',
        'incremental_reference':'H91','primary_reference':'H69','strict_gain_epsilon':1e-10}
    evidence={'hypothesis':'HYP-102','stage':stage,'status':'started','started':timestamp(),
        'configuration':configuration,'code_sha256':_sha(__file__),
        'helper_source_sha256':{'research_memory.py':_sha(ROOT/'research_memory.py')},
        'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'learner_fits':0,'old_model_refits':0}
    atomic_json(out/'evidence.json',evidence);atomic_json(out/'configuration.json',configuration)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    history=Memory(out/'parameter-fit-history.json',ledger_path)
    claim=None;claim_finished=False
    try:
        inputs=[pool_run/'validation_predictions.npz',pool_run/'metrics.json',pool_run/'evidence.json',
            transport_run/'validation_predictions.npz',transport_run/'splits.npz',
            transport_run/'data.json',transport_run/'teacher-audit.json']
        frozen=None
        if stage=='outer':
            selection=parent/'HYP-102-01-inner-amplitude-sprint/metrics.json'
            inputs.append(selection);frozen=_read(selection)
            evidence['frozen_amplitude_source']=str(selection.relative_to(ROOT))
            evidence['frozen_amplitude_source_sha256']=_sha(selection)
        evidence['input_sha256']={str(path.relative_to(ROOT)):_sha(path) for path in inputs}
        hashes=_read(transport_run/'data.json')['hashes']
        aliases={'train':'train.csv','test':'test.csv','sample':'sample_submission.csv',
                 'dictionary':'data_dictionary.csv','task':'task.md'}
        expected={aliases.get(name,name):value for name,value in hashes.items()}
        observed={name:_sha(ROOT/'data/regression'/name) for name in expected}
        if expected!=observed:raise ValueError('Original data SHA mismatch')
        evidence['data_sha256']=observed
        atomic_json(out/'data.json',{'hashes':observed,'target_source':'cached_predictions_only'})
        pool,transport,split=[_arrays(path) for path in
            [pool_run/'validation_predictions.npz',transport_run/'validation_predictions.npz',transport_run/'splits.npz']]
        validate_cached(pool,transport,split)
        if _read(pool_run/'metrics.json').get('status')!='completed':
            raise ValueError('H91 source must be completed')
        evidence['split_sha256']={key:data_digest(np.asarray(split[key]).reshape(-1,1)) for key in ['fit','valid']}
        evidence['prediction_source_sha256']={str(path.relative_to(ROOT)):_sha(path) for path in inputs if path.suffix=='.npz'}
        (out/'splits.npz').write_bytes((transport_run/'splits.npz').read_bytes())
        atomic_json(out/'evidence.json',evidence) # All inputs frozen before Memory claim/optimizer.
        base,prediction_pool,y=transport['reference'],pool['prediction'],pool['y']
        amplitude,claim=select_amplitude(stage,prediction_pool,base,y,configuration,history,frozen)
        prediction=apply_amplitude(prediction_pool,base,amplitude)
        score=float(np.sqrt(np.mean((prediction-y)**2)))
        control=float(np.sqrt(np.mean((base-y)**2)))
        pool_score=float(np.sqrt(np.mean((prediction_pool-y)**2)))
        result={'status':'completed','hypothesis':'HYP-102','stage':stage,'metric':'rmse','score':score,
            'reference_score':control,'H91_reference_score':pool_score,'amplitude':amplitude,
            'incremental_gain_over_H91':pool_score-score,'eligible_for_outer':bool(score<pool_score-1e-10),
            'learner_fits':0,'old_model_refits':0,'parameter_fits':int(stage=='inner'),
            'fingerprint':claim,'configuration':configuration,'finished':timestamp()}
        np.savez_compressed(out/'validation_predictions.npz',positions=pool['positions'],y=y,prediction=prediction,
            reference=base,pool_reference=prediction_pool,delta=prediction_pool-base,amplitude=np.asarray(amplitude))
        atomic_json(out/'metrics.json',result)
        if claim:
            history.finish(claim,result);claim_finished=True
        evidence.update(status='completed',finished=result['finished'],amplitude=amplitude,
                        output_prediction_sha256=_sha(out/'validation_predictions.npz'))
    except Exception as exc:
        records=history.records()
        if claim is None and records:
            claim=records[-1]['fingerprint']
            claim_finished=records[-1].get('status')!='started'
        result={'status':'failed','hypothesis':'HYP-102','stage':stage,'error':type(exc).__name__+': '+str(exc),
                'learner_fits':0,'old_model_refits':0,'parameter_fits':int(claim is not None),
                'finished':timestamp(),'traceback':traceback.format_exc()}
        if claim and not claim_finished:
            history.finish(claim,result)
        atomic_json(out/'metrics.json',result)
        evidence.update(status='failed',error=result['error'],finished=result['finished'])
    atomic_json(out/'evidence.json',evidence)
    (out/'report.md').write_text('# H102 '+stage+'\n\n'+json.dumps(result,indent=2,ensure_ascii=False)+
        '\n\nCached predictions only; zero learner fits. Inner amplitude frozen before outer evaluation; '
        'viewed outer is exploratory. Primary reference H69; incremental improvement must beat H91.\n',encoding='utf8')
    for name in ['HYPOTHESES.md','results/README.md']:
        prefix='results/' if name=='HYPOTHESES.md' else ''
        relative=str(out.relative_to(parent)).replace('\\','/')
        record='HYP-102/'+stage+' '+result['status']
        if result['status']=='completed':record+=' RMSE='+str(result['score'])+' vsH91='+str(result['H91_reference_score'])+' amplitude='+str(result['amplitude'])
        else:record+=' '+result['error']
        with (ROOT/name).open('a',encoding='utf8') as stream:
            stream.write('\n'+record+' [report]('+prefix+'2026-10-24/'+relative+'/report.md). Повтор не запускать.\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['inner','outer'],default='inner')
    sys.exit(main(parser.parse_args().stage))
