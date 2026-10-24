"""H65: eight fixed metamorphic checks, synthetic data and zero estimator fits."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta

import numpy as np
import pandas as pd
from automl_data import load_task,make_submission,atomic_submission

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/2026-10-24/HYP-065-01-schema-semantics'
CASE_NAMES={
 'S01':'train_row_and_feature_column_reordering',
 'S02':'test_permutation_and_inverse_id_alignment',
 'S03':'consistent_target_id_feature_renaming',
 'S04':'lexical_class_label_bijection',
 'S05':'numeric_looking_class_labels_preserve_lexical_values',
 'S06':'probability_class_order_to_sample_column_mapping',
 'S07':'opaque_leading_zero_identifier_roundtrip',
 'S08':'target_unit_prediction_roundtrip_1024'}


def fixture(folder,classification=False,labels=None,ids=None,rename=None,
            feature_order=None,train_order=None,test_order=None,probabilities=False,unit=1.):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    names={'record_id':'record_id','signal':'signal','kind':'kind','target':'target'}
    names.update(rename or {})
    raw_labels=np.array(labels if labels is not None else ['red','blue'],dtype=object)
    train=pd.DataFrame({'record_id':np.arange(100,108),'signal':np.arange(8,dtype=float),
                        'kind':['a','b']*4,'target':np.tile(raw_labels,4) if classification else np.arange(8,dtype=float)*2.5*unit})
    test=pd.DataFrame({'record_id':ids if ids is not None else [91,23,74],
                       'signal':[4.,1.,6.],'kind':['b','new','a']})
    if ids is not None:train['record_id']=[f'train-{i:04d}' for i in range(8)]
    if probabilities:
        sample=pd.DataFrame({'record_id':test.record_id,raw_labels[1]:np.zeros(3),raw_labels[0]:np.zeros(3)})
    else:sample=pd.DataFrame({'record_id':test.record_id,'target':np.repeat(raw_labels[0],3) if classification else np.zeros(3)})
    if train_order is not None:train=train.iloc[train_order].reset_index(drop=True)
    if test_order is not None:
        test=test.iloc[test_order].reset_index(drop=True);sample=sample.iloc[test_order].reset_index(drop=True)
    if feature_order is not None:
        train=train[['record_id',*feature_order,'target']];test=test[['record_id',*feature_order]]
    train=train.rename(columns=names);test=test.rename(columns=names);sample=sample.rename(columns=names)
    train.to_csv(folder/'train.csv',index=False);test.to_csv(folder/'test.csv',index=False);sample.to_csv(folder/'sample_submission.csv',index=False)
    dictionary=pd.DataFrame({'feature':[names[c] for c in ['record_id','signal','kind','target']],
                             'role':['id','feature','feature','target'],
                             'dtype':['str' if ids is not None else 'int','float','category','str' if classification else 'float']})
    dictionary.to_csv(folder/'data_dictionary.csv',index=False)
    metric='Log Loss' if probabilities else 'Macro F1' if classification else 'RMSE'
    task=folder/'task.md'
    task.write_text(('Multiclass classification.' if classification else 'Regression.')+f'\nMetric: {metric}.\n'
                    'Inputs: `train.csv`, `test.csv`, `sample_submission.csv`, `data_dictionary.csv`.\nSave submission.csv next to task.md.\n',encoding='utf-8')
    return load_task(task)


def _write(bundle,prediction):
    frame=make_submission(bundle,prediction)
    destination=bundle.task_path.parent/'submission.csv'
    atomic_submission(bundle,frame,destination)
    return frame,pd.read_csv(destination)


def run_case(case,folder):
    """One prescribed transformation; exceptions remain actual diagnostic failures."""
    folder=Path(folder)
    if case=='S01':
        base=fixture(folder/'base');changed=fixture(folder/'transformed',feature_order=['kind','signal'],train_order=[7,1,5,0,3,6,2,4])
        pd.testing.assert_frame_equal(base.train.sort_values('record_id').reset_index(drop=True),
            changed.train[base.train.columns].sort_values('record_id').reset_index(drop=True))
        answer,_=_write(changed,np.array([3.25,1.5,9.]))
        np.testing.assert_array_equal(answer.record_id,base.test.record_id)
        return {'feature_order':changed.X.columns.tolist(),'restored_train_rows':len(base.train)}
    if case=='S02':
        base=fixture(folder/'base');permutation=np.array([2,0,1]);changed=fixture(folder/'transformed',test_order=permutation)
        prediction=np.array([3.25,1.5,9.]);answer,restored=_write(changed,prediction[permutation])
        got=restored.set_index('record_id').loc[base.test.record_id,'target'].to_numpy()
        np.testing.assert_allclose(got,prediction,rtol=0,atol=1e-12)
        return {'permutation':permutation.tolist(),'original_ids':base.test.record_id.tolist(),'written_ids':answer.record_id.tolist()}
    if case=='S03':
        mapping={'record_id':'opaque_key','target':'response_value','signal':'measurement'}
        changed=fixture(folder/'transformed',rename=mapping)
        assert changed.target=='response_value' and changed.id_column=='opaque_key'
        assert 'measurement' in changed.X and 'response_value' not in changed.X and 'opaque_key' not in changed.X
        answer,_=_write(changed,np.array([3.25,1.5,9.]))
        restored=answer.rename(columns={v:k for k,v in mapping.items()})
        np.testing.assert_array_equal(restored.record_id,[91,23,74])
        np.testing.assert_allclose(restored.target,[3.25,1.5,9.],rtol=0,atol=1e-12)
        return {'rename':mapping,'parsed_target':changed.target,'parsed_id':changed.id_column}
    if case in {'S04','S05'}:
        transformed=['класс-А','β-метка'] if case=='S04' else ['001','010']
        changed=fixture(folder/'transformed',classification=True,labels=transformed)
        actual=set(changed.y.tolist());expected=set(transformed)
        if actual!=expected:raise AssertionError(f'Class label parsing changed lexical values: expected={transformed!r}, actual={sorted(actual)!r}, dtype={changed.y.dtype}')
        original=['blue','red','blue'];mapping={'red':transformed[0],'blue':transformed[1]}
        prediction=np.array([mapping[label] for label in original]);_,roundtrip=_write(changed,prediction)
        inverse={v:k for k,v in mapping.items()}
        restored=[inverse[label] for label in roundtrip.target]
        assert restored==original
        return {'bijection':mapping,'restored_predictions':restored}
    if case=='S06':
        bundle=fixture(folder/'transformed',classification=True,probabilities=True)
        source_classes=['red','blue'];p=np.array([[.2,.8],[.9,.1],[.45,.55]])
        outputs=[c for c in bundle.sample if c!=bundle.id_column]
        mapping={str(label):i for i,label in enumerate(source_classes)}
        mapped=p[:,[mapping[column] for column in outputs]]
        _,roundtrip=_write(bundle,mapped)
        for label,index in mapping.items():np.testing.assert_allclose(roundtrip[label],p[:,index],rtol=0,atol=1e-12)
        return {'model_classes':source_classes,'sample_output_columns':outputs,'explicit_mapping':mapping}
    if case=='S07':
        original=['0007','0012','0100'];bundle=fixture(folder/'transformed',ids=original)
        actual=bundle.test[bundle.id_column].astype(str).tolist()
        if actual!=original:raise AssertionError(f'Identifier parsing lost original representation: expected={original!r}, actual={actual!r}')
        _,roundtrip=_write(bundle,np.array([3.25,1.5,9.]))
        assert roundtrip[bundle.id_column].astype(str).tolist()==original
        return {'original_ids':original,'roundtrip_ids':actual}
    if case=='S08':
        bundle=fixture(folder/'transformed',unit=1024.)
        prediction=np.array([.125,1.5,9.25]);_,roundtrip=_write(bundle,prediction*1024.)
        restored=roundtrip.target.to_numpy()/1024.
        np.testing.assert_allclose(restored,prediction,rtol=0,atol=1e-12)
        return {'unit_factor':1024.,'original_predictions':prediction.tolist(),'restored_predictions':restored.tolist(),
                'training_or_metric_comparison_performed':False}
    raise ValueError('Unknown fixed scenario: '+case)


def main(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'evidence.json').exists():raise RuntimeError('This diagnostic run already has evidence; do not overwrite its history')
    started=time.monotonic();stamp=datetime.now(timezone(timedelta(hours=3))).isoformat()
    source_names=['automl_data.py','experiments/check_schema_semantics.py','tests/test_schema_semantics.py']
    before={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    cases=[]
    for case,name in CASE_NAMES.items():
        try:
            detail=run_case(case,output/'cases'/case);record={'id':case,'name':name,'status':'passed','detail':detail}
        except Exception as exc:
            record={'id':case,'name':name,'status':'failed','error':str(exc),'exception':type(exc).__name__}
            (output/(case+'-traceback.txt')).write_text(traceback.format_exc(),encoding='utf-8')
        cases.append(record)
    artifact_hashes={str(path.relative_to(output)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((output/'cases').rglob('*')) if path.is_file()}
    result={'hypothesis':'HYP-065','started':stamp,'status':'completed','conclusion':'Подтверждена в проверенных условиях' if all(c['status']=='passed' for c in cases) else 'Не подтверждена по заданному критерию',
            'case_count':len(cases),'passed':sum(c['status']=='passed' for c in cases),'failed':sum(c['status']=='failed' for c in cases),
            'model_fits':0,'public_metric_evaluations':0,'http_requests':0,'scientific_independent_metric_test':False,
            'sources':before,'sources_unchanged':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==digest for n,digest in before.items()),
            'ledger_sha256_before_check':hashlib.sha256((ROOT/'HYPOTHESES.md').read_bytes()).hexdigest(),
            'cases':cases,'artifact_hashes':artifact_hashes,'seconds':time.monotonic()-started,
            'runtime':{'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'platform':platform.platform()}}
    (output/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['hypothesis','status','passed','failed','model_fits','conclusion']},ensure_ascii=False))
    return result


if __name__=='__main__':main()
