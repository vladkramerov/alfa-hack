"""Read-only numerical/source/data audit of both completed H82 evaluations."""
import sys, json, hashlib
from pathlib import Path
R=Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack')
sys.path.insert(0,str(R))
import numpy as np, joblib, torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from automl_data import load_task
from learned_neighbors import raw_groups

torch.set_num_threads(4)
b=load_task(R/'data/multiclass/task.md')
records=[]
for number,stage in enumerate(['primary','secondary'],1):
    out=R/f'results/2026-10-24/HYP-082-{number:02d}-learned-neighbors-{stage}'
    metrics=json.loads((out/'metrics.json').read_text())
    evidence=json.loads((out/'evidence.json').read_text())
    selected=json.loads((out/'selection-audit.json').read_text())
    refit=json.loads((out/'refit-audit.json').read_text())
    with np.load(out/'validation_predictions.npz') as file:
        predictions={name:file[name] for name in file.files}
    with np.load(out/'splits.npz') as file:
        splits={name:file[name] for name in file.files}
    checks={}
    checks['data_sha256']=evidence['data_hashes']==b.hashes
    checks['source_sha256']=all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
                                  for name,digest in evidence['source_sha256'].items())
    fit,valid=splits['fit'],splits['valid']
    X=b.X.iloc[fit].reset_index(drop=True)
    y=np.asarray(b.y.iloc[fit],dtype=str)
    checks['outer_positions']=np.array_equal(predictions['positions'],valid)
    checks['outer_targets']=np.array_equal(predictions['y'],np.asarray(b.y.iloc[valid],dtype=str))
    checks['outer_rows_disjoint']=not np.intersect1d(fit,valid).size
    groups=raw_groups(X)
    checks['group_hashes']=np.array_equal(groups,splits['groups'])
    checks['inner_group_disjoint']=not np.intersect1d(groups[splits['inner_fit']],groups[splits['inner_valid']]).size
    expected=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,y,groups))
    checks['frozen_inner_protocol']=np.array_equal(expected[0],splits['inner_fit']) and np.array_equal(expected[1],splits['inner_valid'])
    p,classes=predictions['probabilities'],predictions['classes']
    checks['probability_shape']=p.shape==(len(valid),len(classes))
    checks['probability_valid']=bool(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-7))
    checks['argmax_labels']=np.array_equal(classes[p.argmax(1)],predictions['prediction'])
    score=float(f1_score(predictions['y'],predictions['prediction'],labels=classes,average='macro',zero_division=0))
    checks['exact_macro_f1']=score==metrics['score']
    checks['exact_accuracy']=float(accuracy_score(predictions['y'],predictions['prediction']))==metrics['accuracy']
    best=max(selected['training_trace'],key=lambda row:row['inner_macro_f1'])['epoch']
    checks['epochs_only_inner']=selected['selected_epoch']==best==metrics['frozen_epochs']==refit['epochs_run']
    checks['budget']=metrics['total_seconds']<=900
    model=joblib.load(out/'model.joblib')
    checks['model_fit_rows']=len(model.values_)==len(fit)==model.preprocessor_.audit_['fit_rows']
    checks['model_memory_shape']=tuple(model.memory_.shape)==(len(fit),32)
    recomputed=model.predict_proba(b.X.iloc[valid[:16]])
    checks['serialized_prediction_reproduced']=bool(np.allclose(recomputed,p[:16],atol=3e-6,rtol=1e-5))
    record={'stage':stage,'recomputed_macro_f1':score,'checks':checks,'all_passed':all(checks.values()),
            'inner_checkpoint_note':'inner-model.joblib stores last selection epoch; scored outer model uses fresh frozen-best-epoch refit'}
    assert record['all_passed'],record
    records.append(record)
    del model
print(json.dumps({'hypothesis':82,'public_model_fits_in_verification':0,'records':records,'all_passed':all(r['all_passed'] for r in records)},ensure_ascii=False,indent=2))
