"""SPEC-009 fixed classification evaluation; all selection inputs are fit-only."""
from pathlib import Path
import json
import numpy as np

from automl_class_combo import COMPONENT_ORDER,aligned_probabilities,predict_meta_state
from automl_next_class import build_next_class,mix_soft_targets
from experiments.twenty_advanced import inner_fingerprint,inner_folds

NUMBERS={49:'soft_distillation',51:'balanced_softmax',53:'nystrom_rbf',60:'masked_augmentation'}


def read_teacher_probabilities(X,y,context):
    """Strictly read-only loading: absent base OOF is an error, never a refit."""
    y=np.asarray(y);classes=np.unique(y);positions=np.asarray(context['fit_positions'],dtype=np.int64)
    components={};events=[]
    for component in COMPONENT_ORDER:
        config=({'policy':'legacy'} if component=='raw' else
                {'component':component,'protocol':'H38-bounded-smotenc-v1' if component=='smote' else 'H17-tagged-frequency-native-v1','parameters':{}})
        key=inner_fingerprint('multiclass',X,y,positions,config)
        directory=Path(context['legacy_cache_dir'] if component=='raw' else context['teacher_cache_dir'])/key
        folds=inner_folds('multiclass',X,y)
        if not all((directory/f'fold-{number}.npz').is_file() for number in range(len(folds))):
            raise FileNotFoundError('Required read-only teacher OOF absent: '+str(directory))
        p=np.full((len(X),len(classes)),np.nan)
        for number,(fit,valid) in enumerate(folds):
            file=directory/f'fold-{number}.npz'
            with np.load(file,allow_pickle=False) as saved:
                for field,expected in [('fit',fit),('valid',valid),('fit_positions',positions[fit]),('valid_positions',positions[valid])]:
                    if field not in saved or not np.array_equal(saved[field],expected):raise ValueError('Teacher OOF positional mismatch: '+field)
                if not np.array_equal(saved['classes'].astype(str),classes.astype(str)):raise ValueError('Teacher OOF class mismatch')
                proba=aligned_probabilities(saved['probabilities'],classes,classes)
                if proba.shape!=(len(valid),len(classes)):raise ValueError('Teacher OOF shape mismatch')
                p[valid]=proba
            events.append({'component':component,'fold':number,'key':key,'artifact':str(file),
                           'fit_rows':len(fit),'valid_rows':len(valid),'status':'reused_read_only','raw_groups_disjoint':True})
        if not np.isfinite(p).all():raise ValueError('Incomplete teacher base OOF')
        components[component]=p
    state=context['teacher_meta_state']
    probabilities=predict_meta_state([components[c] for c in COMPONENT_ORDER],state,classes)
    return probabilities,{'teacher_source':'in_sample_meta_on_base_grouped_oof',
                          'teacher_is_independent_oof':False,'teacher_refits':0,
                          'meta_fitted_on_same_base_oof_targets':True,'base_oof_artifacts':events,
                          'component_order':list(COMPONENT_ORDER)}


def evaluate_class(number,X,y,Xvalid,context):
    number=int(str(number).replace('HYP-',''))
    if number not in NUMBERS:raise ValueError('Unsupported SPEC-009 classification hypothesis')
    output=Path(context['output']);output.mkdir(parents=True,exist_ok=True)
    y=np.asarray(y);positions=np.asarray(context['fit_positions'],dtype=np.int64)
    if positions.shape!=(len(X),):raise ValueError('Training source positions mismatch')
    strategy=NUMBERS[number]
    candidate={'strategy':strategy,'parameters':dict(context.get('parameters',{})),
               'family':'logistic' if number==53 else 'xgboost'}
    model=build_next_class(candidate,context.get('threads',4))
    teacher_audit=None
    if number==49:
        teacher,teacher_audit=read_teacher_probabilities(X,y,context)
        classes=np.unique(y);labels=np.searchsorted(classes,y)
        q=mix_soft_targets(labels,teacher,len(classes))
        target_artifact=output/'distillation-targets.npz'
        np.savez_compressed(target_artifact,fit_positions=positions,classes=classes.astype(str),
                            teacher_probabilities=teacher,soft_targets=q,hard_labels=y.astype(str))
        model.fit(X,y,teacher_probabilities=teacher)
        teacher_audit['target_artifact']=str(target_artifact)
    else:model.fit(X,y)
    probabilities=model.predict_proba(Xvalid)
    prediction=model.classes_[np.argmax(probabilities,axis=1)]
    audit=dict(model.audit_)
    if teacher_audit is not None:audit['teacher']=teacher_audit
    training_artifact=output/'training-lineage.npz'
    rows=np.tile(positions,2) if number==60 else positions
    arrays={'source_fit_positions':rows,'original_fit_positions':positions,
            'is_augmented':np.r_[np.zeros(len(X),dtype=bool),np.ones(len(X),dtype=bool)] if number==60 else np.zeros(len(X),dtype=bool)}
    if number==53:
        arrays['landmark_positions']=positions[model.basis_.component_indices_]
        gamma_rows=np.arange(len(X))
        if len(X)>512:gamma_rows=np.random.RandomState(42).choice(gamma_rows,512,replace=False)
        arrays['gamma_fit_positions']=positions[gamma_rows]
    if number==60:
        fields=audit['augmentation']['numeric_fields']
        arrays['masked_numeric_fields']=np.asarray(fields,dtype=str)
        arrays['numeric_mask']=np.random.RandomState(42).random_sample((len(X),len(fields)))<.05
    np.savez_compressed(training_artifact,**arrays)
    audit['training_lineage_artifact']=str(training_artifact)
    (output/'training-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'prediction':prediction,'probabilities':probabilities,'classes':model.classes_,
            'audit':audit,'learned_state':model.learned_state_,'candidate':candidate}
