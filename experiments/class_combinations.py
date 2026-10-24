"""Offline H46--48: new inner branches, strictly reused outer probabilities."""
from pathlib import Path
import json
import joblib
import numpy as np

from automl_advanced import _choose_multipliers, calibrate_probabilities
from automl_class_combo import (ClassBranch, COMPONENT_ORDER, aligned_probabilities,
                               fit_meta_state, predict_meta_state)
from experiments.twenty_advanced import inner_fingerprint, inner_folds, _write_npz


def _cached_component(component,X,y,context):
    """All cache identities include exact raw data, target and original positions."""
    y = np.asarray(y); classes = np.unique(y)
    positions = np.asarray(context['fit_positions'],dtype=np.int64)
    if component=='raw':
        config = {'policy':'legacy'}
        key = inner_fingerprint('multiclass',X,y,positions,config)
        directory = Path(context['legacy_cache_dir']) / key
        if not all((directory / f'fold-{number}.npz').is_file() for number in range(3)):
            raise FileNotFoundError('Frozen legacy raw OOF absent: '+str(directory))
    else:
        config = {'component':component,'protocol':'H38-bounded-smotenc-v1' if component=='smote' else 'H17-tagged-frequency-native-v1',
                  'parameters':context.get('component_parameters',{}).get(component,{})}
        key = inner_fingerprint('multiclass',X,y,positions,config)
        directory = Path(context['cache_dir']) / key
        directory.mkdir(parents=True,exist_ok=True)
    probabilities = np.full((len(y),len(classes)),np.nan)
    fold_id = np.full(len(y),-1,dtype=int)
    events = []
    for number,(fit,valid) in enumerate(inner_folds('multiclass',X,y)):
        file = directory / f'fold-{number}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as saved:
                for name,expected in [('fit',fit),('valid',valid),('fit_positions',positions[fit]),('valid_positions',positions[valid])]:
                    if name not in saved or not np.array_equal(saved[name],expected): raise ValueError('OOF cache identity mismatch: '+name)
                if not np.array_equal(saved['classes'].astype(str),classes.astype(str)): raise ValueError('Cached OOF classes mismatch')
                p = aligned_probabilities(saved['probabilities'],classes,classes)
                if p.shape!=(len(valid),len(classes)): raise ValueError('Cached OOF shape mismatch')
            status = 'reused'
            fold_metadata = directory / f'fold-{number}-audit.json'
            branch_audit = json.loads(fold_metadata.read_text(encoding='utf-8')) if fold_metadata.exists() else {}
        else:
            if component=='raw': raise RuntimeError('Forbidden legacy OOF refit')
            branch = ClassBranch(component,config['parameters'],context.get('threads',4)).fit(X.iloc[fit],y[fit])
            p = aligned_probabilities(branch.predict_proba(X.iloc[valid]),branch.classes_,classes)
            branch_audit = dict(branch.audit_)
            accepted = branch_audit.pop('accepted',[])
            generated_target = branch_audit.pop('generated_target',[])
            branch_audit['original_fit_positions'] = positions[fit].tolist()
            if component=='smote':
                synthetic = directory / f'fold-{number}-accepted-synthetic.csv.gz'
                branch.synthetic_.to_csv(synthetic,index=False,compression='gzip')
                selection = directory / f'fold-{number}-synthetic-selection.npz'
                _write_npz(selection,accepted=np.asarray(accepted,dtype=bool),
                           generated_target=np.asarray(generated_target,dtype=str),original_fit_positions=positions[fit])
                branch_audit.update(synthetic_artifact=str(synthetic),selection_artifact=str(selection))
            (directory/f'fold-{number}-audit.json').write_text(json.dumps(branch_audit,indent=2),encoding='utf-8')
            joblib.dump(branch,directory/f'fold-{number}-model.joblib',compress=3)
            _write_npz(file,fit=fit,valid=valid,fit_positions=positions[fit],valid_positions=positions[valid],
                       prediction=classes[np.argmax(p,axis=1)].astype(str),probabilities=p,classes=classes.astype(str))
            status = 'fitted'
        probabilities[valid] = p; fold_id[valid] = number
        events.append({'fold':number,'status':status,'fit_rows':len(fit),'validation_rows':len(valid),
                       'artifact':str(file),'model_artifact':str(directory/f'fold-{number}-model.joblib'),
                       'audit_artifact':str(directory/f'fold-{number}-audit.json') if component!='raw' else None,
                       'raw_groups_disjoint':True,'branch':branch_audit})
    if not np.isfinite(probabilities).all() or np.any(fold_id<0): raise ValueError('Incomplete component OOF')
    audit = {'key':key,'component':component,'config':config,'folds':events,
             'selection_source':'training_grouped_oof','raw_oof_refitted':False}
    if component!='raw': (directory/'metadata.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return {'probabilities':probabilities,'classes':classes,'fold_id':fold_id,'audit':audit}


def evaluate_class(number,X,y,Xvalid,context):
    """No outer targets are accepted; no outer estimator is fitted here."""
    if number not in (46,47,48): raise ValueError('Expected H46, H47 or H48')
    output = Path(context['output']); output.mkdir(parents=True,exist_ok=True)
    classes = np.unique(y)
    components = ['smote'] if number==46 else ['raw','frequency'] if number==47 else COMPONENT_ORDER
    oof = {c:_cached_component(c,X,y,context) for c in components}
    outer = {}
    for c in components:
        artifact = context['outer_components'][c]
        if 'positions' in artifact and not np.array_equal(artifact['positions'],context['valid_positions']):
            raise ValueError('Frozen outer probability positions mismatch')
        outer[c] = aligned_probabilities(artifact['probabilities'],artifact['classes'],classes)
        if outer[c].shape!=(len(Xvalid),len(classes)): raise ValueError('Frozen outer probability shape mismatch')
    audit = {'hypothesis':f'HYP-{number:03d}','components':{c:oof[c]['audit'] for c in components},
             'outer_fit':'reused_saved_component_probabilities','outer_model_fits':0,
             'selection_source':'training_grouped_oof','component_order':components}
    if number==48:
        state,meta_audit = fit_meta_state([oof[c]['probabilities'] for c in COMPONENT_ORDER],np.asarray(y),classes)
        probabilities = predict_meta_state([outer[c] for c in COMPONENT_ORDER],state,classes)
        learned_state = {'meta_state':state}; audit['meta'] = meta_audit
        strategy = 'probability_stacking'
    else:
        blended = sum(oof[c]['probabilities'] for c in components)/len(components)
        mapping,score = _choose_multipliers(np.asarray(y),blended,classes)
        probabilities = calibrate_probabilities(sum(outer.values())/len(components),classes,mapping)
        learned_state = {'class_multipliers':mapping}
        audit.update(inner_macro_f1=score,multipliers=mapping,calibration_grid=[.75,1.,1.25,1.5,2.],calibration_passes=1,
                     fixed_anchor=str(classes[0]),blend_weights={c:1/len(components) for c in components})
        strategy = 'smote_macro_f1' if number==46 else 'frequency_blend_macro_f1'
    candidate = {'strategy':strategy,'family':'xgboost','parameters':{},**learned_state}
    (output/'learned-state.json').write_text(json.dumps({'candidate':candidate,**learned_state},indent=2),encoding='utf-8')
    audit.update(learned_state=learned_state,learned_state_artifact=str(output/'learned-state.json'),strategy=strategy)
    return {'prediction':classes[np.argmax(probabilities,axis=1)],'probabilities':probabilities,
            'classes':classes,'audit':audit,'learned_state':learned_state,'candidate':candidate}
