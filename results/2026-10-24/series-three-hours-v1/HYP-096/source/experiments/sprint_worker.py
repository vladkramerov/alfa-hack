"""Frozen SPEC011 worker; known predictions/teachers are never retrained."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from research_memory import Memory, atomic_json


class SavedBaseline:
    def __init__(self, positions, prediction):
        self.lookup = dict(zip(map(int, positions), map(float, prediction)))

    def predict(self, X):
        if any(int(i) not in self.lookup for i in X.index):
            raise ValueError('Cached baseline does not cover query positions')
        return np.asarray([self.lookup[int(i)] for i in X.index])


def run(request):
    started = time.monotonic()
    out = Path(request['output'])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(request['repo_root'])
    number, stage = request['number'], request['stage']
    history = Memory(request['history'], root / 'HYPOTHESES.md')
    fingerprint = None
    try:
        task_name = 'regression' if number in [70, 71, 76, 81, 83,85,86,89,90] else 'multiclass'
        if number in [70, 71, 76, 81, 83,85,86,89,90] and stage == 'inner':
            from research_data import load_task
        else:
            from automl_data import load_task
        bundle = load_task(root / 'data' / task_name / 'task.md')
        atomic_json(out / 'data.json', {'task_path': str(bundle.task_path), 'hashes': bundle.hashes,
                                       'target_excluded_from_X': True, 'id_excluded_from_X': bundle.id_column not in bundle.X})
        teacher = None
        if task_name == 'regression':
            if stage == 'inner':
                source = root / 'results/2026-10-24/HYP-062-03-regression-control-rewrite'
            else:
                source = root / 'results/2026-10-24/HYP-069-01-regression-alternatives'
            with np.load(source / 'splits.npz', allow_pickle=False) as split:
                fit, valid = split['fit'], split['valid']
            with np.load(source / 'validation_predictions.npz', allow_pickle=False) as old:
                if not np.array_equal(valid, old['positions']):
                    raise ValueError('Cached regression baseline positions mismatch')
                reference = old['prediction'].copy()
            from sprint_regression import TemporalEntityLinker, RatingNormalizedRegressor
            if number == 76:
                from sprint_panel import PanelTransitionRegressor
                model = PanelTransitionRegressor(**request['parameters'])
            elif number == 83:
                from sprint_ratio_transport import OriginalLossTransport
                model = OriginalLossTransport(**request['parameters'])
            elif number == 85:
                from sprint_uncertainty import ConditionalLognormalRegressor
                model = ConditionalLognormalRegressor(**request['parameters'])
            elif number == 89:
                from sprint_differences import PanelDifferenceRegressor
                model = PanelDifferenceRegressor(**request['parameters'])
            else:
                model = TemporalEntityLinker(**request['parameters']) if number == 70 else RatingNormalizedRegressor(threads=4)
            kwargs = {'base_predictor': SavedBaseline(valid, reference)} if number in [70, 76, 83,89] else {}
            teacher_audit = {'baseline_source': str(source), 'old_model_refits': 0}
            if number in [81,86,90]:
                h70 = root / ('results/2026-10-24/HYP-070-03-inner-high-sprint' if stage == 'inner' else 'results/2026-10-24/HYP-070-04-outer-high-sprint')
                with np.load(h70 / 'validation_predictions.npz', allow_pickle=False) as cache:
                    if not np.array_equal(cache['positions'], valid):raise ValueError('H70 baseline positions mismatch')
                    inherited = cache['prediction'].copy()
                state = json.loads((h70 / 'audit.json').read_text(encoding='utf8'))
                if number == 81:
                    from sprint_linkage import FuzzyTemporalOverlay
                    model = FuzzyTemporalOverlay(**request['parameters'])
                elif number == 86:
                    from sprint_state_space import RatingStateSpaceHistory
                    model = RatingStateSpaceHistory(**request['parameters'])
                else:
                    from sprint_missing_expert import MissingRatingExpert
                    def fit_expert(estimator,X,y,configuration,**fit_kwargs):
                        expert_memory=Memory(out/'expert-fits.json',root/'HYPOTHESES.md')
                        claim=expert_memory.claim(configuration,X,y,stage)
                        try:
                            estimator.fit(X,y,**fit_kwargs)
                            expert_memory.finish(claim,{'status':'completed'})
                        except Exception as exc:
                            expert_memory.finish(claim,{'status':'failed','error':type(exc).__name__})
                            raise
                        return estimator
                    model = MissingRatingExpert(fit_hook=fit_expert,**request['parameters'])
                kwargs = {'base_predictor': SavedBaseline(valid, inherited)}
                if number in [81,86]:kwargs['transport_state']=state
                teacher_audit.update(inherited_baseline_source=str(h70),transport_state_sha256=hashlib.sha256((h70 / 'audit.json').read_bytes()).hexdigest(),
                                     inherited_prediction_sha256=hashlib.sha256(inherited.tobytes()).hexdigest(),old_model_refits=0)
        else:
            from experiments.sprint_cache import load_class_inner, load_class_outer
            if stage == 'inner':
                teacher = load_class_inner(bundle, root=root)
                fit, valid = teacher['fit_positions'], teacher['valid_positions']
            else:
                with np.load(root / 'results/2026-10-23/HYP-048-01-multiclass-combinations/splits.npz', allow_pickle=False) as split:
                    fit, valid = split['fit'], split['valid']
                teacher = load_class_outer(bundle, fit, valid, root=root)
            reference = teacher['classes'][teacher['valid_probabilities'].argmax(axis=1)]
            teacher_audit = teacher['audit']
            if number == 88:
                from experiments.sprint_crossfit_cache import load_honest_teacher
                def fit_teacher(estimator,X,y,configuration):
                    nested_memory=Memory(out/'teacher-fits.json',root/'HYPOTHESES.md')
                    claim=nested_memory.claim(configuration,X,y,stage)
                    try:
                        estimator.fit(X,y)
                        nested_memory.finish(claim,{'status':'completed'})
                    except Exception as exc:
                        nested_memory.finish(claim,{'status':'failed','error':type(exc).__name__})
                        raise
                    return estimator
                teacher=load_honest_teacher(bundle,stage,fit,valid,root=root,output=out,fit_hook=fit_teacher)
                teacher_audit={**teacher['audit'],'comparison_baseline':'cached_H48_outer_or_RAW_inner',
                               'comparison_baseline_audit':teacher_audit}
            if number in [92,93,94,96,97]:
                from experiments.sprint_honest_reuse import load_reused_honest_teacher
                teacher=load_reused_honest_teacher(bundle,stage,fit,valid,root=root,output=out)
                teacher_audit={**teacher['audit'],'comparison_baseline':'cached_H48_outer_or_RAW_inner',
                               'comparison_baseline_audit':teacher_audit}
            if number == 72:
                from sprint_classification import TeacherResidualF1Classifier
                model = TeacherResidualF1Classifier(**request['parameters'])
            elif number == 73:
                from sprint_classification import ECOCClassifier
                model = ECOCClassifier(**request['parameters'])

                def fit_head(estimator, X, y, configuration, **fit_kwargs):
                    identity = {**configuration, 'hypothesis': 'HYP-073',
                                'weight_sha256': hashlib.sha256(np.asarray(fit_kwargs['sample_weight'], dtype='<f8').tobytes()).hexdigest()}
                    head_memory = Memory(out / 'head-fits.json', root / 'HYPOTHESES.md')
                    claim = head_memory.claim(identity, X, y, stage)
                    try:
                        estimator.fit(X, y, **fit_kwargs)
                        head_memory.finish(claim, {'status': 'completed'})
                    except Exception as exc:
                        head_memory.finish(claim, {'status': 'failed', 'error': type(exc).__name__})
                        raise
                    return estimator

                model.fit_hook = fit_head
            elif number == 74:
                from sprint_contextual import ContextualTeacherCorrector
                model = ContextualTeacherCorrector(**request['parameters'])
            elif number == 75:
                from sprint_decision import MacroF1Decision
                model = MacroF1Decision(**request['parameters'])
            elif number == 77:
                from sprint_ambitious import LovaszTeacherBoost
                model = LovaszTeacherBoost(**request['parameters'])
            elif number == 78:
                from sprint_posterior import CalibratedMacroDecision
                model = CalibratedMacroDecision(**request['parameters'])
            elif number == 80:
                from sprint_softboost import MacroTeacherBoost
                model = MacroTeacherBoost(**request['parameters'])
            elif number == 50:
                from sprint_focal import FocalTeacherBoost
                model = FocalTeacherBoost(**request['parameters'])
            elif number == 82:
                from sprint_network import NetworkSemanticTeacherCorrector
                model = NetworkSemanticTeacherCorrector(**request['parameters'])
            elif number == 84:
                from sprint_signatures import SignaturePosteriorClassifier
                model = SignaturePosteriorClassifier(**request['parameters'])
            elif number == 87:
                from sprint_cohort import CohortMacroDecision
                source_state = root / ('results/2026-10-24/HYP-078-01-inner-default-sprint/audit.json' if stage == 'inner' else 'results/2026-10-24/HYP-078-02-outer-default-sprint/audit.json')
                state = json.loads(source_state.read_text(encoding='utf8'))
                model = CohortMacroDecision(calibration_state={'theta':state['theta']}, **request['parameters'])
                teacher_audit = {**teacher_audit,'calibration_state_source':str(source_state),
                                 'calibration_state_sha256':hashlib.sha256(source_state.read_bytes()).hexdigest()}
            elif number == 88:
                from sprint_honest import HonestContextCorrector
                model = HonestContextCorrector(**request['parameters'])
            elif number in [92,93,94]:
                from sprint_honest_objectives import HonestMacroTeacherBoost,HonestLovaszTeacherBoost,HonestResidualF1Classifier
                factory={92:HonestMacroTeacherBoost,93:HonestLovaszTeacherBoost,94:HonestResidualF1Classifier}[number]
                model=factory(**request['parameters'])
            elif number == 96:
                from sprint_posterior import CalibratedMacroDecision
                model=CalibratedMacroDecision(**request['parameters'])
            elif number == 97:
                from sprint_decision import MacroF1Decision
                model=MacroF1Decision(**request['parameters'])
            else:
                raise ValueError('Unregistered sprint hypothesis')
            kwargs = {'teacher_probabilities': teacher['train_probabilities']}
        if set(fit) & set(valid):
            raise ValueError('Training and validation overlap')
        np.savez_compressed(out / 'splits.npz', fit=fit, valid=valid)
        atomic_json(out / 'teacher-audit.json', teacher_audit)
        configuration = model.configuration() if callable(model.configuration) else model.configuration
        if number in [81,86,90]:
            configuration = {**configuration,'transport_state_sha256':teacher_audit['transport_state_sha256'],
                             'inherited_prediction_sha256':teacher_audit['inherited_prediction_sha256']}
        if teacher is not None:
            configuration = {**configuration, 'teacher_train_sha256': hashlib.sha256(teacher['train_probabilities'].tobytes()).hexdigest(),
                             'teacher_classes': teacher['classes'].tolist()}
        atomic_json(out / 'configuration.json', configuration)
        fingerprint = history.claim(configuration, bundle.X.iloc[fit], bundle.y.iloc[fit], stage)
        model.fit(bundle.X.iloc[fit], bundle.y.iloc[fit], **kwargs)
        if teacher is not None:
            probabilities = model.predict_proba(bundle.X.iloc[valid], teacher_probabilities=teacher['valid_probabilities'])
            prediction = model.predict(bundle.X.iloc[valid], teacher_probabilities=teacher['valid_probabilities'])
        else:
            prediction = model.predict(bundle.X.iloc[valid])
        from sklearn.metrics import f1_score, mean_squared_error
        y = bundle.y.iloc[valid].to_numpy()
        measure = (lambda p: float(np.sqrt(mean_squared_error(y, p)))) if task_name == 'regression' else (lambda p: float(f1_score(y, p, average='macro', zero_division=0)))
        score, ref_score = measure(prediction), measure(reference)
        saved = {'positions': valid, 'prediction': prediction, 'y': y, 'reference': reference}
        if teacher is not None:
            saved.update(y=y.astype(str), prediction=np.asarray(prediction, dtype=str), reference=np.asarray(reference, dtype=str),
                         probabilities=probabilities, classes=np.asarray(model.classes_, dtype=str))
        np.savez_compressed(out / 'validation_predictions.npz', **saved)
        atomic_json(out / 'audit.json', getattr(model, 'audit_', {}))
        if hasattr(model, 'last_prediction_audit_'):
            atomic_json(out / 'prediction-audit.json', model.last_prediction_audit_)
        import joblib
        joblib.dump(model, out / 'model.joblib', compress=3)
        result = {'status': 'completed', 'hypothesis': f'HYP-{number:03d}', 'stage': stage,
                  'metric': 'rmse' if task_name == 'regression' else 'macro_f1', 'score': score, 'reference_score': ref_score,
                  'relative_gain': (ref_score - score) / ref_score if task_name == 'regression' else (score - ref_score) / ref_score,
                  'fit_rows': len(fit), 'valid_rows': len(valid), 'fingerprint': fingerprint,
                  'seconds': time.monotonic() - started, 'old_baseline_refits': 0,
                  'inner_is_full_H48_pipeline_validation': False if teacher is not None and stage == 'inner' else None}
        history.finish(fingerprint, result)
    except Exception as exc:
        result = {'status': 'failed', 'hypothesis': f'HYP-{number:03d}', 'stage': stage,
                  'error': type(exc).__name__ + ': ' + str(exc), 'seconds': time.monotonic() - started}
        (out / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
        if fingerprint is not None:
            history.finish(fingerprint, result)
    atomic_json(out / 'metrics.json', result)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(run(json.loads(Path(sys.argv[-1]).read_text(encoding='utf8'))))
