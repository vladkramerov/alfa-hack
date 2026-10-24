"""One registered H82 evaluation per frozen outer protocol, total budget900s."""
from __future__ import annotations
import sys, os, json, time, hashlib, subprocess, traceback, platform, resource, shutil
from pathlib import Path

W = Path(__file__).resolve().parent
R = Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack')
sys.path[:0] = [str(W), str(R)]
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np, pandas as pd, torch, sklearn, joblib
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from automl_data import load_task, split_key
from learned_neighbors import LearnedNeighborClassifier, raw_groups, check_deadline

CONTROL = {'primary': .64472256711304, 'secondary': .6075743572072007}
torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def branch_guard():
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=R, text=True).strip() != 'sergey_branch':
        raise RuntimeError('Only sergey_branch is authorized')


def main():
    branch_guard()
    ledger = (R/'HYPOTHESES.md').read_text()
    card = (R/'docs/hypotheses/HYP-082.md').read_text()
    assert 'HYP-082' in ledger and '4096' in card
    b = load_task(R/'data/multiclass/task.md')
    all_results = []
    budget_start = time.monotonic()
    deadline = budget_start + 900
    for number, stage in enumerate(['primary', 'secondary'], 1):
        branch_guard()
        # Read complete journal immediately before this hypothesis stage.
        ledger = (R/'HYPOTHESES.md').read_text()
        out = R/f'results/2026-10-24/HYP-082-{number:02d}-learned-neighbors-{stage}'
        out.mkdir(parents=True, exist_ok=False)
        stage_start = time.monotonic()
        control = R/f'results/2026-10-24/sergey-improvement-v1/control-multiclass-{stage}'
        with np.load(control/'splits.npz') as source:
            fit, valid = source['fit'], source['valid']
        X, y = b.X.iloc[fit].reset_index(drop=True), np.asarray(b.y.iloc[fit], dtype=str)
        outer_X, outer_y = b.X.iloc[valid].reset_index(drop=True), np.asarray(b.y.iloc[valid], dtype=str)
        groups = raw_groups(X)
        inner_fit, inner_valid = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
        if set(groups[inner_fit]) & set(groups[inner_valid]):
            raise ValueError('Raw groups cross inner folds')
        np.savez_compressed(out/'splits.npz', fit=fit, valid=valid,
                             inner_fit=inner_fit, inner_valid=inner_valid, groups=groups)
        for filename in ['learned_neighbors.py', 'run_learned_neighbors.py', 'test_learned_neighbors.py']:
            shutil.copyfile(W/filename, out/filename)
        source_hashes = {name: hashlib.sha256((W/name).read_bytes()).hexdigest() for name in
                         ['learned_neighbors.py', 'run_learned_neighbors.py', 'test_learned_neighbors.py']}
        save(out/'evidence.json', {'hypothesis':82, 'stage':stage, 'data_hashes':b.hashes,
             'split_key':split_key(fit, valid), 'control':CONTROL[stage],
             'source_sha256':source_hashes, 'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),
             'card_sha256':hashlib.sha256(card.encode()).hexdigest(),
             'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
             'configuration':'fixed in HYP-082.md before fit', 'fit_rows':len(fit), 'valid_rows':len(valid),
             'inner_fit_rows':len(inner_fit), 'inner_valid_rows':len(inner_valid),
             'inner_groups_disjoint':True, 'outer_labels_used_for_fit':False,
             'maximum_hypothesis_seconds':900, 'total_budget_shared_between_protocols':True,
             'command':'python work/spec012/nca/run_learned_neighbors.py',
             'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                         'sklearn':sklearn.__version__,'torch':torch.__version__,'platform':platform.platform()}})
        (out/'report.md').write_text(f'# H82 / {stage}\n\nRegistered before fit in HYP-082.md and SPEC012. '
            'ModernNCA-inspired learned metric, not an exact paper replication and not fixed-distance H52. '
            'Inner grouped validation selects epochs; fresh outer-fit refit uses the frozen epoch count. '
            'Outer labels are used once for scoring only. Total900s across both protocols.\n', encoding='utf-8')
        save(out/'metrics.json', {'status':'started','stage':stage,'hypothesis':82})
        phase = 'inner_selection'
        try:
            check_deadline(deadline)
            with (out/'fit-events.jsonl').open('a') as event:
                event.write(json.dumps({'phase':phase,'status':'started','fit_rows':len(inner_fit)})+'\n')
            def progress(row):
                with (out/'trajectory.jsonl').open('a') as log:
                    log.write(json.dumps({'phase':phase,**row})+'\n')
                print('H82', stage, phase, json.dumps(row), 'elapsed', round(time.monotonic()-budget_start,2), flush=True)
            selection = LearnedNeighborClassifier().fit(X.iloc[inner_fit], y[inner_fit],
                validation=(X.iloc[inner_valid], y[inner_valid]), epochs=20, patience=4,
                deadline=deadline, progress=progress)
            epochs = selection.audit_['selected_epoch']
            save(out/'selection-audit.json', selection.audit_)
            joblib.dump(selection, out/'inner-model.joblib', compress=3)
            with (out/'fit-events.jsonl').open('a') as event:
                event.write(json.dumps({'phase':phase,'status':'completed','selected_epochs':epochs})+'\n')
            del selection
            phase = 'frozen_outer_refit'
            check_deadline(deadline)
            with (out/'fit-events.jsonl').open('a') as event:
                event.write(json.dumps({'phase':phase,'status':'started','fit_rows':len(X),'epochs':epochs})+'\n')
            model = LearnedNeighborClassifier().fit(X, y, epochs=epochs, deadline=deadline, progress=progress)
            save(out/'refit-audit.json', model.audit_)
            joblib.dump(model, out/'model.joblib', compress=3)
            with (out/'fit-events.jsonl').open('a') as event:
                event.write(json.dumps({'phase':phase,'status':'completed','epochs':epochs})+'\n')
            phase = 'outer_prediction'
            p = model.predict_proba(outer_X, deadline=deadline)
            classes, pred = model.classes_, model.classes_[p.argmax(1)]
            score = float(f1_score(outer_y, pred, labels=classes, average='macro', zero_division=0))
            np.savez_compressed(out/'validation_predictions.npz', y=outer_y, prediction=pred,
                probabilities=p, classes=classes, positions=valid)
            save(out/'per-class.json',classification_report(outer_y,pred,labels=classes,output_dict=True,zero_division=0))
            save(out/'confusion.json',{'classes':classes.tolist(),'matrix':confusion_matrix(outer_y,pred,labels=classes).tolist()})
            result = {'status':'completed','metric':'macro_f1','score':score,'stage':stage,
                'control':CONTROL[stage],'relative_gain':score/CONTROL[stage]-1,
                'accuracy':float(accuracy_score(outer_y,pred)), 'fit_rows':len(fit),'valid_rows':len(valid),
                'frozen_epochs':epochs,'model_fit_count':2,'seconds':time.monotonic()-stage_start,
                'total_seconds':time.monotonic()-budget_start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'conclusion':'Подтверждена в проверенных условиях' if score>CONTROL[stage] else 'Не подтверждена по заданному критерию'}
        except Exception as error:
            result = {'status':'interrupted' if isinstance(error,TimeoutError) else 'failed','stage':stage,
                'phase':phase,'error':repr(error),'seconds':time.monotonic()-stage_start,
                'total_seconds':time.monotonic()-budget_start,'score':None,
                'conclusion':'Недостаточно данных; технический исход, не отрицательная научная оценка'}
            (out/'traceback.txt').write_text(traceback.format_exc())
            with (out/'fit-events.jsonl').open('a') as event:
                event.write(json.dumps({'phase':phase,'status':result['status'],'error':repr(error)})+'\n')
        save(out/'metrics.json',result)
        with (out/'report.md').open('a') as report:
            report.write('\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
        print('H82_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
        all_results.append(result)
    return all_results


if __name__ == '__main__':
    main()
