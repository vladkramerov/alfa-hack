"""Single-attempt H79/H80 runner; writes only dedicated new result paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from automl_data import load_task, split_key, validation_split
from paper_neural import CONFIG, DeadlineExceeded, check_deadline, probabilities, tensors, train

ROOT = Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack')
CONTROLS = dict(primary=0.64472256711304, secondary=0.6075743572072007)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')


def branch_guard():
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'sergey_branch':
        raise RuntimeError('Only sergey_branch is authorized')


def main(hypothesis):
    branch_guard()
    # Entire memory is read before every ID, including all historical outcomes.
    ledger = (ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    assert ('HYP-0'+str(hypothesis)) in ledger
    started = time.monotonic()
    deadline = started+900
    bundle = load_task(ROOT/'data/multiclass/task.md')
    groups = pd.util.hash_pandas_object(bundle.X, index=False).to_numpy()
    all_splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(bundle.X, bundle.y, groups))
    members = 1 if hypothesis == 79 else 8
    summaries = []
    for number, stage in enumerate(['primary', 'secondary'], start=1):
        branch_guard()
        out = ROOT/f'results/2026-10-24/HYP-{hypothesis:03d}-{number:02d}-neural-{stage}'
        out.mkdir(parents=True, exist_ok=False)
        fit, valid = all_splits[number-1]
        if number == 1:
            old_fit, old_valid, _ = validation_split(bundle)
            assert np.array_equal(old_fit, fit) and np.array_equal(old_valid, valid)
        relative_fit, relative_holdout = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            bundle.X.iloc[fit], bundle.y.iloc[fit], groups[fit]))
        inner_fit, inner_holdout = fit[relative_fit], fit[relative_holdout]
        assert not set(groups[fit]) & set(groups[valid])
        assert not set(groups[inner_fit]) & set(groups[inner_holdout])
        np.savez_compressed(out/'splits.npz', fit=fit, valid=valid, inner_fit=inner_fit, inner_holdout=inner_holdout)
        source = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        evidence = dict(hypothesis=f'HYP-{hypothesis:03d}', stage=stage, members=members,
                        config=CONFIG, hashes=bundle.hashes, split_key=split_key(fit, valid),
                        inner_split_key=split_key(inner_fit, inner_holdout),
                        ledger_sha256_before_fit=hashlib.sha256(ledger.encode()).hexdigest(),
                        source_sha256=source, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        python=sys.version, torch=torch.__version__, platform=platform.platform(),
                        control=CONTROLS[stage], total_budget_seconds=900,
                        elapsed_before_stage=time.monotonic()-started,
                        outer_score_evaluations=1, inner_selection='SGKF5 first fold seed42 within outer-fit',
                        source_urls=['https://arxiv.org/abs/2410.24210','https://arxiv.org/abs/2203.05556'])
        save(out/'evidence.json', evidence)
        (out/'ledger_before_fit.md').write_text(ledger, encoding='utf8')
        for p in Path(__file__).parent.glob('*.py'):
            shutil.copy2(p, out/p.name)
        (out/'report.md').write_text(f'# HYP-{hypothesis:03d} / {stage}\n\nSPEC012/PLAN012. Started before fit.\n\n'
            'Original small PLE neural implementation; BatchEnsemble-inspired for H80, not full TabM. '
            'No pretrained weights, external data, LLM calls or outer-label model selection. '
            'Both protocols share one 900-second budget.\n', encoding='utf8')
        stage_started = time.monotonic()
        trace_path = out/'epoch_trace.jsonl'
        def trace(phase):
            def callback(row):
                row = dict(phase=phase, elapsed_total=time.monotonic()-started, **row)
                with trace_path.open('a', encoding='utf8') as handle:
                    handle.write(json.dumps(row)+'\n')
                print(json.dumps(dict(hypothesis=hypothesis, stage=stage, **row)), flush=True)
            return callback
        try:
            check_deadline(deadline)
            classes = np.unique(bundle.y.iloc[fit].to_numpy(dtype=str))
            codec = {c:i for i,c in enumerate(classes)}
            encode = lambda y: np.array([codec[str(c)] for c in y], dtype=np.int64)
            save(out/'started.json',dict(status='started', wall_time=time.time(), phase='inner_selection'))
            model, preprocessor, selection = train(bundle.X.iloc[inner_fit], encode(bundle.y.iloc[inner_fit]), classes,
                members, deadline, trace('inner'), valid_frame=bundle.X.iloc[inner_holdout],
                valid_y=encode(bundle.y.iloc[inner_holdout]))
            save(out/'selection.json', selection)
            save(out/'inner_preprocessing.json', preprocessor.state())
            del model, preprocessor
            check_deadline(deadline)
            model, preprocessor, refit = train(bundle.X.iloc[fit], encode(bundle.y.iloc[fit]), classes,
                members, deadline, trace('refit'), epochs=selection['best_epoch'])
            # First and only use of this outer holdout for this candidate.
            prob = probabilities(model, tensors(preprocessor, bundle.X.iloc[valid]), deadline)
            pred = classes[prob.argmax(1)]
            y = bundle.y.iloc[valid].to_numpy(dtype=str)
            metric = float(f1_score(y, pred, average='macro', zero_division=0))
            np.savez_compressed(out/'validation_predictions.npz', positions=valid, y=y.astype(str),
                                prediction=pred.astype(str), probabilities=prob,
                                classes=classes.astype(str))
            save(out/'preprocessing.json', preprocessor.state())
            save(out/'refit.json', refit)
            save(out/'classification_report.json', classification_report(y,pred,output_dict=True,zero_division=0))
            torch.save(model.state_dict(), out/'model_state.pt')
            result = dict(status='completed', metric='macro_f1', score=metric, control=CONTROLS[stage],
                          relative_gain=metric/CONTROLS[stage]-1, accuracy=float(accuracy_score(y,pred)),
                          weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),
                          fit_rows=len(fit), valid_rows=len(valid),
                          frozen_epochs=selection['best_epoch'], selection=selection, refit=refit,
                          seconds=time.monotonic()-stage_started, total_seconds=time.monotonic()-started,
                          max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          conclusion='Подтверждена в проверенных условиях' if metric>CONTROLS[stage]
                          else 'Не подтверждена по заданному критерию')
        except Exception as exc:
            (out/'traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
            result = dict(status='interrupted' if isinstance(exc,DeadlineExceeded) else 'failed',
                          error=type(exc).__name__+': '+str(exc), seconds=time.monotonic()-stage_started,
                          total_seconds=time.monotonic()-started, conclusion='Недостаточно данных')
        branch_guard()
        save(out/'metrics.json', result)
        with (out/'report.md').open('a',encoding='utf8') as handle:
            handle.write('\n## Result\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
        print('RESULT '+json.dumps(dict(hypothesis=hypothesis,stage=stage,path=str(out),**result),ensure_ascii=False), flush=True)
        summaries.append(dict(stage=stage,path=str(out),**result))
        if result['status'] != 'completed':
            break
    print('SUMMARY '+json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis',type=int,choices=[79,80],required=True)
    main(parser.parse_args().hypothesis)
