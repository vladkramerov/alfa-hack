from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
"""H81: missing neural OOF calibration only, no outer scoring or meta fit."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from automl_data import load_task,split_key
from paper_neural import CONFIG,DeadlineExceeded,check_deadline,probabilities,tensors,train

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/2026-10-24'
TREE_CACHE=BASE/'sergey-scientific-v2'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def guard():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='sergey_branch'


def main(stage):
    guard()
    ledger=(ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    assert 'HYP-081' in ledger and 'HYP-081' in (ROOT/'docs/hypotheses/HYP-081.md').read_text()
    number=1 if stage=='primary' else 2
    primary_out=BASE/'HYP-081-01-neural-stack-primary'
    out=BASE/f'HYP-081-{number:02d}-neural-stack-{stage}'
    cache=TREE_CACHE/f'oof-cache-{stage}'
    assert (cache/'calibration.npz').is_file() and (cache/'folds.json').is_file(), 'Required tree OOF cache is not ready'
    if stage=='primary':
        started=time.monotonic()
    else:
        started=json.loads((primary_out/'budget.json').read_text())['monotonic_started']
    # The final head belongs to the parent; reserve 60 of the total 900 sec.
    deadline=started+840
    out.mkdir(parents=True,exist_ok=False)
    save(out/'budget.json',dict(monotonic_started=started,total_seconds=900,
                               oof_deadline_monotonic=deadline,head_reserve_seconds=60))
    stage_started=time.monotonic()
    try:
        b=load_task(ROOT/'data/multiclass/task.md')
        groups=pd.util.hash_pandas_object(b.X,index=False).to_numpy()
        calibration=np.load(cache/'calibration.npz',allow_pickle=False)
        folds=json.loads((cache/'folds.json').read_text())
        assert len(folds)==3
        fit=calibration['fit_positions']
        valid=calibration['valid_positions']
        classes=calibration['classes'].astype(str)
        y=calibration['y'].astype(str)
        assert np.array_equal(y,b.y.iloc[fit].to_numpy(dtype=str))
        assert np.array_equal(classes,np.unique(y))
        codec={value:i for i,value in enumerate(classes)}
        encode=lambda values:np.array([codec[str(v)] for v in values],dtype=np.int64)
        reference=BASE/f'HYP-079-{number:02d}-neural-{stage}/validation_predictions.npz'
        outer_nn=np.load(reference,allow_pickle=False)
        assert np.array_equal(outer_nn['positions'],valid)
        assert np.array_equal(outer_nn['classes'],classes)
        source_paths=[Path(__file__),Path(__file__).parent/'paper_neural.py']
        evidence=dict(hypothesis='HYP-081',stage=stage,role='missing_neural_grouped_OOF_only',
                      config=CONFIG,members=1,hashes=b.hashes,outer_split_key=split_key(fit,valid),
                      tree_cache=str(cache.relative_to(ROOT)),
                      tree_cache_sha256={n:hashlib.sha256((cache/n).read_bytes()).hexdigest()
                                         for n in ['calibration.npz','folds.json']},
                      reused_full_fit_prediction=str(reference.relative_to(ROOT)),
                      reused_full_fit_prediction_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                      ledger_sha256_before_fit=hashlib.sha256(ledger.encode()).hexdigest(),
                      commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      python=sys.version,torch=torch.__version__,platform=platform.platform(),
                      inner_protocol='SGKF5 first fold seed42 inside each fixed tree OOF fit',
                      training_model_fits_planned=6,outer_model_fits=0,outer_score_evaluations=0,
                      command='env PYTHONPATH='+str(ROOT)+' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 '+
                              str(ROOT/'.venv/bin/python')+' -u '+str(Path(__file__))+' --stage '+stage)
        save(out/'evidence.json',evidence)
        (out/'ledger_before_fit.md').write_text(ledger,encoding='utf8')
        for path in source_paths:shutil.copy2(path,out/path.name)
        (out/'report.md').write_text('# HYP-081 neural calibration / '+stage+'\n\n'
            'SPEC012 extension. Started before any fit. New grouped neural OOF predictions augment '
            'the three existing tree families. H79 full-fit outer probabilities are reused unchanged; '
            'this runner never computes an outer score or trains the meta head. '
            'Inner-only epoch selection and frozen OOF-fit refits. Total ID budget900s, '
            '60s reserved for the parent-owned meta head.\n',encoding='utf8')
        oof=np.full((len(fit),len(classes)),np.nan,dtype=np.float32)
        coverage=np.zeros(len(fit),dtype=np.int64)
        audits=[]
        for fold_number,fold in enumerate(folds):
            guard();check_deadline(deadline)
            rel_fit=np.asarray(fold['fit'],dtype=np.int64)
            rel_valid=np.asarray(fold['valid'],dtype=np.int64)
            fold_fit,fold_valid=fit[rel_fit],fit[rel_valid]
            assert not set(groups[fold_fit])&set(groups[fold_valid])
            assert set(rel_fit)|set(rel_valid)==set(range(len(fit)))
            inner_a,inner_b=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(
                b.X.iloc[fold_fit],b.y.iloc[fold_fit],groups[fold_fit]))
            inner_fit,inner_valid=fold_fit[inner_a],fold_fit[inner_b]
            assert not set(groups[inner_fit])&set(groups[inner_valid])
            fold_dir=out/f'fold-{fold_number}'
            fold_dir.mkdir(exist_ok=False)
            np.savez_compressed(fold_dir/'splits.npz',fit=fold_fit,valid=fold_valid,
                                relative_fit=rel_fit,relative_valid=rel_valid,
                                inner_fit=inner_fit,inner_valid=inner_valid)
            save(fold_dir/'evidence.json',dict(status='started',fold=fold_number,
                 fold_split_key=split_key(fold_fit,fold_valid),
                 inner_split_key=split_key(inner_fit,inner_valid),
                 fit_rows=len(fold_fit),valid_rows=len(fold_valid),
                 fit_fingerprint=hashlib.sha256(np.asarray(fold_fit,dtype='<i8').tobytes()+
                     json.dumps(CONFIG,sort_keys=True).encode()+b'PLE1').hexdigest(),
                 total_elapsed=time.monotonic()-started))
            def make_trace(phase):
                def callback(row):
                    record=dict(fold=fold_number,phase=phase,total_elapsed=time.monotonic()-started,**row)
                    with (fold_dir/'epoch_trace.jsonl').open('a',encoding='utf8') as handle:
                        handle.write(json.dumps(record)+'\n')
                    if phase=='inner' and (row['epoch']==1 or row['epoch']%10==0):
                        print(json.dumps(dict(hypothesis=81,stage=stage,**record)),flush=True)
                return callback
            network,preprocessor,selection=train(b.X.iloc[inner_fit],encode(b.y.iloc[inner_fit]),classes,1,
                deadline,make_trace('inner'),valid_frame=b.X.iloc[inner_valid],valid_y=encode(b.y.iloc[inner_valid]))
            save(fold_dir/'selection.json',selection)
            save(fold_dir/'inner_preprocessing.json',preprocessor.state())
            del network,preprocessor
            network,preprocessor,refit=train(b.X.iloc[fold_fit],encode(b.y.iloc[fold_fit]),classes,1,
                deadline,make_trace('refit'),epochs=selection['best_epoch'])
            p=probabilities(network,tensors(preprocessor,b.X.iloc[fold_valid]),deadline)
            assert np.isfinite(p).all() and np.allclose(p.sum(1),1,rtol=1e-6,atol=1e-6)
            oof[rel_valid]=p;coverage[rel_valid]+=1
            np.savez_compressed(fold_dir/'oof_predictions.npz',positions=fold_valid,
                                relative_positions=rel_valid,probabilities=p,classes=classes,
                                y=b.y.iloc[fold_valid].to_numpy(dtype=str))
            state=preprocessor.state()
            state['seen_categories']={c:sorted(values) for c,values in preprocessor.seen_.items()}
            save(fold_dir/'preprocessing.json',state)
            torch.save(network.state_dict(),fold_dir/'model_state.pt')
            audit=dict(status='completed',fold=fold_number,selection=selection,refit=refit,
                       total_elapsed=time.monotonic()-started,outer_score_evaluations=0)
            save(fold_dir/'audit.json',audit);audits.append(audit)
            print('FOLD '+json.dumps(dict(stage=stage,**audit)),flush=True)
        assert np.all(coverage==1) and np.isfinite(oof).all()
        np.savez_compressed(out/'calibration.npz',oof_neural=oof,classes=classes,y=y,fit_positions=fit,
                            valid_positions=valid,outer_neural=outer_nn['probabilities'])
        np.save(out/'coverage.npy',coverage)
        result=dict(status='calibration_ready',stage=stage,rows=len(fit),classes=len(classes),
                    folds=3,training_model_fits=6,outer_model_fits=0,outer_score_evaluations=0,
                    audits=audits,seconds=time.monotonic()-stage_started,total_seconds=time.monotonic()-started,
                    remaining_total_budget_seconds=900-(time.monotonic()-started),
                    conclusion='Не оценивалась; только missing OOF calibration, head/outer score у parent')
    except Exception as exc:
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        result=dict(status='interrupted' if isinstance(exc,DeadlineExceeded) else 'failed',
                    error=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-stage_started,
                    total_seconds=time.monotonic()-started,outer_score_evaluations=0)
    guard();save(out/'metrics.json',result)
    with (out/'report.md').open('a',encoding='utf8') as handle:
        handle.write('\n## Calibration outcome\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
    print('RESULT '+json.dumps(dict(path=str(out),**result),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['primary','secondary'],required=True)
    main(parser.parse_args().stage)
