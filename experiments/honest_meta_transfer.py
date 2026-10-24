"""Transfer H48's grouped-OOF meta learner to our delivered ensemble teacher.

Unlike rejected ST6, meta training never receives in-sample teacher probabilities.
An inner gate precedes any outer fit; no change to the delivered agent here.
"""
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from automl_engine import _search_windows
from automl_models import build_pipeline
from automl_worker import class_decisions, train_priors

OUT = ROOT / 'results/team-runtime/honest-meta.json'
WORK = ROOT / 'local_data/team-runtime'


def run_stage(bundle, fit, valid, candidate, stage, report):
    X, y = bundle.X.iloc[fit], bundle.y.iloc[fit]
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    oof = np.zeros((len(X), len(bundle.classes)))
    started = time.monotonic()
    folds = []
    for number, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=91).split(X, y, groups)):
        assert not set(groups[a]) & set(groups[b])
        cache = WORK / (stage + '-fold-' + str(number) + '.npz')
        if cache.exists():
            with np.load(cache) as z:
                assert np.array_equal(z['fit'], fit[a]) and np.array_equal(z['valid'], fit[b])
                p = z['probabilities']
        else:
            model = build_pipeline('classification', candidate, 8).fit(X.iloc[a], y.iloc[a])
            p = model.predict_proba(X.iloc[b])
            assert np.array_equal(model.classes_, bundle.classes)
            np.savez_compressed(cache, fit=fit[a], valid=fit[b], probabilities=p)
        oof[b] = p
        folds.append({'key': split_key(fit[a], fit[b]), 'fit_rows': len(a), 'valid_rows': len(b)})
    labels, counts = np.unique(y, return_counts=True)
    meta = LogisticRegression(C=.1, max_iter=1000, class_weight=dict(zip(labels, 1/np.sqrt(counts))))
    meta.fit(np.log(np.clip(oof, 1e-6, 1)), y)
    cache = WORK / (stage + '-teacher.npz')
    if cache.exists():
        with np.load(cache) as z:
            assert np.array_equal(z['fit'], fit) and np.array_equal(z['valid'], valid)
            p = z['probabilities']
    else:
        teacher = build_pipeline('classification', candidate, 8).fit(X, y)
        p = teacher.predict_proba(bundle.X.iloc[valid])
        assert np.array_equal(meta.classes_, teacher.classes_)
        np.savez_compressed(cache, fit=fit, valid=valid, probabilities=p)
    assert np.array_equal(meta.classes_, bundle.classes)
    calibrated = meta.predict_proba(np.log(np.clip(p, 1e-6, 1)))
    np.savez_compressed(WORK / (stage + '.npz'), positions=valid, probabilities=p,
                        calibrated=calibrated, classes=bundle.classes)
    base_labels = class_decisions(p, bundle.classes, train_priors(y, bundle.classes), candidate.get('decision_power', 0))
    pred = np.asarray(bundle.classes)[calibrated.argmax(axis=1)]
    record = {'stage': stage, 'key': split_key(fit, valid), 'folds': folds,
              'base': score('macro_f1', bundle.y.iloc[valid], base_labels),
              'score': score('macro_f1', bundle.y.iloc[valid], pred),
              'seconds': time.monotonic()-started,
              'rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              'teacher_model_fits': 8, 'meta_fits': 1}
    report['stages'].append(record)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(record), flush=True)
    return record


def main():
    if OUT.exists() and json.loads(OUT.read_text())['status'] == 'completed':
        raise RuntimeError('Reuse saved receipt; do not repeat completed fits')
    WORK.mkdir(parents=True, exist_ok=True)
    b = load_task(ROOT / 'drive-download-20261023T201455Z-1-001/Многоклассовая классификация/task.md')
    outer_fit, outer_valid, _ = validation_split(b)
    window = _search_windows(b, outer_fit, 1800)[0][0]
    old = json.loads((ROOT / 'results/strongest-team/e2e-v2.json').read_text())
    candidate = next(r['selected'] for r in old['runs'] if r['task']=='classification' and r['scope']=='shadow')
    report = {'status': 'running', 'candidate': candidate, 'hashes': b.hashes, 'stages': [],
              'protocol': 'HT1: H48 grouped3OOF log-probability inverse-sqrt LR(C=.1), transferred to own delivered XGB-frequency/ET teacher. ST6 in-sample defect removed. Inner improvement>=.002 required before one outer. Outer historically viewed; no independent claim. No public coefficients shipped.'}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    inner = run_stage(b, window['fit'], window['valid'], candidate, 'inner', report)
    if inner['score'] >= inner['base'] + .002:
        run_stage(b, outer_fit, outer_valid, candidate, 'outer', report)
    else:
        report['decision'] = 'reject_inner_no_outer_fit'
    report['status'] = 'completed'
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
