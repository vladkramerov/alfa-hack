"""Four new additive numerical views; reuse recorded controls, freeze before audit."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from automl_engine import _search_windows
from automl_models import canonical_candidate, build_pipeline

OUT = ROOT / 'results' / 'numeric-views-v1.json'
MODES = ['rank', 'signed_log', 'projection', 'rank_projection']


def run():
    if OUT.exists():
        raise RuntimeError('Study already exists; reuse it rather than repeat fits')
    report = {'protocol': 'Fixed four recipes, existing public-v5 controls; select by inner only, audit one frozen winner per task.',
              'limitation': 'Historically exposed public folds, no hidden-platform estimate.',
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ['automl_models.py', 'experiments/numeric_views_study.py']},
              'training_seconds': 0., 'tasks': {}}
    def save():
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    save()
    for task, dirname in [('regression', 'Регрессия'), ('multiclass', 'Многоклассовая классификация')]:
        bundle = load_task(ROOT / 'drive-download-20261023T201455Z-1-001' / dirname / 'task.md')
        old = ROOT / 'results/public-v5' / (task + '-adaptive')
        records = json.loads((old / 'experiments.json').read_text())
        control = (next(r for r in records if r.get('status') == 'completed' and r['candidate']['family'] == 'extra_trees')
                   if task == 'regression' else max((r for r in records if r.get('status') == 'completed'), key=lambda r:r['score']))
        metadata = json.loads((old / 'validation.json').read_text())
        assert metadata['data_hashes'] == bundle.hashes, 'Data changed'
        outer_fit, outer_valid, desc = validation_split(bundle)
        assert split_key(outer_fit, outer_valid) == metadata['audit_key']
        windows, _ = _search_windows(bundle, outer_fit, 900)
        assert [split_key(w['fit'], w['valid']) for w in windows] == [w['key'] for w in metadata['search_folds']]
        task_result = {'metric': bundle.metric, 'data_hashes': bundle.hashes, 'validation': metadata,
                       'control_candidate': control['candidate'], 'control_inner': control['score'],
                       'control_outer': 42.19670002719412 if task == 'regression' else .6373379463394646,
                       'control_source': str(old.relative_to(ROOT)), 'experiments': []}
        report['tasks'][task] = task_result
        for mode in MODES:
            candidate = canonical_candidate({**control['candidate'], 'numeric_view': mode})
            result = {'mode': mode, 'candidate': candidate, 'folds': []}
            task_result['experiments'].append(result)
            for window in windows:
                if report['training_seconds'] > 1080:
                    result['status'] = 'budget_skipped'; save(); return report
                fit, valid = window['fit'], window['valid']
                model = build_pipeline(bundle.task_type, candidate, 4)
                before = time.monotonic()
                model.fit(bundle.X.iloc[fit], bundle.y.iloc[fit])
                seconds = time.monotonic() - before
                report['training_seconds'] += seconds
                pred = model.predict(bundle.X.iloc[valid])
                value = score(bundle.metric, bundle.y.iloc[valid].to_numpy(), pred)
                result['folds'].append({'key': split_key(fit, valid), 'score': value,
                                        'weight': window['weight'], 'fit_seconds': seconds})
                save()
            result['score'] = sum(f['weight'] * f['score'] for f in result['folds']) / sum(f['weight'] for f in result['folds'])
            result['status'] = 'completed'
            print(json.dumps({'task':task, 'mode':mode, 'inner':result['score'], 'fit_total':report['training_seconds']}), flush=True)
            save()
        ordering = sorted(task_result['experiments'], key=lambda r:r['score'], reverse=task != 'regression')
        winner = ordering[0]
        improves = winner['score'] < control['score'] if task == 'regression' else winner['score'] > control['score']
        task_result['frozen_selection'] = winner['mode'] if improves else 'control'
        save()  # Persist selection before reading outer labels.
        if improves:
            before = time.monotonic()
            model = build_pipeline(bundle.task_type, winner['candidate'], 4).fit(bundle.X.iloc[outer_fit], bundle.y.iloc[outer_fit])
            seconds = time.monotonic() - before
            report['training_seconds'] += seconds
            pred = model.predict(bundle.X.iloc[outer_valid])
            task_result['selected_outer'] = score(bundle.metric, bundle.y.iloc[outer_valid].to_numpy(), pred)
            task_result['audit_fit_seconds'] = seconds
            if task == 'multiclass':
                from sklearn.metrics import classification_report
                task_result['per_class_outer'] = classification_report(bundle.y.iloc[outer_valid], pred, output_dict=True, zero_division=0)
        else:
            task_result['selected_outer'] = task_result['control_outer']
        print(json.dumps({'task': task, 'selected': task_result['frozen_selection'], 'outer': task_result['selected_outer']}), flush=True)
        save()
    report['status'] = 'completed'
    save()
    return report


if __name__ == '__main__':
    run()
