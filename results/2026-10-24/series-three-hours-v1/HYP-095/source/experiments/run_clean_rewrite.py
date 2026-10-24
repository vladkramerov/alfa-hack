"""SPEC010 fixed protocols, immutable sources and a 900-second hypothesis budget."""
from dataclasses import replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from research_data import load_task, split_data
from research_memory import atomic_json, timestamp

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'results/2026-10-24/series-clean-rewrite-v1'
JOBS = [
    {'hypothesis': 'HYP-052', 'task': 'multiclass', 'recipe': {'family': 'mixed_neighbors', 'parameters': {'n_neighbors': 25}}},
    {'hypothesis': 'HYP-062', 'task': 'regression', 'recipe': {'family': 'factorization_machine', 'parameters': {
        'rank': 8, 'epochs': 20, 'batch_size': 512, 'learning_rate': .005, 'l2': .001}}},
]


def split_digest(fit, valid):
    return hashlib.sha256(np.asarray(fit, dtype='<i8').tobytes() + b'|' + np.asarray(valid, dtype='<i8').tobytes()).hexdigest()


def protocol(task):
    if task.task_type == 'regression':
        years = task.X['Year_Factor'].to_numpy()
        fit, valid = np.flatnonzero(years < 4), np.flatnonzero(years == 4)
        if len(fit) == 0 or len(valid) == 0:
            raise ValueError('Frozen public temporal confirmation requires years <4 ->4')
    else:
        raw = task.raw_X if task.raw_X is not None else task.X
        groups = pd.util.hash_pandas_object(raw, index=False).to_numpy()
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(task.X, task.y, groups))
        fit, valid = folds[1]  # Fixed before scores, never select the best-looking fold.
        if set(task.y.iloc[fit]) != set(task.y) or set(task.y.iloc[valid]) != set(task.y):
            raise ValueError('Confirmation lacks a class')
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError('Raw groups cross confirmation')
    inner_task = replace(task, X=task.X.iloc[fit].copy(), y=task.y.iloc[fit].copy(),
                         raw_X=task.raw_X.iloc[fit].copy() if task.raw_X is not None else None,
                         X_test=task.X.iloc[valid].copy())
    inner_fit, inner_valid, description = split_data(inner_task)
    result = {'confirmation_fit': np.asarray(fit), 'confirmation_valid': np.asarray(valid),
              'inner_fit': np.asarray(fit)[inner_fit], 'inner_valid': np.asarray(fit)[inner_valid]}
    if set(result['inner_fit']) & set(valid) or set(result['inner_valid']) & set(valid):
        raise ValueError('Inner sees confirmation')
    return {**result, 'description': description}


def freeze():
    sources = sorted([*ROOT.glob('research_*.py'), ROOT / 'rewrite_solution.py',
                      *ROOT.glob('automl_*.py'), ROOT / 'experiments/clean_control_worker.py'])
    digests = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    plan_path = SUITE / 'frozen-plan.json'
    if plan_path.exists():
        prior = json.loads(plan_path.read_text(encoding='utf8'))
        if prior['sources'] != digests or prior['jobs'] != JOBS:
            raise RuntimeError('Sources/configuration changed after freeze; do not overwrite evidence')
        return prior
    SUITE.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for job in JOBS:
        task = load_task(ROOT / 'data' / job['task'] / 'task.md')
        positions = protocol(task)
        folder = SUITE / job['task']
        folder.mkdir(exist_ok=True)
        for stage in ['inner', 'confirmation']:
            np.savez_compressed(folder / f'{stage}-split.npz', fit=positions[stage + '_fit'], valid=positions[stage + '_valid'])
        with np.load(ROOT / f'results/2026-10-23/series-{job["task"]}-preprocessing-v1/splits.npz', allow_pickle=False) as old:
            if np.array_equal(positions['confirmation_valid'], old['outer_valid']):
                raise RuntimeError('New confirmation accidentally equals old outer')
        datasets[job['task']] = {'hashes': task.hashes, 'description': positions['description'],
                                'stages': {stage: {'fit_rows': len(positions[stage + '_fit']), 'valid_rows': len(positions[stage + '_valid']),
                                                  'split_sha256': split_digest(positions[stage + '_fit'], positions[stage + '_valid'])}
                                           for stage in ['inner', 'confirmation']}}
    for source in sources:
        target = SUITE / 'source' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    plan = {'created': timestamp(), 'sources': digests, 'jobs': JOBS, 'datasets': datasets,
            'budget_seconds_per_hypothesis': 900, 'control_exception': 'user confirmed Q3 + final plan',
            'old_outer_validation_refits': 0, 'additional_confirmation_is_not_private_proof': True}
    atomic_json(plan_path, plan)
    return plan


def append_outcome(job, stage, directory, result):
    relative = directory.relative_to(ROOT).as_posix()
    detail = f'{result["status"]}, {result.get("metric")}={result.get("score")}'
    note = ('Это разрешённая сравнительная проверка старого лидера на новом confirmation, без frozen public meta-state.'
            if stage == 'control' else 'Новый механизм; параметры не выбирались по confirmation.')
    with (directory / 'report.md').open('a', encoding='utf8') as stream:
        stream.write(f'\nИтог: **{detail}**. {note}\n\n[Метрики](metrics.json), [данные/исходники/split](evidence.json), [лог](worker.log).\n')
    with (ROOT / 'HYPOTHESES.md').open('a', encoding='utf8') as stream:
        stream.write(f'\n- SPEC010 {timestamp()}: **{job["hypothesis"]}/{job["task"]}/{stage}**: {detail}; [отчёт]({relative}/report.md). {note} Повтор этой проверки запрещён.\n')
    with (ROOT / 'results/README.md').open('a', encoding='utf8') as stream:
        stream.write(f'\n| {job["hypothesis"]} | {job["task"]}/{stage} | {detail} | [report](../{relative}/report.md) |\n')
    card = ROOT / f'docs/hypotheses/{job["hypothesis"]}.md'
    with card.open('a', encoding='utf8') as stream:
        stream.write(f'\n- {stage}: {detail}; [отчёт](../../{relative}/report.md).\n')


def run_stage(job, stage, plan, deadline):
    from research_agent import launch_worker
    registry = SUITE / 'attempts.json'
    attempts = json.loads(registry.read_text(encoding='utf8')) if registry.exists() else []
    if any(a['hypothesis'] == job['hypothesis'] and a['stage'] == stage for a in attempts):
        raise RuntimeError('Scientific stage already attempted; no replay')
    ledger = (ROOT / 'HYPOTHESES.md').read_text(encoding='utf8')
    if job['hypothesis'] not in ledger:
        raise RuntimeError('Hypothesis not registered')
    parent = ROOT / 'results/2026-10-24'
    count = 1
    while list(parent.glob(f'{job["hypothesis"]}-{count:02d}-*')):
        count += 1
    directory = parent / f'{job["hypothesis"]}-{count:02d}-{job["task"]}-{stage}-rewrite'
    directory.mkdir()
    split_stage = 'confirmation' if stage == 'control' else stage
    split_path = SUITE / job['task'] / f'{split_stage}-split.npz'
    (directory / 'splits.npz').write_bytes(split_path.read_bytes())
    recipe = ({'family': 'old_leader', 'strategy': 'temporal_profile_memory' if job['task'] == 'regression' else 'probability_stacking'}
              if stage == 'control' else job['recipe'])
    event = {'hypothesis': job['hypothesis'], 'task': job['task'], 'stage': stage, 'recipe': recipe,
             'status': 'started', 'started': timestamp(), 'directory': str(directory.relative_to(ROOT)),
             'ledger_sha256_before_fit': hashlib.sha256(ledger.encode()).hexdigest(),
             'sources': plan['sources'], 'data': plan['datasets'][job['task']],
             'source_snapshot': str(SUITE / 'source'), 'split_sha256': plan['datasets'][job['task']]['stages'][split_stage]['split_sha256']}
    attempts.append(event)
    atomic_json(registry, attempts)
    atomic_json(directory / 'evidence.json', event)
    (directory / 'report.md').write_text(f'# {job["hypothesis"]}/{job["task"]}/{stage}\n\nStarted before fit: {event["started"]}. [SPEC010](../../../docs/specs/SPEC-010-clean-rewrite.md).\n\nКоманда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.run_clean_rewrite`.\n\nМаксимум900сек на все стадии гипотезы. Старый outer не переобучается. Source snapshot неизменен.\n', encoding='utf8')
    request = {'task_path': str(ROOT / 'data' / job['task'] / 'task.md'), 'recipe': recipe,
               'mode': stage, 'output': str(directory), 'split_path': str(directory / 'splits.npz'),
               'history': str(SUITE / 'hypotheses.json'), 'global_ledger': str(ROOT / 'HYPOTHESES.md'),
               'worker_source': str(SUITE / 'source' / ('experiments/clean_control_worker.py' if stage == 'control' else 'research_agent.py')),
               'control_fit_log': str(SUITE / 'control-fits')}
    # Production modules are immutable copied sources; verify bytes before every child.
    for name, expected in plan['sources'].items():
        if hashlib.sha256((SUITE / 'source' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source bytes changed')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        result = {'status': 'timeout', 'error': 'hypothesis total budget exhausted before stage', 'seconds': 0.}
        atomic_json(directory / 'metrics.json', result)
    else:
        result = launch_worker(request, remaining)
    event.update(result, finished=timestamp())
    attempts[-1] = event
    atomic_json(registry, attempts)
    atomic_json(directory / 'evidence.json', event)
    append_outcome(job, stage, directory, result)
    print(json.dumps({k: event.get(k) for k in ['hypothesis', 'task', 'stage', 'status', 'score', 'seconds', 'error']}, ensure_ascii=False), flush=True)
    return event


def main(only=None, prepare_only=False):
    plan = freeze()
    if prepare_only:
        print(json.dumps(plan['datasets'], ensure_ascii=False), flush=True)
        return
    for job in JOBS:
        if only and job['hypothesis'] not in only:
            continue
        registry = SUITE / 'attempts.json'
        previous = json.loads(registry.read_text(encoding='utf8')) if registry.exists() else []
        if any(a['hypothesis'] == job['hypothesis'] for a in previous):
            print('SKIP already attempted ' + job['hypothesis'], flush=True)
            continue
        deadline = time.monotonic() + 900
        first = run_stage(job, 'inner', plan, deadline)
        if first['status'] != 'completed':
            continue
        final = run_stage(job, 'confirmation', plan, deadline)
        if final['status'] == 'completed':
            run_stage(job, 'control', plan, deadline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--prepare-only', action='store_true')
    arguments = parser.parse_args()
    main(arguments.only, arguments.prepare_only)
