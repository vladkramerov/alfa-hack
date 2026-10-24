"""Bounded, append-only SPEC011 trial launcher with per-hypothesis frozen code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

from research_memory import atomic_json, timestamp

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'results/2026-10-24/series-three-hours-v1'
PARAMETERS = {50: {'default': {}},70: {'low': {'alpha': .5}, 'mid': {'alpha': .8}, 'high': {'alpha': .95}},
              71: {'default': {}}, 72: {'default': {}}, 73: {'default': {}}, 74: {'default': {}}, 75: {'default': {}},
              76: {'default': {}}, 77: {'default': {}}, 78: {'default': {}}, 80: {'default': {}}, 81: {'default': {}},82:{'default':{}},83:{'default':{}},
              84:{'default':{}},85:{'default':{}},86:{'default':{}},87:{'default':{}},88:{'default':{}},89:{'default':{}},90:{'default':{}},
              92:{'default':{},'extended':{'iterations':900}},93:{'default':{}},94:{'default':{},'long':{'epochs':150}},96:{'default':{}},97:{'default':{}}}


def main(number, stage='inner', variant='default'):
    config = PARAMETERS[number][variant]
    ledger = (ROOT / 'HYPOTHESES.md').read_text(encoding='utf8')
    if f'HYP-{number:03d}' not in ledger:
        raise RuntimeError('Unregistered hypothesis')
    directory = SUITE / f'HYP-{number:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    registry = directory / 'attempts.json'
    attempts = json.loads(registry.read_text(encoding='utf8')) if registry.exists() else []
    if any(row['stage'] == stage and row['variant'] == variant for row in attempts):
        raise RuntimeError('Trial already attempted; do not replay')
    if stage == 'outer' and any(row['stage'] == 'outer' for row in attempts):
        raise RuntimeError('Only one final configuration per hypothesis')
    models = ['sprint_regression.py'] if number in [70, 71, 76, 81, 83,85,86,89,90] else ['sprint_classification.py']
    if number == 76:
        models += ['sprint_panel.py']
    if number == 81:
        models += ['sprint_linkage.py']
    if number == 83:
        models += ['sprint_ratio_transport.py']
    if number == 85:
        models += ['sprint_uncertainty.py']
    if number == 86:
        models += ['sprint_state_space.py']
    if number == 89:
        models += ['sprint_differences.py']
    if number == 90:
        models += ['sprint_missing_expert.py']
    if number == 74:
        models += ['sprint_contextual.py']
    if number == 75:
        models += ['sprint_decision.py']
    if number == 77:
        models += ['sprint_ambitious.py']
    if number == 78:
        models += ['sprint_decision.py', 'sprint_posterior.py']
    if number == 80:
        models += ['sprint_ambitious.py', 'sprint_softboost.py']
    if number == 50:
        models += ['sprint_ambitious.py', 'sprint_focal.py']
    if number == 82:
        models += ['sprint_contextual.py', 'sprint_network.py']
    if number == 84:
        models += ['sprint_signatures.py']
    if number == 87:
        models += ['sprint_cohort.py','sprint_decision.py']
    if number == 88:
        models += ['sprint_honest.py','sprint_contextual.py']
    if number in [92,93,94,96,97]:
        models += ['sprint_honest.py','sprint_contextual.py','sprint_ambitious.py','sprint_softboost.py','sprint_honest_objectives.py']
    if number in [96,97]:
        models += ['sprint_posterior.py','sprint_decision.py']
    sources = [*ROOT.glob('automl_*.py'), ROOT / 'research_memory.py', ROOT / 'research_data.py',
               *[ROOT / name for name in models], *ROOT.joinpath('experiments').glob('*.py')]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    frozen = directory / 'frozen.json'
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding='utf8'))
        # Only copied frozen modules execute after first trial. Main-code additions
        # to other hypotheses do not alter this hypothesis's scientific sources.
        source_hashes = old['sources']
    else:
        for source in sources:
            destination = directory / 'source' / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        atomic_json(frozen, {'created': timestamp(), 'sources': source_hashes, 'variants': PARAMETERS[number], 'spec': 'SPEC011'})
    for name, expected in source_hashes.items():
        if hashlib.sha256((directory / 'source' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source mutation')
    parent = ROOT / 'results/2026-10-24'
    sequence = 1
    while list(parent.glob(f'HYP-{number:03d}-{sequence:02d}-*')):
        sequence += 1
    out = parent / f'HYP-{number:03d}-{sequence:02d}-{stage}-{variant}-sprint'
    out.mkdir()
    event = {'hypothesis': f'HYP-{number:03d}', 'stage': stage, 'variant': variant, 'parameters': config,
             'started': timestamp(), 'status': 'started', 'directory': str(out.relative_to(ROOT)),
             'ledger_sha256': hashlib.sha256(ledger.encode()).hexdigest(), 'sources': source_hashes}
    attempts.append(event)
    atomic_json(registry, attempts)
    atomic_json(out / 'evidence.json', event)
    (out / 'report.md').write_text(f'# HYP-{number:03d}/{stage}/{variant}\n\nStarted before fit {event["started"]}. [SPEC011](../../../docs/specs/SPEC-011-three-hour-improvement.md).\n\nNo old estimator refits. Fixed source/parameters; exploratory research, not private score.\n', encoding='utf8')
    request = {'number': number, 'stage': stage, 'parameters': config, 'repo_root': str(ROOT), 'output': str(out), 'history': str(directory / 'model-fit-history.json')}
    atomic_json(out / 'request.json', request)
    elapsed = sum(row.get('seconds', 0) for row in attempts[:-1])
    deadline = datetime.fromisoformat('2026-10-24T04:15:07+03:00').timestamp()
    remaining = min(900 - elapsed, deadline - time.time() - 1800)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONIOENCODING='utf8')
    for key in list(env):
        if 'API_KEY' in key:
            env.pop(key, None)
    started = time.monotonic()
    try:
        if remaining <= 0:
            raise TimeoutError('Research reserve reached')
        with (out / 'worker.log').open('w', encoding='utf8') as log:
            child = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(directory / 'source/experiments/sprint_worker.py'), str(out / 'request.json')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
        result = json.loads((out / 'metrics.json').read_text(encoding='utf8'))
        if child.returncode and result['status'] == 'completed':
            raise RuntimeError('Exit/status mismatch')
    except Exception as exc:
        result = {'status': 'timeout' if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else 'failed',
                  'error': type(exc).__name__ + ': ' + str(exc), 'seconds': time.monotonic() - started}
        atomic_json(out / 'metrics.json', result)
        from research_memory import Memory
        memory = Memory(request['history'], ROOT / 'HYPOTHESES.md')
        for claim in memory.records():
            if claim['status'] == 'started':
                memory.finish(claim['fingerprint'], result)
        for nested in ['teacher-fits.json','expert-fits.json','head-fits.json']:
            journal=Memory(out/nested,ROOT/'HYPOTHESES.md')
            for claim in journal.records():
                if claim['status']=='started':journal.finish(claim['fingerprint'],result)
    event.update(result, finished=timestamp())
    attempts[-1] = event
    atomic_json(registry, attempts)
    atomic_json(out / 'evidence.json', event)
    detail = f'{result["status"]}; {result.get("metric")}={result.get("score")}; reference={result.get("reference_score")}'
    with (out / 'report.md').open('a', encoding='utf8') as stream:
        stream.write('\nИтог: ' + detail + '. [Метрики](metrics.json), [аудит](audit.json), [fit claim](../series-three-hours-v1/HYP-' + f'{number:03d}' + '/model-fit-history.json). Повтор не запускать.\n')
    with (ROOT / 'HYPOTHESES.md').open('a', encoding='utf8') as stream:
        stream.write(f'\n- SPEC011 {timestamp()}: **HYP-{number:03d}/{stage}/{variant}** {detail}; [отчёт]({out.relative_to(ROOT).as_posix()}/report.md). Повтор этой конфигурации запрещён.\n')
    with (ROOT / 'results/README.md').open('a', encoding='utf8') as stream:
        stream.write(f'\n| HYP-{number:03d} | {stage}/{variant} | {detail} | [report]({out.relative_to(ROOT / "results").as_posix()}/report.md) |\n')
    print(json.dumps({k: event.get(k) for k in ['hypothesis', 'stage', 'variant', 'status', 'score', 'reference_score', 'relative_gain', 'seconds', 'error', 'directory']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, required=True)
    parser.add_argument('--stage', choices=['inner', 'outer'], default='inner')
    parser.add_argument('--variant', default='default')
    args = parser.parse_args()
    main(args.number, args.stage, args.variant)
