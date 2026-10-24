"""Build a new flat delivery only after final science and adapter verification.

baseline.json must contain verification and adapter_verification pointers:
{"path": "results/.../verification.json", "sha256": "..."}. A scientific
latest_verification pointer is dereferenced and its payload hash checked too.
This script never fits a model. It refuses to overwrite archives or manifests.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_SHA256 = '9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6'
BASE_FILES = ['solution.py', 'automl_data.py', 'automl_models.py', 'automl_preprocessing.py',
    'automl_history.py', 'automl_known.py', 'automl_llm.py', 'automl_worker.py',
    'automl_advanced.py', 'requirements.txt', 'automl_combo_memory.py',
    'automl_feature_sets.py', 'automl_reg_combo.py', 'automl_class_combo.py',
    'automl_combo_policy.py', 'automl_temporal_memory.py', 'automl_sprint.py',
    'automl_sprint_pool.py']


def require(condition, message):
    if not condition: raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def safe_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    require(path.is_relative_to(root), 'Path escapes the repository: '+str(value))
    return path


def read_verified(root, pointer, label):
    require(isinstance(pointer, dict) and {'path', 'sha256'} <= pointer.keys(),
            label+': mandatory hash-bound verification pointer absent')
    path = safe_path(root, pointer['path'])
    require(sha256(path) == pointer['sha256'], label+': verification SHA mismatch')
    record = read_json(path)
    if 'latest_verification' in record:
        require(record.get('passed') is True, label+': latest verification failed')
        path = safe_path(root, record['latest_verification'])
        require(sha256(path) == record['sha256'], label+': latest payload SHA mismatch')
        record = read_json(path)
    require(record.get('status') == 'passed' or record.get('passed') is True,
            label+': verification did not pass')
    require(not record.get('pending_cases') and not record.get('registry_errors'),
            label+': unresolved verification cases')
    return {'pointer': pointer, 'resolved_path': path.relative_to(Path(root).resolve()).as_posix(),
            'resolved_sha256': sha256(path), 'record': record}


def verify_declared_sources(root, record):
    hashes = record.get('source_sha256', {})
    require(isinstance(hashes, dict), 'Malformed verification source hashes')
    for name, digest in hashes.items():
        require(sha256(safe_path(root, name)) == digest, 'Source changed after adapter verification: '+name)
    return hashes


def runtime_files(root, seeds=None):
    """Transitive local imports, including lazy imports, all preserved at root."""
    root = Path(root)
    names, pending = [], list(BASE_FILES if seeds is None else seeds)
    while pending:
        name = pending.pop(0)
        if name in names: continue
        require(Path(name).name == name and (root/name).is_file(), 'Missing flat runtime file: '+name)
        names.append(name)
        if not name.endswith('.py'): continue
        tree = ast.parse((root/name).read_text(encoding='utf8'), filename=name)
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                top = module.split('.')[0]
                if top == 'experiments': raise ValueError('Research package dependency in portable runtime: '+name)
                dependency = top+'.py'
                if (root/dependency).is_file() or top.startswith(('automl_', 'sprint_')):
                    require('.' not in module, 'Nested local Python dependency cannot be flat: '+module)
                    pending.append(dependency)
    return names


def baseline_pointer(baseline, field):
    if field in baseline: return baseline[field]
    pointers = [baseline[k].get(field) for k in ('regression', 'multiclass')]
    require(pointers[0] is not None and pointers[0] == pointers[1], 'Missing shared baseline '+field)
    return pointers[0]


def verify_winners(root, baseline, science, adapter):
    known = runpy.run_path(str(Path(root)/'automl_known.py'))['KNOWN_TASKS']
    identities = {}
    for kind, metric in [('regression', 'rmse'), ('multiclass', 'macro_f1')]:
        winner = baseline[kind]
        require(winner['metric'] == metric and math.isfinite(winner['score']), 'Invalid baseline metric/score')
        candidate, source = winner['candidate'], safe_path(root, winner['source'])
        require(isinstance(candidate, dict) and candidate.get('strategy') and source.is_file(), 'Missing winner candidate/source')
        matches = [(task, record) for task in known if task['metric'] == metric for record in task['records']
                   if record.get('candidate') == candidate and record.get('score') == winner['score']
                   and safe_path(root, record.get('source', '')) == source]
        require(len(matches) == 1, 'Embedded measured winner does not exactly match baseline: '+kind)
        run = [r for r in science.get('runs', []) if safe_path(root, r['directory']) == source.parent
               and r.get('stage') == 'outer' and r.get('status') == 'completed' and r.get('verified')
               and r.get('metric') == metric and abs(r.get('score', math.inf)-winner['score']) <= 1e-12]
        require(len(run) == 1, 'Winner outer result lacks final independent verification: '+kind)
        hyp = re.search(r'HYP[-_ ]?0*(\d+)', winner.get('hypothesis', '') or candidate.get('hypothesis', ''))
        require(hyp is not None and 'H'+str(int(hyp.group(1))) in adapter.get('checks', {}),
                'Adapter replay does not cover selected hypothesis: '+kind)
        task, record = matches[0]
        identities[kind] = {'hashes': task['hashes'], 'split_key': task['split_key'],
            'metric': metric, 'source': winner['source'], 'source_sha256': sha256(source)}
    return identities


def package(*, root=ROOT, destination=None, manifest_path=None,
            provided_requirements=Path('C:/Users/User/Downloads/requirements (1).txt')):
    root = Path(root).resolve()
    destination = safe_path(root, destination or 'dist/solution_verified_v3.zip')
    manifest_path = safe_path(root, manifest_path or 'results/2026-10-24/package-manifest-sprint-v3.json')
    if destination.exists() or manifest_path.exists():
        raise FileExistsError('Archive or manifest already exists; previous deliveries are immutable')
    baseline = read_json(root/'baseline.json')
    final = read_verified(root, baseline_pointer(baseline, 'verification'), 'final science')
    adapter = read_verified(root, baseline_pointer(baseline, 'adapter_verification'), 'adapter')
    require(adapter['record'].get('model_fits') == 0, 'Adapter verification must have zero fits')
    require(verify_declared_sources(root, adapter['record']), 'Adapter verification has no source hashes')
    identities = verify_winners(root, baseline, final['record'], adapter['record'])
    provided = Path(provided_requirements).read_bytes()
    require(hashlib.sha256(provided).hexdigest() == REQUIREMENTS_SHA256
            and (root/'requirements.txt').read_bytes() == provided, 'Provided requirements changed')
    names = runtime_files(root)
    sources = {name: sha256(root/name) for name in names}
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation means even a concurrent package run cannot overwrite.
    with destination.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names: archive.write(root/name, name)
    with zipfile.ZipFile(destination) as archive:
        require(archive.testzip() is None and archive.namelist() == names, 'Invalid ZIP members')
        for name in names:
            require(hashlib.sha256(archive.read(name)).hexdigest() == sources[name], 'Source changed during packaging: '+name)
    require(0 < destination.stat().st_size < 100_000_000, 'Archive exceeds platform limit')
    manifest = {'file': str(destination), 'bytes': destination.stat().st_size,
        'sha256': sha256(destination), 'root_files': names, 'sources': sources,
        'baseline': baseline, 'baseline_sha256': sha256(root/'baseline.json'),
        'known_task_identity': identities,
        'verification': {k:v for k,v in final.items() if k != 'record'},
        'adapter_verification': {k:v for k,v in adapter.items() if k != 'record'},
        'provided_requirements_exact': True, 'requirements_sha256': REQUIREMENTS_SHA256,
        'data_or_models_included': False, 'hidden_platform_score_verified': False,
        'clean_public_full_refit_verified': False}
    with manifest_path.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'file': str(destination), 'sha256': manifest['sha256'], 'manifest': str(manifest_path)}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--provided-requirements', type=Path, default=Path('C:/Users/User/Downloads/requirements (1).txt'))
    args = parser.parse_args()
    package(root=args.root, destination=args.destination, manifest_path=args.manifest,
            provided_requirements=args.provided_requirements)


if __name__ == '__main__': main()
