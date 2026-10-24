"""Clean final archive verification: exactly two public full refits, no evaluation.

Running this entry point performs real full-data refits. Preparation/tests never
invoke it. No source, baseline, archive or previous verification is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from experiments.package_sprint_delivery import ROOT, REQUIREMENTS_SHA256, read_json, require, safe_path, sha256


def clean_environment(environment=None):
    env = dict(os.environ if environment is None else environment)
    for key in list(env):
        if key.upper().startswith(('OPENAI', 'SPEC006', 'COMBO')) or key.upper() == 'PYTHONPATH':
            env.pop(key)
    env.update(AUTOML_MAX_EXPERIMENTS='0', AUTOML_PORTFOLIO='0', AUTOML_TIME_LIMIT='1200',
               PYTHONIOENCODING='utf8', PYTHONNOUSERSITE='1')
    return env


def verify_archive_members(archive, expected_sources):
    names = archive.namelist()
    require(len(names) == len(set(names)) and set(names) == set(expected_sources), 'ZIP member set mismatch')
    require(all(Path(name).name == name and '/' not in name and '\\' not in name
                and (name.endswith('.py') or name == 'requirements.txt') for name in names), 'Nonflat or nonruntime ZIP member')
    require(archive.testzip() is None, 'ZIP CRC failure')
    for name, digest in expected_sources.items():
        require(hashlib.sha256(archive.read(name)).hexdigest() == digest, 'ZIP source SHA mismatch: '+name)


def verify_summary(summary, expected, initial_refits, *, train_rows, test_rows):
    require(summary.get('status') == 'completed' and summary.get('cached_history') is True,
            'Public entry point did not select cached measured history')
    require(summary.get('metric') == expected['metric'] and summary.get('selected') == expected['candidate'],
            'Selected candidate or frozen state differs from baseline')
    require(isinstance(summary.get('selected_score'), (float, int))
            and abs(summary['selected_score']-expected['score']) <= 1e-12, 'Selected score differs from baseline')
    require(summary.get('submission_rows') == test_rows and summary.get('llm_calls', 0) == 0,
            'Unexpected submission size or LLM calls')
    require(len(initial_refits) == 1 and initial_refits[0]['candidate'] == expected['candidate'],
            'Expected exactly one full winner refit')
    result = initial_refits[0]['result']
    require(result.get('status') == 'completed' and result.get('mode') == 'refit'
            and result.get('train_rows') == train_rows and result.get('test_rows') == test_rows,
            'Winner full refit did not cover all supplied rows')


def verify(*, root=ROOT, manifest_path=None, output=None):
    from automl_data import load_task, read_submission, split_key, validate_submission, validation_split
    import numpy as np
    root = Path(root).resolve()
    manifest_path = safe_path(root, manifest_path or 'results/2026-10-24/package-manifest-sprint-v3.json')
    manifest = read_json(manifest_path)
    archive_path = safe_path(root, manifest['file'])
    require(sha256(archive_path) == manifest['sha256'], 'Archive SHA differs from manifest')
    require(sha256(root/'baseline.json') == manifest['baseline_sha256'], 'Baseline changed after packaging')
    for name, digest in manifest['sources'].items():
        require(sha256(root/name) == digest, 'Runtime source changed after packaging: '+name)
    for field in ('verification', 'adapter_verification'):
        evidence = manifest[field]
        require(sha256(safe_path(root, evidence['resolved_path'])) == evidence['resolved_sha256'],
                'Verified evidence changed after packaging: '+field)
    stamp = str(time.time_ns())
    output = safe_path(root, output or 'results/2026-10-24/sprint-delivery-'+stamp)
    output.mkdir(parents=True, exist_ok=False)
    extracted = safe_path(root, 'tmp/sprint-flat-'+stamp)
    extracted.mkdir(parents=True, exist_ok=False)
    records = []
    result = {'status': 'failed', 'archive': str(archive_path), 'archive_sha256': manifest['sha256'],
        'clean_extraction': str(extracted), 'records': records, 'repeated_validation_fits': 0,
        'hidden_score_verified': False, 'real_platform_api_tested': False}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            verify_archive_members(archive, manifest['sources'])
            require(hashlib.sha256(archive.read('requirements.txt')).hexdigest() == REQUIREMENTS_SHA256,
                    'Provided requirements SHA mismatch')
            archive.extractall(extracted)
        for kind in ('regression', 'multiclass'):
            expected = manifest['baseline'][kind]
            identity = manifest['known_task_identity'][kind]
            bundle = load_task(root/'data'/kind/'task.md')
            require(all(bundle.hashes.get(k) == v for k,v in identity['hashes'].items()), 'Public data SHA changed')
            fit, valid, _ = validation_split(bundle)
            require(split_key(fit, valid) == identity['split_key'], 'Cached history split identity changed')
            runs = bundle.task_path.parent/'agent_runs'
            before = set(runs.iterdir()) if runs.exists() else set()
            started = time.monotonic()
            process = subprocess.run([sys.executable, '-X', 'utf8', str(extracted/'solution.py'), str(bundle.task_path)],
                cwd=extracted, env=clean_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf8', timeout=1200)
            (output/(kind+'.log')).write_text(process.stdout, encoding='utf8')
            require(process.returncode == 0, kind+': public CLI failed; see saved log')
            created = set(runs.iterdir())-before
            require(len(created) == 1, 'Expected exactly one new public agent run')
            run = created.pop()
            summary = read_json(run/'summary.json')
            initial = read_json(run/'initial_refit.json')
            verify_summary(summary, expected, initial, train_rows=len(bundle.train), test_rows=len(bundle.test))
            requests = list(run.glob('*/request.json'))
            require(len(requests) == 1 and not list(run.glob('experiment-*')), 'Unexpected public validation/evaluation fit')
            request = read_json(requests[0])
            require(request['mode'] == 'refit' and request['candidate'] == expected['candidate'], 'Nonfinal or wrong worker fit')
            require(len(list(run.glob('refit-*'))) == 1, 'Unexpected extra refit')
            profile = read_json(run/'profile.json')
            require(profile['hashes'] == bundle.hashes and profile['split_key'] == identity['split_key'], 'Run data/source identity differs')
            matches = [r for r in profile['known']['records'] if r.get('candidate') == expected['candidate']
                       and r.get('score') == expected['score'] and r.get('source') == expected['source']]
            require(len(matches) == 1, 'Selected cached source provenance differs')
            refit = requests[0].parent
            audit = read_json(refit/'preprocessing-audit.json')
            require(audit['fit_original_rows'] == len(bundle.train)
                    and audit['fit_retained_rows'] == len(bundle.train), 'Final wrapper removed training rows')
            with np.load(refit/'training-filter.npz', allow_pickle=False) as filters:
                require(filters['keep_mask'].shape == (len(bundle.train),) and filters['keep_mask'].all()
                        and filters['removed_positions'].size == 0, 'Training row-filter mismatch')
            if 'decision_state' in expected['candidate']:
                require(audit.get('state_source') == 'frozen_candidate' and audit.get('decision_fit_count') == 0
                        and audit.get('decision_state') == expected['candidate']['decision_state']
                        and audit.get('meta_state') == expected['candidate']['meta_state'], 'Frozen classifier state changed')
            if 'pool_weights' in expected['candidate']:
                require(audit.get('pool_weights') == expected['candidate']['pool_weights']
                        and audit.get('weight_fit_count') == 0, 'Frozen pool weights changed')
            frame = read_submission(bundle, bundle.task_path.parent/'submission.csv')
            validate_submission(bundle, frame)
            candidate_frame = read_submission(bundle, refit/'candidate_submission.csv')
            require(frame.equals(candidate_frame), 'Final submission differs from verified worker output')
            require(len(frame) == {'regression': 22449, 'multiclass': 41166}[kind], 'Unexpected public task row count')
            record = {'task': kind, 'strategy': expected['candidate']['strategy'], 'selected_score': summary['selected_score'],
                'source': expected['source'], 'rows': len(frame), 'columns': list(frame),
                'all_ids_order_exact': True, 'full_refits': 1, 'validation_fits': 0,
                'seconds': time.monotonic()-started, 'run': str(run), 'summary': summary, 'audit': audit,
                'submission_sha256': sha256(bundle.task_path.parent/'submission.csv')}
            (output/(kind+'.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
            records.append(record)
            print(json.dumps({k:record[k] for k in ('task', 'strategy', 'selected_score', 'rows', 'seconds')}), flush=True)
        result.update(status='passed', total_full_refits=2)
    except Exception as exc:
        result.update(error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    (output/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    (output/'report.md').write_text('# Sprint delivery: '+result['status']+'\n\n'
        'Clean archive extraction; two public final full refits only, validation disabled. '
        'Candidate, cached score/source, frozen state, all rows/columns/IDs and source hashes checked. '
        'Details and any failure: [verification.json](verification.json). '
        'Hidden platform score and real platform API are unverified.\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'verification': str(output/'verification.json')}))
    return 0 if result['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    return verify(root=args.root, manifest_path=args.manifest, output=args.output)


if __name__ == '__main__': raise SystemExit(main())
