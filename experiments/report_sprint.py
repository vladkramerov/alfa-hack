"""Build the immutable SPEC011 report from final, hash-checked evidence only.

No model, optimizer, metric computation, or scientific result writer is imported.
Run after promotion, packaging, clean delivery checks, and final unit tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUITE = Path('results/2026-10-24/series-three-hours-v1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_under(root, value):
    root = Path(root).resolve()
    path = (root / str(value).replace('\\', '/')).resolve()
    require(path.is_relative_to(root), 'Path escapes repository: ' + str(value))
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def passed(record, label):
    require(record.get('passed') is True or record.get('status') == 'passed', label + ' did not pass')
    require(not record.get('error') and not record.get('registry_errors') and not record.get('pending_cases'),
            label + ' has unresolved errors')


def checked_json(root, value, hashes, label, expected=None):
    path = path_under(root, value)
    if path.is_dir():
        path /= 'verification.json'
    digest = sha256(path)
    if expected is not None:
        require(digest == expected, label + ' SHA256 mismatch')
    hashes[path.relative_to(root).as_posix()] = digest
    return path, read_json(path)


def pointer_record(root, pointer, hashes, label):
    require(isinstance(pointer, dict) and {'path', 'sha256'} <= pointer.keys(), label + ' missing pointer')
    path, record = checked_json(root, pointer['path'], hashes, label, pointer['sha256'])
    if 'latest_verification' in record:
        passed(record, label + ' pointer')
        path, record = checked_json(root, record['latest_verification'], hashes, label, record['sha256'])
    passed(record, label)
    return path, record


def validate_hash_map(root, values, hashes, label):
    require(isinstance(values, dict) and values, label + ' hash inventory missing')
    for name, digest in values.items():
        path = path_under(root, name)
        require(sha256(path) == digest, label + ' changed: ' + name)
        hashes[path.relative_to(root).as_posix()] = digest


def number(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
            label + ' is not finite')
    return value


def same_number(left, right, label):
    require(abs(number(left, label) - number(right, label)) <= 1e-12, label + ' differs')


def reference_label(run, metrics, teacher):
    if run['hypothesis'] == 'HYP-091':
        return 'cached H70; original H69 is reported separately'
    if run['hypothesis'] == 'HYP-079':
        return 'cached H70'
    if run['hypothesis'] == 'HYP-102':
        return 'cached H69; incremental cached H91 also recorded'
    if run['metric'] == 'macro_f1':
        return 'cached H48 original outer' if run['stage'] == 'outer' else 'cached RAW fold0 proxy (not full H48)'
    return str(teacher.get('baseline_source', 'cached H69 compatible ' + run['stage']))


def scientific_trials(root, suite, science_path, science, hashes):
    runs = science['runs']
    require(len({r['directory'].replace('\\', '/') for r in runs}) == len(runs), 'Duplicate scientific run directory')
    rows = []
    for run in runs:
        directory = path_under(root, run['directory'])
        metric_path, metrics = checked_json(root, directory / 'metrics.json', hashes, 'trial metrics')
        require(metric_path.stat().st_mtime_ns <= science_path.stat().st_mtime_ns,
                'Metric file newer than final verification: ' + str(metric_path))
        require(metrics['status'] == run['status'], 'Trial terminal status changed')
        report_path = directory / 'report.md'
        require(report_path.is_file(), 'Missing trial report: ' + str(directory))
        hashes[report_path.relative_to(root).as_posix()] = sha256(report_path)
        row = {k: run.get(k) for k in ('hypothesis', 'stage', 'status', 'attempt_kind', 'metric', 'score', 'reference_score')}
        row['directory'] = directory.relative_to(root).as_posix()
        row['report'] = report_path.relative_to(root).as_posix()
        row['variant'] = metrics.get('variant', directory.name)
        if run['status'] == 'completed':
            require(run.get('verified') is True and all(c['passed'] for c in run['checks']), 'Unverified completed trial')
            for field in ('score', 'reference_score'):
                same_number(metrics[field], run[field], 'Trial ' + field)
            require(metrics['metric'] == run['metric'], 'Trial metric changed')
            teacher_path = directory / 'teacher-audit.json'
            teacher = read_json(teacher_path) if teacher_path.is_file() else {}
            row['cached_reference'] = reference_label(run, metrics, teacher)
            row['explicit_fit_counters'] = {k: metrics[k] for k in ('learner_fits', 'parameter_fits', 'old_model_refits', 'old_baseline_refits') if k in metrics}
            if 'H91_reference_score' in metrics:
                row['H91_reference_score'] = metrics['H91_reference_score']
        rows.append(row)
    require(len(rows) == science['registered_attempts'], 'Registered attempt count differs')
    completed = [r for r in rows if r['status'] == 'completed']
    require(len(completed) == science['completed_attempts'] == science['verified_completed_attempts'], 'Completed metric count differs')
    # Claims are recorded recipes, not an inferred count of low-level learners.
    claims = []
    for file in sorted(suite.glob('HYP-*/model-fit-history.json')):
        _, records = checked_json(root, file, hashes, 'claim registry')
        require(isinstance(records, list), 'Malformed claim registry')
        for record in records:
            claims.append({k: record.get(k) for k in ('hypothesis', 'stage', 'status', 'fingerprint')})
    require(len({r['fingerprint'] for r in claims}) == len(claims), 'Duplicate registered recipe fingerprints')
    return rows, claims


def diagnostics(root, science, hashes):
    preflights = []
    for check in science['pre_registration_diagnostic_checks']:
        require(check['passed'] is True, 'Scientific preflight diagnostic not verified')
        directory = path_under(root, check['detail']['directory'])
        _, record = checked_json(root, directory / 'metrics.json', hashes, 'pre-registration diagnostic')
        require(record['status'] == 'blocked_preflight' and record['scientific_fits'] == record['fit_claims'] == 0,
                'Pre-registration stop contains a fit or score')
        preflights.append({'directory': directory.relative_to(root).as_posix(), 'status': record['status'],
                           'scientific_fits': 0, 'fit_claims': 0, 'report': (directory / 'report.md').relative_to(root).as_posix()})
    require(len(preflights) == 2, 'Expected the two separately archived H93/H94 zero-fit preflight stops')
    diagnostic = root / 'results/2026-10-24/HYP-063-01-feature-only-audit'
    _, h63 = checked_json(root, diagnostic / 'counts.json', hashes, 'H63 diagnostic')
    require(h63['hypothesis_id'] == 'HYP-063' and h63['model_fits'] == h63['metric_evaluations'] == 0
            and h63['target_column_read'] is False, 'H63 is not a no-fit feature diagnostic')
    require(h63['status'] == 'completed_after_output_serialization_error'
            and len(h63['attempts']) == 2 and h63['attempts'][0]['status'] == 'output_error'
            and h63['attempts'][1]['status'] == 'completed', 'H63 diagnostic history incomplete')
    require(sha256(path_under(root, h63['data_path'])) == h63['declared_data_sha256'], 'H63 data SHA changed')
    for directory in [diagnostic] + [path_under(root, p['directory']) for p in preflights]:
        path = directory / 'report.md'
        hashes[path.relative_to(root).as_posix()] = sha256(path)
    return preflights, {'hypothesis': 'HYP-063', 'status': h63['status'], 'model_fits': 0,
                        'metric_evaluations': 0, 'report': (diagnostic / 'report.md').relative_to(root).as_posix(),
                        'attempts': h63['attempts'], 'claim_limit': h63['claim_limit']}


def tests_result(root, value, hashes):
    path = path_under(root, value)
    text = path.read_text(encoding='utf8', errors='strict')
    matches = re.findall(r'^Ran (\d+) tests? in ([\d.]+)s\s*\n\s*OK(?:\s*\([^\n]*\))?\s*$', text, re.MULTILINE)
    require(len(matches) == 1 and not re.search(r'^(?:FAILED|ERROR:|FAIL:)', text, re.MULTILINE), 'Final unit-test log does not show one passing suite')
    hashes[path.relative_to(root).as_posix()] = sha256(path)
    return {'path': path.relative_to(root).as_posix(), 'tests_run': int(matches[0][0]),
            'seconds': float(matches[0][1]), 'status': 'passed', 'sha256': sha256(path)}


def publish_pair(outputs, payloads):
    """Publish complete files exclusively and roll back this call on failure.

    Each directory entry is atomic; filesystems do not offer an atomic two-file
    commit. Exceptions roll back links owned by this call. Existing paths are
    never replaced or deleted. An abrupt process/OS crash is outside this
    rollback guarantee.
    """
    outputs = [Path(path) for path in outputs]
    require(len(outputs) == len(payloads) == 2 and outputs[0] != outputs[1], 'Expected two distinct output paths')
    require(outputs[0].parent == outputs[1].parent, 'Report pair must share one directory')
    # Encode both complete payloads before any filesystem mutation.
    encoded = [payload.encode('utf8') for payload in payloads]
    for path in outputs:
        if path.exists():
            raise FileExistsError('Immutable report already exists: ' + str(path))
    temporary, published = [], []
    try:
        for path, payload in zip(outputs, encoded):
            descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
            temp = Path(name)
            temporary.append(temp)
            try:
                stream = os.fdopen(descriptor, 'wb')
            except BaseException:
                os.close(descriptor)
                raise
            with stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for temp, path in zip(temporary, outputs):
            os.link(temp, path)  # Exclusive creation; never replace an existing path.
            published.append((temp, path))
    except BaseException:
        for temp, path in reversed(published):
            try:
                if os.path.samefile(temp, path):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)


def build_report(*, root=ROOT, delivery, preflight, lexical, tests, manifest=None):
    root = Path(root).resolve()
    suite = path_under(root, SUITE)
    outputs = [suite / 'REPORT.md', suite / 'summary.json']
    require(not any(p.exists() for p in outputs), 'Final report already exists; reports are immutable')
    hashes = {}
    baseline_path, baseline = checked_json(root, 'baseline.json', hashes, 'baseline')
    science_path, science = pointer_record(root, {'path': (SUITE / 'verification.json').as_posix(),
        'sha256': sha256(suite / 'verification.json')}, hashes, 'final scientific verification')
    require(science['all_attempts_terminal'] and science['all_completed_runs_verified']
            and science['checks_total'] == science['checks_passed'] and science['model_fits_performed'] == 0,
            'Final scientific verification is incomplete')
    verifier = path_under(root, science['verifier'])
    require(sha256(verifier) == science['verifier_sha256'], 'Scientific verifier changed after final check')
    hashes[verifier.relative_to(root).as_posix()] = science['verifier_sha256']
    frozen_science_path, _ = pointer_record(root, baseline['verification'], hashes, 'promoted scientific verification')
    require(frozen_science_path == science_path, 'Promotion does not use latest passed scientific verification')
    adapter_path, adapter = pointer_record(root, baseline['adapter_verification'], hashes, 'adapter verification')
    require(adapter['model_fits'] == 0 and not adapter.get('fit_attempts'), 'Adapter verification attempted a fit')
    validate_hash_map(root, adapter['source_sha256'], hashes, 'adapter runtime sources')
    validate_hash_map(root, adapter['artifact_sha256'], hashes, 'adapter historical artifacts')
    _, promotion = checked_json(root, suite / 'promotion.json', hashes, 'promotion')
    _, original = checked_json(root, suite / 'baseline-before-promotion.json', hashes, 'original baseline')
    require(promotion['model_fits'] == 0 and promotion['baseline_sha256'] == sha256(baseline_path)
            and promotion['old_baseline_sha256'] == sha256(suite / 'baseline-before-promotion.json')
            and promotion['known_source_sha256'] == sha256(root / 'automl_known.py'), 'Promotion hashes changed')
    manifest_path, package = checked_json(root, manifest or 'results/2026-10-24/package-manifest-sprint-v3.json', hashes, 'manifest')
    require(package['baseline_sha256'] == sha256(baseline_path) and package['baseline'] == baseline, 'Package baseline differs')
    for key, file in [('verification', science_path), ('adapter_verification', adapter_path)]:
        require(path_under(root, package[key]['resolved_path']) == file
                and package[key]['resolved_sha256'] == sha256(file), 'Manifest ' + key + ' differs')
    archive = path_under(root, package['file'])
    require(sha256(archive) == package['sha256'] and 0 < archive.stat().st_size < 100_000_000, 'Archive SHA/size mismatch')
    hashes[archive.relative_to(root).as_posix()] = package['sha256']
    validate_hash_map(root, package['sources'], hashes, 'package runtime sources')
    require(package['provided_requirements_exact'] is True and package['requirements_sha256'] == sha256(root / 'requirements.txt'),
            'Provided requirements not verified')
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None and zipped.namelist() == package['root_files']
                and set(zipped.namelist()) == set(package['sources']), 'ZIP inventory differs')
        for name, digest in package['sources'].items():
            require(Path(name).name == name and hashlib.sha256(zipped.read(name)).hexdigest() == digest, 'ZIP source mismatch: ' + name)
    external = {}
    for label, value in [('delivery', delivery), ('preflight', preflight), ('lexical', lexical)]:
        path, record = checked_json(root, value, hashes, label)
        passed(record, label)
        require(record['archive_sha256'] == package['sha256']
                and path_under(root, record['archive']) == archive, label + ' checked another archive')
        external[label] = {'path': path.relative_to(root).as_posix(), 'sha256': sha256(path), 'record': record}
    delivered = external['delivery']['record']
    require(delivered['total_full_refits'] == 2 and delivered['repeated_validation_fits'] == 0
            and len(delivered['records']) == 2 and {r['task'] for r in delivered['records']} == {'regression', 'multiclass'}, 'Clean full-refit evidence incomplete')
    for record in delivered['records']:
        winner = baseline[record['task']]
        require(record['full_refits'] == 1 and record['validation_fits'] == 0 and record['all_ids_order_exact'] is True
                and record['strategy'] == winner['candidate']['strategy']
                and record['source'] == winner['source'], 'Delivery candidate or row preservation differs')
        same_number(record['selected_score'], winner['score'], 'Delivery selected score')
        run = path_under(root, record['run'])
        submission = run.parent.parent / 'submission.csv'
        require(sha256(submission) == record['submission_sha256'], 'Verified submission changed')
        hashes[submission.relative_to(root).as_posix()] = record['submission_sha256']
    require(external['preflight']['record']['scenarios'] and all(r['status'] == 'passed' for r in external['preflight']['record']['scenarios']), 'No passing generic preflight scenarios')
    require(external['lexical']['record']['synthetic_only'] is True
            and external['lexical']['record']['original_ids'] == external['lexical']['record']['output_ids'], 'Lexical ID proof missing')
    test_record = tests_result(root, tests, hashes)
    rows, claims = scientific_trials(root, suite, science_path, science, hashes)
    preflights, h63 = diagnostics(root, science, hashes)
    completed = [r for r in rows if r['status'] == 'completed']
    tasks = {}
    for kind, metric, old_id, new_id in [('regression', 'rmse', 'HYP-069', 'HYP-091'), ('multiclass', 'macro_f1', 'HYP-048', 'HYP-078')]:
        old, new = original[kind], baseline[kind]
        require(old['hypothesis'] == old_id and new['hypothesis'] == new_id, 'Unexpected original/promoted hypothesis')
        verified_target = science['target']['tasks'][metric]
        same_number(old['score'], verified_target['control'], 'Original baseline score')
        best = verified_target['best_new_outer']
        same_number(new['score'], best['score'], 'Promoted best score')
        require(path_under(root, new['source']).parent == path_under(root, best['directory']), 'Promoted best source differs')
        threshold = old['score'] * (.8 if metric == 'rmse' else 1.2)
        gain = (old['score'] - new['score']) / old['score'] if metric == 'rmse' else (new['score'] - old['score']) / old['score']
        met = new['score'] <= threshold if metric == 'rmse' else new['score'] >= threshold
        require(met == verified_target['target_met'], 'Target outcome differs')
        tasks[kind] = {'metric': metric, 'original_hypothesis': old_id, 'original_score': old['score'],
            'winner_hypothesis': new_id, 'winner_score': new['score'], 'relative_gain': gain,
            'threshold': threshold, 'target_met': met, 'source': new['source']}
    both_met = all(t['target_met'] for t in tasks.values())
    require(both_met == science['target']['both_20pct_targets_met'] == promotion['target_met'] == baseline['goal']['both_targets_met'], 'Combined target outcome differs')
    explicit = Counter()
    for row in completed:
        for key, value in row['explicit_fit_counters'].items():
            require(isinstance(value, int) and value >= 0, 'Invalid explicit fit counter')
            explicit[key] += value
    acceptance = [
        {'gate': 'REQ1101: both original-outer metrics improve at least 20%', 'status': 'passed' if both_met else 'not_met', 'evidence': science_path.relative_to(root).as_posix()},
        {'gate': 'Scientific artifacts, controls, frozen configurations and metric arithmetic', 'status': 'passed', 'evidence': science_path.relative_to(root).as_posix()},
        {'gate': 'No-fit portable adapter replay', 'status': 'passed', 'evidence': adapter_path.relative_to(root).as_posix()},
        {'gate': 'Promotion and flat hash-bound package with provided requirements', 'status': 'passed', 'evidence': manifest_path.relative_to(root).as_posix()},
        *[{'gate': label, 'status': 'passed', 'evidence': external[label]['path']} for label in ('delivery', 'preflight', 'lexical')],
        {'gate': 'Final repository unit tests', 'status': 'passed', 'evidence': test_record['path']},
        {'gate': 'Real platform API and hidden score', 'status': 'not_verified', 'evidence': None},
        {'gate': 'Git publication', 'status': 'not_checked_by_this_report', 'evidence': None}]
    summary = {'spec': 'SPEC011', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'reporter_sha256': sha256(__file__), 'scientific_status': 'target_met' if both_met else 'target_not_met',
        'completed_metric_evaluations': len(completed),
        'unique_evaluated_hypothesis_ids': sorted({r['hypothesis'] for r in completed}),
        'unique_evaluated_hypotheses': len({r['hypothesis'] for r in completed}),
        'registered_attempts': len(rows), 'registered_hypothesis_ids': sorted({r['hypothesis'] for r in rows}),
        'stage_counts': dict(Counter(r['stage'] for r in completed)),
        'metric_counts': dict(Counter(r['metric'] for r in completed)),
        'zero_fit_preflight_failures': preflights, 'feature_only_diagnostic': h63,
        'recipe_claim_records': len(claims), 'recipe_claims': claims,
        'explicit_metric_fit_counter_subtotals': dict(explicit),
        'evaluations_with_explicit_learner_fit_counter': sum('learner_fits' in r['explicit_fit_counters'] for r in completed),
        'total_scientific_learner_fits': None,
        'fit_count_caveat': 'Recipe/parameter claims are not learner-fit counts; missing counters are unknown, not zero. Composite learners and reused components prevent inference from attempt counts.',
        'report_generation_model_fits': 0, 'delivery_full_refits': delivered['total_full_refits'],
        'tasks': tasks, 'both_20pct_targets_met': both_met, 'trials': rows, 'acceptance': acceptance,
        'tests': test_record, 'evidence_sha256': hashes,
        'limitations': ['Inner classification is a RAW-fold proxy, not validation of the complete H48 pipeline; some historical training meta teachers are in-sample.',
            'Outer was viewed repeatedly during adaptive selection; observed gains are exploratory, not an independent generalization guarantee.',
            'Hidden platform labels, hidden score and the real platform API are not verified.',
            'Historical scientific source SHA values refer to frozen snapshots; they are not rewritten to current runtime hashes.',
            'Zero old-refit and ledger-read statements are recorded claims, not an independent complete execution trace.']}
    report = render(summary, suite, root)
    payloads = [report, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n']
    publish_pair(outputs, payloads)
    return summary


def render(summary, suite, root):
    def link(value):
        relative = os.path.relpath(path_under(root, value), suite).replace('\\', '/')
        return '[артефакт](' + relative + ')'
    lines = ['# SPEC011: итог трёхчасового поиска', '',
        '**Цель обеих метрик +20%: ' + ('ВЫПОЛНЕНА.' if summary['both_20pct_targets_met'] else 'НЕ ВЫПОЛНЕНА.') + '**', '',
        '| Метрика | Исходный контроль | Новый лучший outer | Улучшение относительно контроля | Порог | +20% |',
        '|---|---:|---:|---:|---:|---|']
    for task in summary['tasks'].values():
        lines.append(f"| {task['metric']} | {task['original_hypothesis']}: {task['original_score']:.14g} | {task['winner_hypothesis']}: {task['winner_score']:.14g} | {task['relative_gain']:+.4%} | {task['threshold']:.14g} | {'PASS' if task['target_met'] else 'FAIL'} |")
    lines += ['', f"Завершённых вычислений метрики: **{summary['completed_metric_evaluations']}**; уникальных проверенных ID: **{summary['unique_evaluated_hypotheses']}**. Каждая inner/outer/вариантная оценка считается отдельно; составные компоненты повторно не считаются.",
        'Две остановки регистрации H93/H94 без fit/claim и один feature-only диагноз H63 не входят в число оценок и проверенных ID.', '',
        f"В реестрах {summary['recipe_claim_records']} recipe/parameter claims. Полное количество научных learner fits **не установлено**: claim, оценка и обучение отдельного компонента — разные сущности. Явные частичные счётчики приведены в summary.json; отсутствующие значения не заменены нулями.",
        f"Чистая проверка поставки отдельно выполнила {summary['delivery_full_refits']} полных refit без повторной public validation; сам этот отчёт не обучает модели.", '',
        '## Границы вывода', '', *['- ' + text for text in summary['limitations']], '',
        '## Все зарегистрированные испытания', '',
        '| ID / вариант | Этап | Статус | Метрика | Результат | Готовый контроль | Отчёт |',
        '|---|---|---|---|---:|---|---|']
    for row in summary['trials']:
        score = format(row['score'], '.14g') if row['score'] is not None else '—'
        ref = (format(row['reference_score'], '.14g') + '; ' + row['cached_reference']) if row.get('cached_reference') else '—'
        lines.append(f"| {row['hypothesis']} / {row['variant']} | {row['stage']} | {row['status']} | {row['metric'] or '—'} | {score} | {ref} | {link(row['report'])} |")
    lines += ['', '## Отдельные технические исходы', '']
    for record in summary['zero_fit_preflight_failures']:
        lines.append('- blocked_preflight: 0 scientific fits, 0 fit claims; ' + link(record['report']) + '.')
    lines += ['- H63: 0 fit, 0 metric; initial JSON TypeError, затем успешное vectorized продолжение; ' + link(summary['feature_only_diagnostic']['report']) + '.',
        '', '## Фактическая приёмка', '', '| Условие | Статус | Доказательство |', '|---|---|---|']
    for gate in summary['acceptance']:
        lines.append(f"| {gate['gate']} | {gate['status']} | {link(gate['evidence']) if gate['evidence'] else '—'} |")
    lines += ['', f"Unit tests: {summary['tests']['tests_run']} passed; {link(summary['tests']['path'])}.",
        'Полный перечень hash-bound доказательств, ID, стадий и отдельных счётчиков: [summary.json](summary.json).', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--delivery', type=Path, required=True, help='Clean delivery verification.json or directory')
    parser.add_argument('--preflight', type=Path, required=True, help='Generic flat-package verification.json or directory')
    parser.add_argument('--lexical', type=Path, required=True, help='Lexical archive verification.json or directory')
    parser.add_argument('--tests', type=Path, required=True, help='Final root unit-test log')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    summary = build_report(**vars(args))
    print(json.dumps({key: summary[key] for key in ('completed_metric_evaluations', 'unique_evaluated_hypotheses', 'both_20pct_targets_met')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
