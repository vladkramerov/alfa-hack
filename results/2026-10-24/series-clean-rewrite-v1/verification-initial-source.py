"""Independent arithmetic, split, source and lineage audit; no estimator fits."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'results/2026-10-24/series-clean-rewrite-v1'


def frame_digest(frame):
    digest = hashlib.sha256(json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def main():
    checks, comparisons = [], {}

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})

    plan = json.loads((SUITE / 'frozen-plan.json').read_text(encoding='utf8'))
    attempts = json.loads((SUITE / 'attempts.json').read_text(encoding='utf8'))
    history = json.loads((SUITE / 'hypotheses.json').read_text(encoding='utf8'))
    for name, digest in plan['sources'].items():
        check('immutable source ' + name, hashlib.sha256((SUITE / 'source' / name).read_bytes()).hexdigest() == digest)
    for name in plan['sources']:
        if not Path(name).name.startswith('research_'):
            continue
        parsed = ast.parse((SUITE / 'source' / name).read_text(encoding='utf8'))
        imports = [node.module or '' for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(parsed) if isinstance(node, ast.Import) for alias in node.names]
        check('no old agent dependency ' + name, not any(i.startswith('automl_') for i in imports))
    for task_name in ['regression', 'multiclass']:
        train_path = ROOT / 'data' / task_name / 'train.csv'
        target = 'site_eui' if task_name == 'regression' else 'attack_cat'
        train = pd.read_csv(train_path, dtype=str, keep_default_na=False)
        check('original train SHA ' + task_name, hashlib.sha256(train_path.read_bytes()).hexdigest() == plan['datasets'][task_name]['hashes']['train'])
        for role, filename in [('test', 'test.csv'), ('sample', 'sample_submission.csv'), ('dictionary', 'data_dictionary.csv'), ('task', 'task.md')]:
            check('original ' + role + ' SHA ' + task_name, hashlib.sha256((train_path.parent / filename).read_bytes()).hexdigest() == plan['datasets'][task_name]['hashes'][role])
        fit_features = train.drop(columns=['id', target]).copy()
        for column in fit_features:
            original = fit_features[column]
            present = original.ne('')
            numeric = pd.to_numeric(original.where(present), errors='coerce')
            fit_features[column] = numeric if present.any() and numeric[present].notna().all() else original.where(present, np.nan)
        fit_target = pd.to_numeric(train[target]) if task_name == 'regression' else train[target]
        for stage in ['inner', 'confirmation']:
            with np.load(SUITE / task_name / f'{stage}-split.npz', allow_pickle=False) as split:
                fit, valid = split['fit'], split['valid']
            digest = hashlib.sha256(np.asarray(fit, dtype='<i8').tobytes() + b'|' + np.asarray(valid, dtype='<i8').tobytes()).hexdigest()
            check(task_name + '/' + stage + ' frozen split', digest == plan['datasets'][task_name]['stages'][stage]['split_sha256'])
            check(task_name + '/' + stage + ' disjoint valid positions', not set(fit) & set(valid) and len(set(valid)) == len(valid) and min(valid) >= 0 and max(valid) < len(train))
            if task_name == 'regression':
                years = train['Year_Factor'].astype(float).to_numpy()
                check(stage + ' strictly past training', years[fit].max() < years[valid].min())
            else:
                groups = pd.util.hash_pandas_object(train.drop(columns=['id', target]), index=False).to_numpy()
                check(stage + ' raw groups do not cross', not set(groups[fit]) & set(groups[valid]))
        with np.load(SUITE / task_name / 'confirmation-split.npz') as confirmation, np.load(SUITE / task_name / 'inner-split.npz') as inner:
            check(task_name + ' inner wholly inside confirmation training', set(inner['fit']) | set(inner['valid']) == set(confirmation['fit']))
            confirmation_positions = confirmation['valid'].copy()
        with np.load(ROOT / f'results/2026-10-23/series-{task_name}-preprocessing-v1/splits.npz') as previous:
            check(task_name + ' confirmation differs from old outer', not np.array_equal(confirmation_positions, previous['outer_valid']))
        scored = {}
        for event in [r for r in attempts if r['task'] == task_name]:
            stage = event['stage']
            directory = ROOT / event['directory']
            metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf8'))
            check(task_name + '/' + stage + ' terminal status', metrics['status'] != 'started')
            check(task_name + '/' + stage + ' human ledger read before fit', bool(event['ledger_sha256_before_fit']))
            if metrics['status'] != 'completed':
                continue
            with np.load(directory / 'validation_predictions.npz', allow_pickle=False) as saved, np.load(directory / 'splits.npz', allow_pickle=False) as split:
                positions, prediction = saved['positions'], saved['prediction']
                expected_stage = 'confirmation' if stage == 'control' else stage
                with np.load(SUITE / task_name / f'{expected_stage}-split.npz', allow_pickle=False) as frozen_split:
                    check(task_name + '/' + stage + ' per-run split equals frozen protocol',
                          np.array_equal(split['fit'], frozen_split['fit']) and np.array_equal(split['valid'], frozen_split['valid']))
                matches = [row for row in history if row['fingerprint'] == metrics.get('fingerprint')]
                actual_claim = matches[0] if len(matches) == 1 else {}
                expected_identity = {'recipe': event['recipe'],
                                     'X_sha256': frame_digest(fit_features.iloc[split['fit']]),
                                     'y_sha256': frame_digest(pd.DataFrame(fit_target.iloc[split['fit']].to_numpy().reshape(-1, 1)))}
                fingerprint = hashlib.sha256(json.dumps(expected_identity, sort_keys=True, default=str).encode()).hexdigest()
                check(task_name + '/' + stage + ' actual fit lineage matches frozen training data',
                      len(matches) == 1 and actual_claim.get('status') == 'completed'
                      and actual_claim.get('fingerprint') == fingerprint and event.get('fingerprint') == fingerprint)
                expected_y = train.iloc[positions][target].to_numpy()
                if task_name == 'regression':
                    expected_y = expected_y.astype(float)
                check(task_name + '/' + stage + ' saved y from original CSV', np.array_equal(saved['y'], expected_y))
                check(task_name + '/' + stage + ' exact validation positions', np.array_equal(positions, split['valid']))
                independent = float(np.sqrt(mean_squared_error(expected_y, prediction))) if task_name == 'regression' else float(f1_score(expected_y, prediction, average='macro', zero_division=0))
                check(task_name + '/' + stage + ' independent score', abs(independent - metrics['score']) < 1e-10)
                check(task_name + '/' + stage + ' finite correct-shape predictions', len(prediction) == len(positions) and (task_name != 'regression' or np.isfinite(prediction).all()))
                if task_name == 'multiclass':
                    probabilities = saved['probabilities']
                    check(stage + ' class/probability contract', probabilities.shape == (len(positions), len(saved['classes'])) and np.isfinite(probabilities).all() and np.all(probabilities >= 0) and np.allclose(probabilities.sum(axis=1), 1.) and set(prediction) <= set(train[target]))
            scored[stage] = independent
            if stage == 'control':
                check(task_name + ' no foreign frozen meta', metrics.get('frozen_public_meta_reused') is False)
                if task_name == 'multiclass':
                    audit = json.loads((directory / 'audit.json').read_text(encoding='utf8'))
                    check('H48 newly learned training-only meta', audit.get('state_source') == 'training_grouped_oof')
        if 'confirmation' in scored and 'control' in scored:
            control, candidate = scored['control'], scored['confirmation']
            gain = (control - candidate) / control if task_name == 'regression' else (candidate - control) / control
            comparisons[task_name] = {'control': control, 'candidate': candidate, 'relative_gain': gain, 'inner': scored.get('inner')}
        else:
            comparisons[task_name] = {'status': 'insufficient_data', 'available_scores': scored}
    check('actual learner claims all unique', len({r['fingerprint'] for r in history}) == len(history))
    check('actual learner claims all terminal and ledger read', all(r['status'] != 'started' and r['ledger_sha256'] for r in history))
    control_path = SUITE / 'control-fits/model-fit-attempts.jsonl'
    controls = [json.loads(line) for line in control_path.read_text(encoding='utf8').splitlines()] if control_path.exists() else []
    check('old component fits unique', len({r['fingerprint'] for r in controls}) == len(controls))
    for name in ['HYP-052', 'HYP-062']:
        elapsed = sum(float(r.get('seconds', 0.)) for r in attempts if r['hypothesis'] == name)
        check(name + ' within total 900sec worker budget', elapsed <= 900.)
    # Independently apply the agreed two-task trade-off; no import of product gate.
    complete = all('relative_gain' in result for result in comparisons.values())
    gains = [result['relative_gain'] for result in comparisons.values()] if complete else []
    eps = 1e-12
    accepted = all(c['passed'] for c in checks) and complete and ((min(gains) >= -eps and max(gains) > eps) or (max(gains) > .15 + eps and min(gains) > -.1 + eps))
    report = {'status': 'passed' if all(c['passed'] for c in checks) else 'failed',
              'checks_passed': sum(c['passed'] for c in checks), 'checks': checks, 'comparisons': comparisons,
              'promotion': bool(accepted), 'learner_claims': len(history), 'old_component_fit_claims': len(controls),
              'verifier_model_fits': 0, 'private_score_verified': False}
    (SUITE / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# SPEC010: независимый агент, H52/H62', '',
             f'Сверка: {report["status"]}, {report["checks_passed"]}/{len(checks)}. Verifier model fits:0.', '',
             '| Задача | Текущий лидер на новом split | Новый кандидат на том же split | Относительный выигрыш |',
             '|---|---:|---:|---:|']
    for name, result in comparisons.items():
        if 'relative_gain' in result:
            lines.append(f'| {name} | {result["control"]:.9f} | {result["candidate"]:.9f} | {result["relative_gain"]:+.2%} |')
        else:
            lines.append(f'| {name} | — | — | Недостаточно данных |')
    lines += ['', f'Правило поставки выполнено: **{bool(accepted)}**.',
              'Направление: RMSE меньше — лучше; Macro-F1 больше — лучше. Старый public outer не переобучался. Новое confirmation — дополнительный протокол на уже известных публичных данных, не private test.',
              'Один кандидат на задачу, параметры и источники зафиксированы до fit. Контроль H48 переобучен без frozen meta-state иного fold по прямому разрешению пользователя.',
              'Если выигрыша нет, baseline.json и ранее опубликованный ZIP сохраняются. Новая реализация не получает название «улучшенная» только за переписывание.', '', '[Подробная сверка](verification.json), [замороженный план](frozen-plan.json), [попытки](attempts.json).']
    (SUITE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in ['status', 'checks_passed', 'comparisons', 'promotion', 'learner_claims', 'old_component_fit_claims']}, ensure_ascii=False))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
