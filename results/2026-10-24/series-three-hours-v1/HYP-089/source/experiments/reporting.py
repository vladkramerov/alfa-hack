"""Build a Russian report from completed, frozen experiment suites; never train.

Usage: python -m experiments.reporting --date 2026-10-23 --tag preprocessing-v1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
TASKS = ('regression', 'multiclass')
RECIPES = ('baseline', 'correlation', 'vif', 'scaled', 'gaussian', 'engineered')
TASK_NAMES = {'regression': 'Регрессия', 'multiclass': 'Многоклассовая классификация'}
FAMILY_NAMES = {'catboost': 'CatBoost', 'forest': 'Random Forest', 'l1': 'L1', 'l2': 'L2'}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metric(task):
    return 'rmse' if task == 'regression' else 'macro_f1'


def number(value):
    return '—' if value is None else f'{float(value):.6f}'


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def config_name(config):
    result = f"{FAMILY_NAMES[config['family']]} / {config['recipe']}"
    if 'strength' in config:
        key = 'l2_leaf_reg' if config['family'] == 'catboost' else 'strength'
        result += f" / {key}={config['strength']}"
    return result


def relative(path, report_directory):
    return Path(os.path.relpath(path, report_directory)).as_posix()


def find_record(summary, config, stage='outer'):
    candidates = [record for record in summary['records']
                  if record['stage'] == stage and record['config'] == config
                  and record['status'] == 'completed']
    if len(candidates) != 1:
        raise ValueError(f'Expected one completed {stage} record for {config}; found {len(candidates)}')
    return candidates[0]


def predictions(record, root):
    frame = pd.read_csv(root / record['path'] / 'validation_predictions.csv')
    required = {'row_position', 'actual', 'prediction'}
    if not required.issubset(frame) or frame.row_position.duplicated().any():
        raise ValueError(f"Invalid prediction artifact: {record['run_id']}")
    return frame.sort_values('row_position').reset_index(drop=True)


def recompute(frame, task):
    if task == 'regression':
        return float(np.sqrt(mean_squared_error(frame.actual, frame.prediction)))
    actual, predicted = frame.actual.astype(str), frame.prediction.astype(str)
    return float(f1_score(actual, predicted, labels=sorted(set(actual)), average='macro', zero_division=0))


def paired_comparison(first, second, task, root):
    """Return A-B on identical saved validation objects, not a significance test."""
    a, b = predictions(first, root), predictions(second, root)
    if not a.row_position.equals(b.row_position) or not a.actual.equals(b.actual):
        raise ValueError(f"Unpaired validation records: {first['run_id']} / {second['run_id']}")
    scores = [recompute(frame, task) for frame in (a, b)]
    for score, record in zip(scores, (first, second)):
        if not np.isclose(score, record['scores'][metric(task)], rtol=1e-8, atol=1e-8):
            raise ValueError(f"Saved metric differs from predictions: {record['run_id']}")
    return scores[0], scores[1], scores[0] - scores[1]


def chosen_records(summary):
    frozen = summary['frozen']
    return {
        'catboost': find_record(summary, frozen['catboost_native_l2']),
        'forest': find_record(summary, frozen['preprocessing']['forest']),
        'l1': find_record(summary, frozen['linear']['l1']),
        'l2': find_record(summary, frozen['linear']['l2']),
    }


def regression_diagnosis(series, directory, tag):
    evidence_path = series / 'method-review-evidence.json'
    if not evidence_path.is_file():
        return []
    evidence = load_json(evidence_path)
    if evidence.get('tag') != tag:
        return []
    inner, outer = evidence['ridge_inner'], evidence['ridge_outer']
    return ['### Диагностика нестабильного линейного результата', '',
            f"Ridge inner RMSE={number(inner['rmse'])}, outer RMSE={number(outer['rmse'])}. "
            f"В inner прогнозы достигают {inner['prediction_min']:.2f}. Для ноябрьской температуры "
            f"сохранённый Yeo–Johnson λ={inner['lambda_november']:.6f} при входе "
            f"{inner['raw_cold_value']:.6f} °C даёт z≈{inner['z_cold_value']:.1f}; "
            'это выход далеко за диапазон обучающей части.', '',
            f"{inner['below_fit_min_count']} строк ниже minimum температуры inner fit дают "
            f"{100*inner['below_fit_min_sse_share']:.2f}% SSE Ridge. "
            f"State_4 отсутствует в inner fit; {inner['state4_count']} его строки дают "
            f"{100*inner['state4_sse_share']:.2f}% SSE. "
            'На outer VIF уже удалил этот признак: при новом fit изменился состав предикторов. '
            'Это согласуется с нестабильной экстраполяцией при сдвиге данных, но не доказывает '
            'полную причинность одного признака; сохранённых коэффициентов линейной модели нет. '
            'Вычисление метрик и соответствие таргетов проверены отдельно, новых fit не проводилось.', '',
            f"[Подробная проверка]({relative(series / 'method-review.md', directory)}), "
            f"[численные доказательства]({relative(evidence_path, directory)}), "
            f"[крайние прогнозы]({relative(series / 'ridge-inner-tail.csv', directory)}).", '']


def engineering_sections(directory, summaries, root):
    paths = {task: directory / f'series-{task}-engineering-control-v1' / 'engineering-control-summary.json'
             for task in TASKS}
    if not all(path.is_file() for path in paths.values()):
        return [], []
    sections = ['## Дополнительная проверка сохранённого feature engineering — HYP-008', '',
                'В основном cumulative recipe pruning может удалить новые признаки. '
                'Здесь включён только engineer=True: correlation, VIF, scaling и Yeo–Johnson выключены. '
                'Это разведочный анализ после просмотра основной диагностики и части regression outer; '
                'validation split использован повторно, независимого подтверждения нет. '
                'Замороженный основной выбор и его файлы не изменены; возможный дополнительный '
                'артефакт указан отдельно.', '']
    comparison_rows, csv_rows = [], []
    for task, path in paths.items():
        supplement = load_json(path)
        if (supplement.get('status') != 'completed' or supplement.get('primary_winner_changed') is not False
                or not isinstance(supplement.get('final_refit_performed'), bool)):
            raise ValueError(f'{task}: unexpected engineering-control status or changed main winner')
        status_path = path.parent / 'engineering-control-status.json'
        if status_path.is_file():
            status = load_json(status_path)
            if status.get('status') in ('error', 'interrupted'):
                sections += [f"Технический сбой дополнительного wrapper ({TASK_NAMES[task]}): "
                             f"`{status.get('error', status['status'])}`. "
                             'Это ошибка выполнения обвязки, а не отрицательный ML-результат гипотезы; '
                             'ниже приведены завершённые оценки из итогового summary. '
                             f"[Исходный статус сбоя]({relative(status_path, directory)}).", '']
                provenance = [candidate for candidate in path.parent.glob('*.json')
                              if any(word in candidate.stem.lower() for word in ('resume', 'continuation'))]
                if provenance:
                    sections += ['Происхождение продолжения: ' + ', '.join(
                        f'[{candidate.name}]({relative(candidate, directory)})' for candidate in sorted(provenance)) + '.', '']
        for stage in ('inner', 'outer'):
            stage_names = None
            for family in ('catboost', 'forest'):
                config = {'family': family, 'recipe': 'engineering_only'}
                control = find_record(supplement, config, stage)
                baseline = find_record(summaries[task], {'family': family, 'recipe': 'baseline'}, stage)
                if control.get('reference') != baseline['run_id']:
                    raise ValueError('Engineering control uses unexpected baseline reference')
                a, b, delta = paired_comparison(control, baseline, task, root)
                control_audit = load_json(root / control['path'] / 'preprocessing_audit.json')
                baseline_audit = load_json(root / baseline['path'] / 'preprocessing_audit.json')
                new_names = [name for name in control_audit['numeric_columns']
                             if name not in baseline_audit['numeric_columns']]
                if not new_names or control_audit['dropped_columns'] or control_audit.get('vif_dropped_columns'):
                    raise ValueError('Engineering-only must retain new fields without pruning')
                stage_names = new_names
                comparison_rows.append([TASK_NAMES[task], stage, FAMILY_NAMES[family], number(b),
                                        number(a), number(delta), len(new_names),
                                        f"[run]({relative(root / control['path'] / 'report.md', directory)})"])
                csv_rows.append(dict(task=task, series='engineering-control-v1', stage=stage,
                                     run_id=control['run_id'], **config, **control['scores'],
                                     seconds=control.get('seconds'), n_features=control.get('n_features'),
                                     converged=control.get('converged'), reference=control.get('reference'),
                                     gain=control.get('gain'),
                                     report=relative(root / control['path'] / 'report.md', directory)))
            sections += [f"{TASK_NAMES[task]}, {stage}: сохранены {len(stage_names)} новых numeric: "
                         + ', '.join(f'`{name}`' for name in stage_names) + '.', '']
        sections += [f"[Supplement summary — {TASK_NAMES[task]}]({relative(path, directory)}).", '']
        if supplement['final_refit_performed']:
            final = supplement['final']
            selection = supplement['supplemental_selection']
            if final.get('status') != 'completed' or final['config'] != selection['selected_config']:
                raise ValueError('Supplemental refit differs from its frozen selection or is incomplete')
            sections += [f"Дополнительный exploratory артефакт {TASK_NAMES[task]}: "
                         f"`{config_name(final['config'])}`. Его выбор зафиксирован по дополнительному inner "
                         'перед дополнительной outer-оценкой; основной winner сохраняется.', '']
            for name in ('submission.csv', 'model.joblib', 'preprocessed_data.npz', 'metrics.json'):
                artifact = root / final['path'] / name
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
                sections.append(f'- [{TASK_NAMES[task]}: дополнительный {name}]({relative(artifact, directory)})')
            sections += ['']
    sections += [table(['Задача', 'Часть', 'Модель', 'Baseline', 'Engineering only', 'Δ к baseline',
                        'Новых numeric', 'Артефакты'], comparison_rows), '',
                 'Для регрессии таблица показывает RMSE (меньше лучше), для классификации Macro F1 '
                 '(больше лучше). Новые numeric включают индикаторы пропусков, если они появились. '
                 'Улучшение в этой дополнительной проверке не изменяет основной заранее замороженный winner.', '']
    return sections, csv_rows


def load_suites(root, date, tag):
    directory = root / 'results' / date
    summaries = {}
    for task in TASKS:
        series = directory / f'series-{task}-{tag}'
        summary = load_json(series / 'summary.json')
        if summary['task'] != task or summary.get('final', {}).get('status') != 'completed':
            raise ValueError(f'{task}: suite/final refit is incomplete')
        frozen_file = load_json(series / 'frozen_selection.json')
        if summary['frozen'] != frozen_file:
            raise ValueError(f'{task}: summary and frozen selection differ')
        for config in frozen_file['outer_panel']:
            find_record(summary, config)
        summaries[task] = summary
    return directory, summaries


def plot_results(directory, summaries, audits, root):
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'tmp' / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = directory / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for family in ('catboost', 'forest'):
            records = [find_record(summaries[task], {'family': family, 'recipe': recipe}, 'inner')
                       for recipe in RECIPES]
            ax.plot(RECIPES, [r['scores'][metric(task)] for r in records], marker='o', label=FAMILY_NAMES[family])
        ax.set_title(TASK_NAMES[task] + ' — inner')
        ax.set_ylabel('RMSE ↓' if task == 'regression' else 'Macro F1 ↑')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(plot_dir / 'inner_ablation.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, task in enumerate(TASKS):
        selected = chosen_records(summaries[task])
        for column, families in enumerate((('catboost', 'forest'), ('l1', 'l2'))):
            ax = axes[row, column]
            values = [selected[f]['scores'][metric(task)] for f in families]
            bars = ax.bar([FAMILY_NAMES[f] for f in families], values, color=['#315e9b', '#cf7845'])
            for bar, value in zip(bars, values):
                ax.annotate(number(value), (bar.get_x()+bar.get_width()/2, bar.get_height()),
                            xytext=(0, 4), textcoords='offset points', ha='center')
            ax.set_title(TASK_NAMES[task] + ' — frozen outer')
            ax.set_ylabel('RMSE ↓' if task == 'regression' else 'Macro F1 ↑')
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
            ax.grid(axis='y', alpha=.25)
    fig.savefig(plot_dir / 'outer_comparison.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        audit = audits[task]
        names = sorted(audit['skewness_before'], key=lambda name: abs(audit['skewness_before'][name]), reverse=True)[:10]
        positions = np.arange(len(names))
        ax.scatter([audit['skewness_before'][n] for n in names], positions, label='До Yeo–Johnson', marker='o')
        ax.scatter([audit['skewness_after'][n] for n in names], positions, label='После Yeo–Johnson', marker='x')
        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        ax.set_title(TASK_NAMES[task] + ' — engineered inner fit')
        ax.set_xlabel('Выборочная асимметрия (skewness)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(plot_dir / 'skewness_before_after.png', dpi=170)
    plt.close(fig)


def build_report(root, date, tag):
    root = Path(root)
    directory, summaries = load_suites(root, date, tag)
    rows, sections, audits = [], [], {}
    for task, summary in summaries.items():
        key = metric(task)
        series = directory / f'series-{task}-{tag}'
        split = load_json(series / 'splits.json')
        selected = chosen_records(summary)
        # Check prediction pairing and metrics before writing aggregate outputs.
        tree_pair = paired_comparison(selected['catboost'], selected['forest'], task, root)
        linear_pair = paired_comparison(selected['l1'], selected['l2'], task, root)
        native_control = find_record(summary, summary['frozen']['preprocessing']['catboost'])
        native_pair = paired_comparison(selected['catboost'], native_control, task, root)
        outer_trees = [r for r in summary['records'] if r['stage'] == 'outer' and r['status'] == 'completed'
                       and r['config']['family'] in ('catboost', 'forest')]
        observed_best = (min if task == 'regression' else max)(outer_trees, key=lambda r: r['scores'][key])
        final_outer = find_record(summary, summary['frozen']['best_tree_for_final_refit'])
        same_family_baseline = find_record(summary, {'family': final_outer['config']['family'], 'recipe': 'baseline'})
        final_baseline_pair = paired_comparison(final_outer, same_family_baseline, task, root)
        engineered = find_record(summary, {'family': 'catboost', 'recipe': 'engineered'}, 'inner')
        audit = load_json(root / engineered['path'] / 'preprocessing_audit.json')
        audits[task] = audit
        sections += [f'## {TASK_NAMES[task]}', '',
                     f"**Замороженный выбор по inner:** `{config_name(final_outer['config'])}`, "
                     f"outer {key}={number(final_outer['scores'][key])}. "
                     f"**Лучший наблюдавшийся outer среди деревьев:** `{config_name(observed_best['config'])}`, "
                     f"{key}={number(observed_best['scores'][key])}; это описательный минимум/максимум "
                     'панели, его использование для выбора создало бы оптимизм outer.', '',
                     f"У замороженного inner-победителя Δ outer к baseline того же семейства: "
                     f"{number(final_baseline_pair[2])}. "
                     'Итоговый файл сохраняет заранее выбранный вариант; он не объявляется доказанно лучшим.', '',
                     f"Разбиение: `{summary['split']}`.", '',
                     table(['Часть', 'Объектов'], split['sizes'].items()), '']
        if task == 'multiclass':
            if 'group' in summary['split'].lower():
                sections += ['Использован StratifiedGroupKFold: первый fold из 5 для outer, '
                             'первый fold из 4 внутри outer fit для inner, seed42. '
                             'Группы — одинаковые исходные строки признаков без id и таргета; '
                             'группа целиком находится по одну сторону split. '
                             'Размеры и баланс могут отличаться от точных 80/20 и 75/25.', '']
            counts = split.get('class_counts', {}).get('outer_valid', {})
            if counts:
                rare, count = min(counts.items(), key=lambda pair: pair[1])
                sections += [f'Самый редкий класс outer: **{rare}**, support={count}; '
                             'его F1 чувствителен к нескольким ошибкам.', '']
        sections += ['### Поэтапная предобработка — inner', '',
                     'Baseline: imputation + one-hot. Последовательность cumulative: '
                     'correlation → VIF → StandardScaler → Yeo–Johnson → engineering. '
                     'Engineering вычисляется в исходных единицах перед pruning и преобразованиями. '
                     'Δ — текущая метрика минус предыдущая; для RMSE лучше отрицательное значение, '
                     'для Macro F1 — положительное.', '']
        inner_rows = []
        for family in ('catboost', 'forest'):
            previous = None
            for recipe in RECIPES:
                record = find_record(summary, {'family': family, 'recipe': recipe}, 'inner')
                score = record['scores'][key]
                inner_rows.append([FAMILY_NAMES[family], recipe, number(score),
                                   number(None if previous is None else score-previous), record['n_features'],
                                   number(record.get('seconds'))])
                previous = score
        sections += [table(['Модель', 'Recipe', key, 'Δ к предыдущему', 'Признаков', 'Секунд'], inner_rows), '',
                     '### Замороженная панель — outer', '',
                     'Все конфигурации выбраны до outer. Сравнение cumulative recipe с baseline '
                     'измеряет совокупный эффект pipeline; отдельный вклад шага показан только на inner.', '']
        outer_rows = []
        for record in summary['records']:
            if record['status'] != 'completed':
                continue
            config = record['config']
            record_path = root / record['path']
            rows.append(dict(task=task, series=tag, stage=record['stage'], run_id=record['run_id'], **config,
                             **record['scores'], seconds=record.get('seconds'), n_features=record.get('n_features'),
                             converged=record.get('converged'), reference=record.get('reference'),
                             gain=record.get('gain'), report=relative(record_path / 'report.md', directory)))
            if record['stage'] == 'outer':
                outer_rows.append([config_name(config), number(record['scores'][key]),
                                   number(record.get('seconds')), 'да' if record.get('converged') else 'нет',
                                   f"[run]({relative(record_path / 'report.md', directory)})"])
        sections += [table(['Конфигурация', key, 'Секунд', 'Сходимость', 'Артефакты'], outer_rows), '',
                     'Парные сравнения используют одни и те же сохранённые validation-объекты; '
                     'метрики пересчитаны из прогнозов. Δ=A−B; это разность метрик, не тест значимости.', '',
                     table(['A', 'B', key+' A', key+' B', 'Δ A−B'], [
                         ['CatBoost: inner winner + native L2', 'Random Forest: inner winner', *map(number, tree_pair)],
                         ['L1: inner strength winner', 'L2: inner strength winner', *map(number, linear_pair)],
                         ['CatBoost: выбранный native L2', 'CatBoost: тот же recipe, l2=3', *map(number, native_pair)],
                     ]), '',
                     f"Выбранный CatBoost recipe: `{selected['catboost']['config']['recipe']}`, "
                     f"l2_leaf_reg={selected['catboost']['config'].get('strength', 3)}; "
                     f"RF recipe: `{selected['forest']['config']['recipe']}`.", '',
                     'L1/L2 относятся к Lasso/Ridge в регрессии и multinomial LogisticRegression '
                     'в классификации, а не к Random Forest. Native L2 CatBoost показан отдельно. '
                     'Числа alpha у Lasso и Ridge имеют разные нормировки и не означают равную силу штрафа.', '']
        linear_grid = [r for r in summary['records'] if r['stage'] == 'inner' and r['status'] == 'completed'
                       and r['config']['family'] in ('l1', 'l2')]
        sections += ['Подбор линейных штрафов на inner (сходимость не гарантирует хорошего качества):', '',
                     table(['Модель', 'Сила штрафа / C', key, 'Сходимость'],
                           [[FAMILY_NAMES[r['config']['family']], r['config']['strength'], number(r['scores'][key]),
                             'да' if r.get('converged') else 'нет'] for r in linear_grid]), '']
        warned = [r for r in summary['records'] if r.get('warnings')]
        if warned:
            sections += ['Предупреждения обучения:', '']
            for record in warned:
                sections.append(f"- `{record['run_id']}`: " + '; '.join(dict.fromkeys(record['warnings'])))
            sections += ['']
        unreliable = [name for name in ('l1', 'l2') if not selected[name].get('converged', False)
                      or not find_record(summary, selected[name]['config'], 'inner').get('converged', False)]
        if unreliable:
            sections += ['**Недостаточно данных для надёжного сравнения штрафов:** '
                         + ', '.join(unreliable) + ' не сошлись на inner или outer; '
                         'значения outer приведены диагностически.', '']
        else:
            sections += ['Оба выбранных линейных outer fit завершились без ConvergenceWarning; '
                         'вывод всё равно ограничен одним split и проверенной сеткой.', '']
        if task == 'regression':
            sections += regression_diagnosis(series, directory, tag)
        if task == 'multiclass':
            per_class = {}
            for family, record in selected.items():
                frame = predictions(record, root)
                per_class[family] = classification_report(frame.actual.astype(str), frame.prediction.astype(str),
                                                         output_dict=True, zero_division=0)
            labels = [label for label, values in per_class['catboost'].items() if isinstance(values, dict)
                      and label not in ('macro avg', 'weighted avg', 'micro avg')]
            labels.sort(key=lambda label: per_class['catboost'][label]['support'])
            class_rows = [[label, int(per_class['catboost'][label]['support']),
                           *[number(per_class[family][label]['f1-score']) for family in selected]] for label in labels]
            sections += ['### F1 каждого класса — outer', '',
                         table(['Класс', 'Support', 'CatBoost F1', 'RF F1', 'L1 F1', 'L2 F1'], class_rows), '',
                         'Полные precision/recall/F1 сохранены в `classification_report.txt` каждого run; '
                         'порядок строк выше — от редких классов к частым.', '']
        sections += ['### Диагностика полного engineered recipe на inner fit', '',
                     'Удалены по Pearson/Spearman ≥0.95: '
                     + (', '.join(f"`{entry['column']}`" for entry in audit['dropped_columns']) or 'нет') + '.', '',
                     'Удалены при итеративном VIF>10/вырожденности: '
                     + (', '.join(f"`{entry['column']}` ({entry['reason']})" for entry in audit.get('vif_dropped_columns', [])) or 'нет') + '.', '',
                     'VIF относится только к числовым признакам после median imputation, '
                     'до нелинейных преобразований и one-hot; его порог не гарантируется для итоговой матрицы.', '',
                     'Yeo–Johnson fit: ' + (', '.join(f'`{name}`' for name in audit['gaussian_columns']) or 'нет') + '.', '']
        if audit.get('gaussian_skipped'):
            sections += ['Пропущены из-за численной ошибки: ' + str(audit['gaussian_skipped']) + '.', '']
        names = sorted(audit['skewness_before'], key=lambda name: abs(audit['skewness_before'][name]), reverse=True)[:10]
        sections += [table(['Оставленный numeric', 'Skew до', 'Skew после'],
                           [[name, number(audit['skewness_before'][name]), number(audit['skewness_after'][name])]
                            for name in names]), '',
                     '### Итоговое обучение', '',
                     f"На полном train переобучен inner-победитель деревьев: `{config_name(summary['final']['config'])}`. "
                     'Выбор не менялся по outer; скрытый test здесь не оценён.', '']
        final_directory = root / summary['final']['path']
        for name in ('submission.csv', 'model.joblib', 'preprocessed_data.npz', 'preprocessing_audit.json', 'metrics.json'):
            if not (final_directory / name).is_file():
                raise FileNotFoundError(final_directory / name)
            sections.append(f'- [{name}]({relative(final_directory / name, directory)})')
        sections += ['', f"[Замороженный выбор]({relative(series / 'frozen_selection.json', directory)}), "
                     f"[полный summary]({relative(series / 'summary.json', directory)}), "
                     f"[окружение и hashes]({relative(series / 'environment.json', directory)}).", '']
    extra_sections, extra_rows = engineering_sections(directory, summaries, root)
    sections += extra_sections
    rows += extra_rows
    plot_results(directory, summaries, audits, root)
    intro = ['# Результаты экспериментов по предобработке и регуляризации', '',
             f'Дата: {date}; серия: `{tag}`. Значения ниже получены из завершённых runs; '
             'генератор не обучает модели и не изменяет историю экспериментов.', '',
             '[Все оценки: comparison.csv](comparison.csv)', '',
             '![Поэтапная предобработка на inner](plots/inner_ablation.png)', '',
             '![Замороженные конфигурации на outer](plots/outer_comparison.png)', '',
             '![Асимметрия до и после преобразования](plots/skewness_before_after.png)', '']
    limitations = ['## Ограничения выводов', '',
                   'Каждый эффект относится к этим данным, одному validation split и фиксированному бюджету. '
                   'Подбор на inner создаёт оптимизм inner-результатов; outer использован только для '
                   'заранее замороженной панели. Разности метрик не доказывают универсального превосходства '
                   'модели и не оценивают leaderboard или скрытый test.', '',
                   'Стандартное масштабирование почти не меняет деревья. Нормальность признаков '
                   'и таргета не является их требованием; Yeo–Johnson не гарантирует нормального распределения. '
                   'Таргет не преобразовывался. Приведённая асимметрия характеризует marginal distribution '
                   'оставленных числовых признаков, а не совместную нормальность.', '',
                   'Точный групповой split классификации защищает от одинаковых строк по обе стороны '
                   'разбиения, но не доказывает независимость всех сетевых наблюдений. '
                   'Регрессия проверена переносом на следующий год; проверка не охватывает все возможные '
                   'временные сдвиги.', '',
                   '## Документация методов', '',
                   '[CatBoost: l2_leaf_reg](https://catboost.ai/docs/en/references/training-parameters/common), '
                   '[Random Forest: параметры и cost-complexity pruning]'
                   '(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), '
                   '[StandardScaler и Yeo–Johnson](https://sklearn.org/stable/modules/preprocessing.html), '
                   '[масштабирование и деревья]'
                   '(https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html).', '']
    pd.DataFrame(rows).to_csv(directory / 'comparison.csv', index=False)
    output = directory / 'EXPERIMENT_REPORT.md'
    output.write_text('\n'.join(intro + sections + limitations), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--tag', default='preprocessing-v1')
    args = parser.parse_args()
    print(build_report(ROOT, args.date, args.tag))


if __name__ == '__main__':
    main()
