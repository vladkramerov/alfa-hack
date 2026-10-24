"""Rebuild the human-readable registry from immutable per-run evidence."""
import argparse
import json
import os
from pathlib import Path

from experiments.run_suite import ROOT, save_json


def relative(path, start):
    return Path(os.path.relpath(path,start)).as_posix()


def update(date):
    day = ROOT/'results'/date
    records=[]
    for path in day.glob('HYP-*/metrics.json'):
        record=json.loads(path.read_text(encoding='utf-8'))
        record['_directory']=path.parent
        record['_hypothesis']='-'.join(path.parent.name.split('-')[:2])
        records.append(record)
    records.sort(key=lambda r:(r.get('started',''),r['_directory'].name))
    complete=all((day/f'series-{task}-preprocessing-v1'/'summary.json').exists() for task in ['regression','multiclass'])
    header=('# Результаты проверок\n\n'
            '[Итоговый отчёт](2026-10-23/EXPERIMENT_REPORT.md) и [CSV всех оценок](2026-10-23/comparison.csv) доступны после завершения серии.\n\n'
            'SPEC-002: два датасета, inner для выбора, заранее замороженная outer-панель. '
            'Для классификации одинаковые исходные строки сгруппированы; прежние random-split метрики напрямую не сравниваются.\n'
            'Каждая папка запуска содержит report.md, metrics.json, preprocessing audit и validation predictions. '
            'Полное обучение — технический этап без новой метрики. Экспорт полного запрошенного preprocessing хранится отдельно от выбранной модели.\n\n'
            'Повторный запуск не перезаписывает историю; правила структуры и [шаблон](_template.md) сохраняются.\n\n'
            '## Журнал\n\n'
            '| Начало (Москва) | Гипотеза | Задача / часть | Модель / recipe | Метрика | Статус | Отчёт |\n'
            '| --- | --- | --- | --- | --- | --- | --- |\n')
    rows=[]
    grouped={}
    for r in records:
        cfg=r.get('config',{})
        label=f"{cfg.get('family','—')}/{cfg.get('recipe','—')}"
        if 'strength' in cfg: label+=f" ({cfg['strength']})"
        metric=', '.join(f'{k}={v:.6f}' for k,v in r.get('scores',{}).items() if k in ['rmse','macro_f1']) or 'Не измерялась'
        stage=r.get('stage','полное обучение')
        path=r['_directory']/'report.md'
        row=f"| {r.get('started','—')} | {r['_hypothesis']} | {r['task']} / {stage} | {label} | {metric} | {r['status']} | [отчёт]({relative(path,ROOT/'results')}) |"
        rows.append(row)
        grouped.setdefault(r['_hypothesis'],[]).append((r,label,metric,stage,path))
    (ROOT/'results'/'README.md').write_text(header+'\n'.join(rows)+'\n',encoding='utf-8')
    for hyp,items in grouped.items():
        path=ROOT/'docs'/'hypotheses'/f'{hyp}.md'
        text=path.read_text(encoding='utf-8')
        text=text.split('\n## Запуски\n')[0]
        text=text.replace('- Статус: Запланирована.','- Статус: Проверена.' if complete else '- Статус: Проверяется.')
        text=text.replace('- Статус: Проверяется.','- Статус: Проверена.' if complete else '- Статус: Проверяется.')
        text+='\n## Запуски\n\n| Задача / часть | Конфигурация | Метрика | Выигрыш к указанному контролю | Статус | Отчёт |\n| --- | --- | --- | --- | --- | --- |\n'
        for r,label,metric,stage,report in items:
            gain=r.get('gain')
            text+=f"| {r['task']} / {stage} | {label} | {metric} | {'—' if gain is None else f'{gain:.6f}'} | {r['status']} | [отчёт]({relative(report,path.parent)}) |\n"
        text+='\n## Текущий вывод\n\n'
        if complete:
            text+='Вывод зависит от задачи и модели; отдельные эффекты преобразований измерены на inner, outer сравнивает замороженные варианты.\n'
            text+='Сводный вывод и ограничения: [итоговый отчёт](../../results/2026-10-23/EXPERIMENT_REPORT.md).\n'
        else:
            text+='Серия выполняется; итоговая оценка по двум задачам ещё не сформирована.\n'
        path.write_text(text,encoding='utf-8')
    progress=dict(date=date,validation_runs=sum('scores' in r for r in records),
                  technical_full_refits=sum('stage' not in r for r in records),
                  statuses={s:sum(r['status']==s for r in records) for s in ['running','completed','error','interrupted']},
                  both_suites_complete=complete)
    save_json(day/'progress.json',progress)
    print(json.dumps(progress,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',default='2026-10-23')
    update(parser.parse_args().date)
