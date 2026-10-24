"""Build teammate-facing Markdown directly from measured artifacts."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'results/2026-10-24/sergey-improvement-v1'


def main():
    verified=json.loads((SUITE/'independent-metric-verification.json').read_text())
    delivery=json.loads((SUITE/'clean-public-delivery/verification.json').read_text())
    package=json.loads((SUITE/'package-manifest.json').read_text())
    assert verified['status']==delivery['status']=='passed'
    assert delivery['archive_sha256']==package['sha256']
    fresh=ROOT/'results/2026-10-24/HYP-075-02-multiclass-clean-runtime-continuation'
    fresh_metric=json.loads((fresh/'metrics.json').read_text())
    with np.load(fresh/'validation_predictions.npz',allow_pickle=False) as z:
        macro=float(f1_score(z['y'],z['prediction'],average='macro',zero_division=0))
        weighted=float(f1_score(z['y'],z['prediction'],average='weighted',zero_division=0))
        accuracy=float(accuracy_score(z['y'],z['prediction']))
    assert abs(macro-fresh_metric['score'])<1e-12
    baseline=json.loads((ROOT/'baseline.json').read_text())
    baseline['multiclass']['previous_historical_score']=.6495311620074445
    baseline['multiclass']['previous_same_environment']=.6443063238768081
    baseline['multiclass']['score']=macro
    baseline['multiclass']['candidate'].pop('meta_state',None)
    baseline['multiclass']['candidate']['hypothesis']='HYP-075 task-local grouped OOF stacking; no embedded weights'
    baseline['multiclass']['source']='results/2026-10-24/HYP-075-02-multiclass-clean-runtime-continuation/report.md'
    baseline['multiclass']['hypothesis']='HYP-075'
    (ROOT/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2))
    labels={70:'Устойчивый профиль без ELEVATION; прежняя формула памяти',
            71:'Последнее прошлое наблюдение; доверие из training-пар',
            72:'Причинная модель годового изменения относительно истории',
            73:'Фиксированное 50/50 объединение H70 и H72',
            74:'Прямое среднее target по состоянию панели, ExtraTrees'}
    lines=['# Результаты Сергея: проверка решения и готовый агент',
      '', 'Дата: 24 октября 2026. Все изменения выполнены в `sergey_branch`.',
      '', '## Что сдавать', '',
      '[**solution_sergey_panel_v1.zip**](../dist/solution_sergey_panel_v1.zip) — единственный актуальный архив этой работы. Старые ZIP в dist сохранены как история.',
      '',f"Размер **{package['bytes']:,} байт**, **17 файлов строго в корне**, **0 папок**. SHA256: `{package['sha256']}`.",
      '', 'На платформу загружается сам ZIP. CSV с готовыми предсказаниями, папку проекта, `.env` и отчёт в него добавлять не нужно.',
      '', '## Что изменилось', '',
      'Регрессия: вместо требования совпадения всех полей вместе с ELEVATION используется устойчивый профиль. Он находит больше прошлых наблюдений. Итог объединяет такую память с моделью изменения энергопотребления; обе части используют только строго предыдущие годы. Для неизвестного профиля остаётся прежний H43. Совпадение профиля не объявляется доказанной идентичностью здания; коллизии сохраняются в аудитах.',
      '', 'Классификация: прежние три ветви H48 сохранены, но их метамодель теперь обучается на grouped OOF внутри текущей задачи. Из ZIP удалены заранее сохранённые классы, веса, метрики и выбор по SHA публичных данных. Все задачи стартуют с пустой встроенной историей; дальнейшая память находится только в каталоге конкретной задачи.',
      '', 'Это универсальный агент с условно применяемыми стратегиями: незнакомые данные без физического профиля используют обычный портфель моделей. Параметры и алгоритмы входят в код; заранее обученные модели, обучающие данные и секреты в ZIP отсутствуют.',
      '', '## С чем сравнивали', '',
      'Источник: `ivan_branch@4283105bc445d6511680901a714687300b77d29d`. Лучший сохранённый архив: `solution_alternative_v2.zip`, SHA256 `a51d351d8657aa04bd6a1628fa0204de3a805f49a1de29f732aa793dd8901cb7`. Последняя SPEC010 не улучшила его и не использована как более слабый контроль.',
      '', 'Сохранённые validation predictions старых запусков отсутствовали в Git, поэтому нужные контроли воспроизведены один раз на Mac. Различия Windows/Mac не скрываются: исторические39.392194/.649531 не подставляются вместо полученных на этом Mac39.366894/.644306. Все сравнения ниже используют одинаковые данные, splits и окружение.',
      '', '## Условия проверки', '',
      '| Задача/протокол | Обучение | Проверка | Метрика |',
      '| --- | ---: | ---: | --- |',
      '| Регрессия primary: Year<5 →5 | 35 000 | 18 308 | RMSE, меньше лучше |',
      '| Регрессия secondary: Year<4 →4 | 22 054 | 12 946 | RMSE |',
      '| Классификация primary: grouped5 fold0 seed42 | 70 529 | 17 141 | Macro-F1, больше лучше |',
      '| Классификация secondary: grouped5 fold1 seed42 | 69 916 | 17 754 | Macro-F1 |',
      '', 'Группы классификации определяются одинаковыми исходными X, ID исключён. Между train и validation одинаковые группы не пересекаются. Все медианы, кодировщики, динамика и калибровка учатся на fit; validation/test targets в fit не участвуют.',
      '', '## Все новые модельные проверки', '',
      '| ID | Что проверено | RMSE primary | Выигрыш primary | RMSE secondary | Выигрыш secondary | Решение |',
      '| --- | --- | ---: | ---: | ---: | ---: | --- |',
      '| Контроль H69 | Исходный лучший регрессионный алгоритм Ивана, воспроизведение на Mac | 39.366894 | — | 42.443824 | — | Сохранён для сравнения |']
    for number in [70,71,72,73,74]:
        pair=[next(x for x in verified['regression'] if x['hypothesis']==f'HYP-{number:03d}' and x['stage']==stage) for stage in ['primary','secondary']]
        a,b=pair
        link=f'../docs/hypotheses/HYP-{number:03d}.md'
        choice='**Включён в ZIP**' if number==73 else 'Лучше исходного, уступил финалисту по primary'
        lines.append(f'| [H{number}]({link}) | {labels[number]} | {a["rmse"]:.6f} | {100*a["relative_gain"]:.2f}% | {b["rmse"]:.6f} | {100*b["relative_gain"]:.2f}% | {choice} |')
    control=verified['classification'][0]['macro_f1'];secondary=verified['classification'][1]['macro_f1']
    lines += ['', '### Классификация и переносимость', '',
      '| Вариант | Macro-F1 primary | Accuracy primary | Weighted-F1 primary | Macro-F1 secondary |',
      '| --- | ---: | ---: | ---: | ---: |',
      f'| Исходный H48 с публичными frozen коэффициентами, Mac | {control:.6f} | см. сохранённые прогнозы | см. сохранённые прогнозы | — |',
      f'| [H75: обучение метаслоя внутри задачи](../docs/hypotheses/HYP-075.md) | **{macro:.6f}** | {accuracy:.6f} | {weighted:.6f} | {secondary:.6f} |',
      '', 'Secondary уже был измерен тем же fresh OOF алгоритмом в контрольном запуске; повторного обучения ради новой строки таблицы не было. Он используется как имеющееся доказательство, а не новая независимая проверка.',
      '', '## Насколько надёжно улучшение', '',
      'Для H73: primary RMSE ниже на12.31%, secondary — на26.85%. Это подтверждённое локальное улучшение; многократное снижение ошибки и выигрыш на скрытых задачах не доказаны. Цель15% primary не достигнута и не переименована в выполненную.',
      '', 'Парный bootstrap по целым физическим профилям,500 повторов: описательный95% интервал относительного снижения RMSE primary9.51–15.07%, secondary22.29–31.65%. Интервалы не учитывают отбор гипотез. Эти публичные данные и разбиения уже многократно изучались; это не независимый private-тест.',
      '', '## Проверка поставки', '',
      '- Исходный requirements.txt совпал побайтно; все12 исходных файлов организаторов остались неизменными.',
      '- 271 unit-тест: причинность, отсутствие собственного/будущего y в истории, группы, кодирование, неизвестные категории/поля, формат, ограничения и история.',
      '- Независимый пересчёт10 регрессионных результатов и классификационных прогнозов; отдельная сверка H75.',
      '- Чистый ZIP: CRC, whitelist17имён, нет папок/дубликатов/конфигов/данных/моделей/секретов; `.env` исключён из Git и ZIP.',
      '- Точная симуляция платформы: только root .py + requirements копируются в каталог задачи, запуск `python3 solution.py task.md`, история пустая, агент сам выполняет validation и full refit.',
      '- На public получены корректные22449/41166строк submission.csv рядом с task.md; ID/порядок/столбцы/значения проверены.',
      '- 5 локальных mock-сценариев проверяют неизвестные схемы, ошибки/пустые ответы API и сохранение результата; это не обращения к реальной модели.',
      '- По прямому указанию пользователя реальная LLM не проверялась; на платформе OPENAI_BASE_URL/API_KEY/MODEL задаёт организатор. Автоматической отправки на платформу не было.',
      '', '| Задача, из финального ZIP с нуля | Время, с | Пиковая сумма RSS процессов, ГиБ | Итог |',
      '| --- | ---: | ---: | --- |']
    for record in delivery['records']:
        lines.append(f'| {record["task"]} | {record["seconds"]:.1f} | {format(record["peak_process_tree_rss_bytes"]/1024**3,".2f") if record['peak_process_tree_rss_bytes'] is not None else "не сохранён"} | {record["rows"]}строк, passed |')
    lines += ['', 'Локальный benchmark запускается с одним заранее выбранным применимым кандидатом, без LLM, чтобы отдельно проверить перенос кода в ZIP. Стандартный агент имеет ограниченный портфель/LLM цикл и общий лимит10200с с запасом до3часов. Полная длительность стандартного LLM-цикла не измерялась.',
      '', '## Как коллегам работать дальше', '',
      '1. Работать в своей разрешённой ветке. Эта работа и архив находятся в `sergey_branch`; `ivan_branch` не изменена.',
      '2. Начать с этой таблицы и [единого журнала](../HYPOTHESES.md). H70–H75 уже проверены; не повторять их под новыми именами или seed.',
      '3. Сохранённые прогнозы `.npz`, splits, metrics.json и audit.json рядом с каждым results/.../HYP-* позволяют пересчитать метрики без обучения. Крупные fitted joblib остаются локально и в Git не отправляются.',
      '4. До нового fit зарегистрировать отдельную гипотезу: механизм, отличие от проверенного, параметры, данные/split, критерий и лимит. После любого исхода сразу добавить отдельный report и строку в журнал.',
      '5. Следующие полезные направления: отдельный регрессор для профилей без истории; проверка на новых задачах/заранее отложенном протоколе; содержательная диагностика ошибок для LLM и проверка организаторского API после получения адреса. Это предложения, а не новые полученные результаты.',
      '', 'Не выбирать веса или параметры по уже просмотренному outer и затем не называть этот же outer независимым подтверждением. Не менять ZIP после проверки без новой проверки SHA и состава.',
      '', '## Навигация по доказательствам', '',
      '- [Спецификация](../docs/specs/SPEC-011-sergey-panel-research.md), [план](../docs/plans/PLAN-011-sergey-panel-research.md).',
      '- [Пересчёт метрик и интервалы](../results/2026-10-24/sergey-improvement-v1/independent-metric-verification.json).',
      '- [Манифест финального ZIP](../results/2026-10-24/sergey-improvement-v1/package-manifest.json).',
      '- [Запуски из чистого каталога](../results/2026-10-24/sergey-improvement-v1/clean-public-delivery/verification.json).',
      '- [Все файлы этой серии](../results/2026-10-24/sergey-improvement-v1/).',
      '', 'Исторические артефакты со словами cached-state-technical сохраняют промежуточную техническую проверку прежней схемы; не использовать их как доказательства финального ZIP.']
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    (reports/'TEAM_REPORT_2026-10-24.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Team Markdown report built from verified artifacts')


if __name__=='__main__':main()
