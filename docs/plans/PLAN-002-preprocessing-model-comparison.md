# PLAN-002: поэтапные ML-эксперименты

**Цель:** выполнить SPEC-002 на обоих датасетах и дать вывод по измеренным метрикам.
**Архитектура:** общий train-only sklearn-препроцессор, единый runner с inner/outer оценкой, отдельные папки результатов запусков и общий отчёт.
**Стек:** Python3.12, numpy1.26.4, pandas2.2.3, scipy1.12.0, sklearn1.5.2, catboost1.2.8, matplotlib3.9.4.
**Спецификация:** [SPEC-002](../specs/SPEC-002-preprocessing-model-comparison.md).
**Метод выполнения:** субагенты по прямому поручению пользователя; root интегрирует runner и результаты.

## Шаги

- [x] Подготовить изолированное окружение .venv-ml и проверить состав зависимостей.
- [x] Субагент data_inventory: импортировать исходные data/regression и data/multiclass без изменения байтов; создать профиль и проверить дубликаты.
- [x] Субагент preprocessing: tests/test_preprocessing.py RED→GREEN и experiments/preprocessing.py; fit/transform без утечек, audit, конечный float32 output.
- [x] Root: tests/test_runner.py RED→GREEN и experiments/run_suite.py; разбиения, метрики, сохранение уникальных run directories, замораживание конфигураций до outer.
- [x] Записать гипотезы HYP-001…HYP-007 до экспериментов; HYP-008 дополнительно записан до отдельного engineering-only контроля после аудита удалённых новых полей.
- [x] Прогнать полный unittest suite; запустить два независимых task runners с лимитом два потока каждый.
- [x] Проверить inner ablation, замороженные outer сравнения, native L2 и отдельные linear L1/L2 grid.
- [x] Сохранить финальные train-fit pipelines, transformed arrays и submission; проверить порядок id, форму, конечность и классы.
- [x] Субагент method_review: независимая проверка кода и статистических выводов; root устраняет существенные замечания.
- [x] Создать общий отчёт по пунктам, графики, таблицы и обновить статусы документов.

## Проверки

- `python -m unittest discover -s tests -v`: fit isolation, unknown categories, constants/NaN/inf, split leakage, grouped duplicates, metric direction, output validation and refusing run overwrite.
- `python -m experiments.run_suite --task regression --threads 2` и аналогично multiclass; все опции и seed сохраняются.
- Итоговый сверочный скрипт заново рассчитывает метрики из сохранённых validation predictions, сверяет freeze manifest, run hashes и финальные CSV.

## Риски и решения

Id не используется ни как feature, ни для переноса меток train→test; exact duplicate groups проверяются отдельно по признакам.
Gaussian и scaling оцениваются, но не объявляются полезными без сравнения.
Параллельно запущено не более двух обучающих процессов; артефакты каждого task пишутся в независимые пути, общий реестр обновляет root после завершения.
Окружение проекта создаётся отдельно: исходная .venv и bundled runtime не изменяются.

Дополнительный HYP-008 сохраняет новые признаки без pruning и не меняет основную frozen selection. Повторное использование splits явно маркируется как разведочный анализ.
Техническая ошибка supplemental wrapper после двух успешных regression inner fits сохранена вместе с исходным source snapshot; исправленное продолжение переиспользовало эти результаты без нового fit, с отдельными continuation hashes.
