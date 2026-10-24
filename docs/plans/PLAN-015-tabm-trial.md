# TabM-mini trial implementation plan

Execution: native, текущая сессия; разрешённый пользователем исследовательский spike. Spec: [SPEC015](../specs/SPEC-015-tabm-trial.md). Работа в текущей папке с существующим WIP, только новые task-local файлы и append журналов; новая git-ветка/worktree не создаётся.

Goal: оценить реальную пользу новой модели/смеси без изменения поставки.
Architecture: NumPy общий MLP + входные адаптеры + независимые головы, sklearn fit-only preprocessing; отдельный bounded runner с cached controls.
Stack: существующие NumPy1.26.4, sklearn1.5.2, pandas2.2.3, scipy; unittest.

## Constraints and review focus

- ≤900сек/H108, ≤240сек/fit, 60epochs; без повторов предыдущих моделей.
- Unknown categories, NaN/Inf, новые/переставленные колонки: предсказания конечны, схема сохраняется.
- Несмежные строковые labels: сохранение classes/порядка вероятностей.
- Градиенты всех ветвей и раздельных голов: независимая finite-difference проверка.
- Отсутствие обучения на query/outer: fit-only статистики, frozen blend weight, training-only loss.
- Ошибки/timeout: отдельный отчёт и terminal запись; численные метрики только при валидных прогнозах.

## Task 1 — estimator (REQ001–003)

- [x] Создать tests/test_tabm_trial.py: gradient finite differences, обучение синтетических reg/class, сохранение labels, unseen/missing features, batch invariance и pickle.
- [x] Запустить unittest, подтвердить отсутствие реализации.
- [x] Создать experimental_tabm.py: MiniEnsemble.loss_grad, TabMMini.fit/predict/predict_proba, configuration/audit.
- [x] Пройти тесты и проверить training-loss improvement.

## Task 2 — bounded comparison (REQ004–005)

- [x] Зарегистрировать H108 и до запуска сохранить карточку/конфиг.
- [x] Создать experiments/tabm_trial.py: read-only controls, проверки SHA/positions/classes, before-fit Memory.claim, per-run source snapshots и reports.
- [x] Выполнить regression-inner и classification-inner последовательно; завершать журналы сразу после каждого.
- [x] Только прошедшие inner-gate части допустить к outer с замороженным весом.

## Task 3 — evidence (REQ006)

- [x] Выполнить unittest discover -s tests и сохранить полный лог: 574 tests, 569 pass/5 existing report-publication errors WinError5; изолированно воспроизведены.
- [x] Независимо пересчитать RMSE/F1, проверить split-disjointness, fingerprints и frozen weights.
- [x] Сохранить итоговый report.md, результаты и ограничения; solution/baseline/ZIP не менять.
