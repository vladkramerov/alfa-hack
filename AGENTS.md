# Alfa Hack AutoResearch — рабочая память

Общение по-русски. Минимальные связные изменения; без субагентов и Superpowers. Не трогать main/ivan_branch/sergey_branch, не применять reset/clean/force push. Секреты, данные и обученные модели не включать в Git/ZIP.

## Текущее состояние — 24.10.2026

Ветка `maxim/autoresearch`; старая remote `codex/research-optimized` сохранена. Checkpoints: `be9d9ae` (предыдущая серия), `384a5a7` (диагноз/механизмы), `4239100` (deployment selection). Оба прежних worktree сохранены. История — `docs/EXPERIMENT_LOG.md`; текущие машинные результаты — `results/score-recovery/`.

**ERRATUM:** прежние regression inner окна были локальными позициями внутри outer_fit, но применялись к исходному train. В search-fit попадали3556/3041 строк outer года5. Старые заявления о независимом audit неверны; численные результаты оставлены как exploratory. Исправленные окна/доказательство: `temporal-index-diagnosis.json`. Старую numeric-view серию не повторять.

Score13.7676 (118/186) и ориентир26.6404 сообщены пользователем. Версия ZIP неизвестна — пользователь подтвердил. Инвентаризация архивов/SHA: `submission-provenance.json`; ни одному архиву score не приписан.

## Архитектура

`solution.py` → `automl_engine.py`: ранний атомарный fallback, сильный старт до LLM, default36 regression/16 classification (hard64) trials, до4 планов, дедупликация и резерв refit. Regression selection использует последние известные train-периоды; search score не independent audit. Classification сохраняет проверенный nested/legacy маршрут; новый boosting/geometric default отвергнут после ухудшения policy. Честная оценка — внешний masked test. `AUTOML_VALIDATION_POLICY=nested` сохраняет исправленный nested protocol.

`automl_data.py` — схема/метрика/split/submission; `automl_models.py` — модели, безопасный DSL, grouped train-only stopping с полным refit, арифметические/геометрические blends; `automl_panel.py` — поиск entity key без test/target и strictly-past history; `automl_worker.py` — изолированный fit и выровненные probability caches; `automl_runtime.py` — deadline/memory/process; `automl_llm.py` — platform API, JSON wrappers, bounded reasoning compatibility fallback.

Контракт: Python3.12,8CPU,32GB,до3ч; `python3 solution.py task.md`, CSV рядом с task. ZIP ≤100MB: корневые `.py` и неизменённый официальный requirements. Без embedded public meta-state, известных имён датасетов/полей, вложенных модулей и сторонних LLM API.

## Измерения

- Финальная regression policy на masked `<5→5`,35000/18308,key68ef0344…: старый ZIP **44.810781 → новый37.578330 RMSE (−16.14%)**. Selected:80/20 auto-history ET(log+signed-log)/LGB. Подтверждено фактическим итоговым CSV, не только отдельным fit.
- Промежуточная новая nested-policy дала41.599136; не стала финальной. Generic history ET ablation:48.230237→40.499557 на исправленных period4/3; outer36.129126. Старое окно без истории не ухудшилось: cold fallback.
- Classification ablation, nested group key64e73424…: XGB control inner.636708318; LGB+stopping.640445243; geometric.644824653. Frozen outer key239200bc…: .637337946→.644093520. Native CatBoost recipe .556644350 отвергнут. End-to-end новый default оказался хуже: .644093520 против .649412479. Откат подтверждён побайтно одинаковым CSV .649412479; новые class proposals/LLM schema opt-in, не default.
- Team best на тех же outer: Sergey H73 RMSE34.520269, Ivan H48 MacroF1.649531162; не объявлять эти значения Boosters score. Разные splits не смешивать.
- Пять направлений ограничены текущей серией: auto history; native boosting/stopping; cached geometric pooling; сильный старт/распределение бюджета; deployment-aligned selection. Mapping/bagging/checkpoint исправлены отдельно как ошибки. Не начинать новую серию до итогов текущего E2E.

## Проверки и продолжение

77 tests passed (`unit-final.xml`),9 clean-room passed (`cleanroom-final.json`). Итоговый E2E завершён: `e2e-final.json`; контролей повторно не fit. Исходные задачи: reg22449/class41166 строк, refit completed;1461.81/263.26с,peak1.19/1.17GB. Regression arms переиспользованы после classifier-only rollback с точным replay15/20 recipes и byte-identical estimator/data/runtime: `regression-source-equivalence.json`; их исходный ZIP SHA сохранён. `experiments/score_recovery_e2e.py --resume` проверяет ZIP/source, пропускает готовые arms и подключается к сохранённому process-state. Артефакты/tmp сохранять; predictions не коммитить. Linux/официальный Docker и hidden score не проверены.

`.env.local` ignored и автоматически агентом не читается. Есть ключ, нет OPENAI_BASE_URL/OPENAI_MODEL; live calls0. Не брать чужие значения из публичных репозиториев. Qwen reasoning-disable проверен только mocks; чужие наблюдения не считать проверкой нашего endpoint.

Python: `/private/tmp/alfa-research-venv/bin/python`; macOS native libs: `DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/alfa-openmp/libomp/23.1.4/lib`.

- `python -m pytest -p no:cacheprovider -q tests` (localhost mocks требуют bind).
- `python experiments/package_solution.py` — ZIP/manifest; frozen E2E SHA должен совпасть.
- `python experiments/verify_package.py --output NEW_RESULT_PATH` — чистая распаковка.

Серия завершена и остановлена. ZIP46651 байт, SHA256 e12d691382bc280fae5b2063e2a28ca2d21a18847f1c23f786522dafd464a641. Далее: ручной submit с фиксацией SHA и per-task логов; independent period/data и доверие entity key. Не повторять эту серию и не возвращать отвергнутый class default без новых доказательств. Никаких загрузок на Boosters без отдельного разрешения.
