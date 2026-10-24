# Alfa Hack AutoResearch — рабочая память

Общение по-русски. Изменения минимальны; без попутного рефакторинга, субагентов и Superpowers. Не трогать main, ivan_branch, sergey_branch; без reset/clean/force push, секретов и данных в Git/ZIP.

## Состояние на 24.10.2026

Ветка `codex/research-optimized`, база `eab478a`. Импортированы незакоммиченные наработки `/Users/maxchibisov/Projects/HACKS/alfa-ultimate`, public-v1…v5 и тесты. Оба прежних worktree оставлены без изменений. Инвентаризация: `docs/sources/research-optimized-intake.json`. Исторический журнал сохранён; текущая серия и ограничения — **`docs/EXPERIMENT_LOG.md`**.

14 исследовательских fit, 631.22 секунды; серия остановлена. Повторять `numeric_views_study.py`/`numeric_control_audit.py` нельзя: использовать готовые результаты.

## Архитектура и контракт

`solution.py` → `automl_engine.py`: ранний атомарный fallback, nested search/rolling windows, audit без выбора по outer. `automl_data.py` — схема/split/метрики/submission; `automl_models.py` — модели, безопасный DSL, дополнительные numeric views; `automl_worker.py` — изолированный fit и probability cache; `automl_runtime.py` — deadline/process/memory; `automl_llm.py` — bounded platform API и offline fallback.

Python3.12, до8CPU/32GB/3ч. `python3 solution.py task.md`; submission рядом с task. ZIP ≤100MB: только корневые `.py` и неизменённый официальный requirements. Никаких обязательных вложенных модулей/публичных имён полей/frozen pretrained state/сторонних LLM API. LLM только из OPENAI_API_KEY/BASE_URL/MODEL платформы; `.env.local` автоматически не загружается.

## Сопоставимые метрики и проверенные гипотезы

- Regression outer `<5 → 5`, 35000/18308, key68ef0344…: current ET log-target **42.282936 → 42.012317 RMSE** с raw+signed-log (−0.640%). Это результат кандидата; финальная search policy заново не измерена. Лидер Sergey H73 **34.520269**, не превзойдён.
- Classification group outer, 70529/17141, key239200bc…: прежний AutoResearch **0.637337946**, лидер Ivan H48 **0.649531162**. Новые варианты отсеяны по inner; новых outer fit нет.
- NV1 ranks, NV2 signed-log, NV3 robust projections, NV4 ranks+projections проверены на обеих задачах; только NV2 продвинут в условную ET/log-target/RMSE очередь. JSON: `results/numeric-views-v1.json`, `results/numeric-control-audit.json`.
- Inner regression: 0.65×period4+0.35×period3; classification nested exact-feature-group split. Полные ключи/хеши в JSON. Разные split не смешивать; outer публичный и исторически исследованный.
- Не повторять full-v1, smearing/threshold/focal/OOF stacking, H52/H62 или causal panel H70…H74 под новым названием: сначала читать сохранённые результаты/историю.

## Проверки и команды

67 tests passed (`results/unit-research-optimized.xml`); 9 clean-room сценариев passed (`results/cleanroom-research-optimized.json`). Linux/официальный контейнер не проверен. `.env.local` имеет только ключ; BASE_URL/MODEL отсутствуют, live API не вызывался (`results/local-api-check.json`).

Локально Python: `/private/tmp/alfa-research-venv/bin/python`; для native boosting macOS: `DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/alfa-openmp/libomp/23.1.4/lib`. Пакеты ML соответствуют официальным версиям (`results/research-environment.json`).

- `python -m pytest -p no:cacheprovider -q tests` (HTTP mocks требуют localhost bind).
- `python experiments/package_solution.py` → `dist/alfa_autoresearch_submission.zip` + manifest.
- `python experiments/verify_package.py --output results/cleanroom-research-optimized.json`.

Далее: независимая проверка signed-log и frozen policy; универсальный поиск causal panel keys с проверкой стабильности; shrinkage-калибровка rare-class на disjoint OOF. Сначала проверить новизну и кэш, затем ограничить бюджет. Не выдавать candidate score за agent/Boosters score.
