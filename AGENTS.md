# Alfa Hack AutoResearch — рабочая память

Язык общения — русский. Цель: универсальный исследовательский ML-агент и проверенный ZIP для Boosters. Задание пользователя хранится локально в `/Users/maxchibisov/.codex/attachments/c9e0282d-79d7-4115-9835-088a6257b821/pasted-text-1.txt`.

## Текущий этап

2026-10-23: ветки изучены; плоский AutoResearch-агент реализован. Ветка `codex/ultimate-autoresearch-20261023` создана от `origin/main` (`f5d7d87`) в отдельном worktree. Исходный checkout и пользовательские файлы не изменять. Источники: `ivan_branch` (`a6a1b94`) и `sergey_branch` (`d9022e2`).

Пройдены 45 unit/integration/failure tests (`results/unit-v2.xml`) и 9 clean-room сценариев (`results/cleanroom.json`, первоначальный архив). Публичное сравнение baseline/adaptive v1 выполняется в `/private/tmp/alfa-public-v1`; не перезапускать живой процесс без проверки. Его snapshot старее текущих ограниченных defaults ET/RF и audit без выбора по outer. Финальные метрики и повтор clean-room на финальном архиве ещё требуются.

## Обязательный контракт

- Python 3.12, 8 CPU, 32 GB RAM, до 3 часов на задачу; без внешнего интернета и установки пакетов.
- ZIP до 100 MB: только корневые `.py` и неизменённый официальный `requirements.txt`.
- Запуск `python3 solution.py task.md`; `submission.csv` рядом с `task.md`, атомарная запись и ранний fallback.
- LLM: только предоставленные организатором `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`; значения не придумывать и не публиковать.
- Никаких данных, моделей, ключей в Git/ZIP. Приватные задачи неизвестны; публичные имена не хардкодить.
- Разрешена публикация собственной ветки, без merge. Отправка на Boosters требует отдельного разрешения.

## Продолжение

Читайте `docs/RESEARCH_STATE.md`, `docs/HYPOTHESES.md`, `docs/EXPERIMENTS.md`, затем актуальный код. Проверяйте новизну эксперимента, фиксируйте split, seed, метрику, wall time и ограничения. Не называйте локальную метрику результатом Boosters.

## Архитектура и команды

`solution.py` → `automl_engine.py`: ранний атомарный fallback, nested search и фиксированный парный audit; outer не выбирает модель. `automl_data.py`: task/schema/metrics/submission; `automl_models.py`: модели и DSL → исполняемый Python; `automl_worker.py`: изолированный fit; `automl_runtime.py`: deadline/process/memory; `automl_llm.py`: bounded platform API. История прошлого публичного исследования не встраивается в платформенный код.

Локальный Python 3.12: `/private/tmp/alfa-research-venv/bin/python`. Для macOS LightGBM/XGBoost: `DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/alfa-openmp/libomp/23.1.4/lib`; это dev runtime, в ZIP не входит. Docker daemon локально недоступен. Реальные API credentials отсутствуют; используются HTTP mocks, не настоящий Qwen endpoint.

- Тесты: `python -m pytest -p no:cacheprovider -q tests` (localhost mocks требуют разрешённого bind).
- Сборка: `python experiments/package_solution.py`.
- Чистая проверка: `python experiments/verify_package.py`.
- Публичные прогоны: `python experiments/run_public.py --data-root PATH --work-root NEW_TMP_PATH`.

## Следующие действия

Дождаться v1, проверить метрики и стоимость; сравнить с текущими bounded defaults/scheduler, провести финальные публичные и clean-room проверки, обновить документы и журнал новых гипотез, secret scan, коммиты и push собственной ветки. Исторический лучший regression — ExtraTrees log1p RMSE42.196700; classification XGBoost MacroF1 .591451371 (outer уже использован ранее, не hidden score). Новые результаты не выдумывать. Не переносить обязательный full-v1: он ухудшил обе метрики.
