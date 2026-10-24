# Сохранённое состояние — 24 октября 2026

Текущая и единственная разрешённая ветка: **another_solution**. Не переходить и не вносить изменения в другие ветки без нового прямого поручения пользователя.

## Что завершено

1. Архив `solution-senior-agent-2 (1) (1).zip` импортирован побайтно; исходный коммит `568fc9879e677371b75fb879d49011cc6c2377f6`. Оригинальный ZIP не менялся.
2. Исходный агент проверен на двух задачах с300-секундным лимитом/3итерациями/4CPU безLLM. Все10 фактически выполненных обучающих запусков сохранены.
3. Проверены ровно3 небольшие гипотезы на тех же OOF, без новых tree fits. AS002 принята; AS001 не внедрена из-за небольшого ухудшения общего Macro-F1; AS003 без прироста. **Не повторять**.
4. Финальные метрики: RMSE OOF29.34207680247969 (контроль29.373823495698232), B28.64439520546687 (контроль28.685722725959646). Macro-F1 OOF.6534905279816051/B.6481770971418532 сохранены.
5. Поставка: dist/solution_another_solution_v1.zip; 136784байта,30rootфайлов,0папок; SHA256efa97f675a8c80409eedccacbd19d0a1c0d44100180ccabbd974aedf2da64d31. 28/30файлов исходные; измененыag_finalize.py иag_config.py.
6. 38 независимых проверок метрик,6тестов,2чистых CLI из самогоZIP. Формат submission.csv совпал с sample. Чистые CLI имели120s/0researchiterations: это техническая проверка, не3часовойDockerbenchmark. Реальный API и платформа не проверялись/не вызывались.

## Где данные для продолжения

- reports/ANOTHER_SOLUTION_RESULTS_2026-10-24.md — полный командный отчёт и таблицы.
- results/another_solution/2026-10-24/original-{regression,multiclass}/replay.joblib — только кэш результатов/OOF/метаданные, без ключей и fitted моделей. Позволяет продолжать анализ без переобучения исходника.
- Там же exp-*-predictions.npz, splits.npz, metrics.json, command.log и task_spec.json.
- AS001/AS002/AS003-{task}/ — каждый исход, прогнозы, evidence и report.
- experiments/another_solution/ — harness, три исследованных варианта, точный оригинал финализации, независимый verifier, package и clean delivery scripts.
- HYPOTHESES.md и docs/ANOTHER_SOLUTION_PLAN.md — пререгистрация/реестр.

Новые модели/гипотезы не запускаются автоматически: последнее поручение ограничивалось2–3 точечными вариантами, все3 проверены. Следующий шаг после загрузки пользователем — записать платформенный результат и SHA отправленного ZIP; при новом поручении продолжить только небольшие изменения, не переписывать сильную исходную архитектуру.

Предыдущая серия SPEC012 сохранена отдельно: reports/PREVIOUS_SERIES_CHECKPOINT_2026-10-24.md и локальный outputs/spec012-checkpoint-2026-10-24 в Codex workspace. Её метрики другого агента/протокола, не контроль этого ZIP.

## Новая серия: полный агентский цикл с Qwen

Новое поручение пользователя заменяет прежний запрет live API и ограничение AS001–003: улучшать полный агентский цикл, сравнивать с реальной Qwen и публиковать подтверждённые улучшения в another_solution. Не повторять AS001–003.

База — origin/another_solution 3567ad073b1ab3fd5bb1aec3423d264b28ee8a02. Изолированная рабочая копия: /private/tmp/alfa-another-work, ветка another_solution. Основная /Users/maxchibisov/Projects/HACKS/alfa-hack остаётся maxim/autoresearch с пользовательскими изменениями dist ZIP/manifest, завершённым e2e.json и qwen-hypothesis.json; ничего там не переключать/перезаписывать. Прежние процессы ML завершены. Исторические offline OOF/B этой ветки не равны внешним masked метрикам Maxim и не доказывают пользу Qwen; происхождение роста места69→26 неизвестно.

API: подтверждённый https://changellenge.up.railway.app/v1, qwen/qwen3.8-27b из конфигурации ветки/прежнего authenticated models discovery и Qwen3.8-27B в официальной PDF. Ключ только в .env.local основной копии, не копировать в source/task/ZIP/Git; локальный launcher получает его в окружении. Платформенный агент — только среда Boosters.

Сначала frozen baseline ZIP efa97f… (источник dist/solution_another_solution_v1.zip из3567ad0), затем парные masked запуски обеих задач с реальной Qwen. Первичный конечный бюджет каждого run:600s/8CPU/60research iterations/TUNE_TRIALS0/20000LLMtokens. Измеритель сохраняет внешние predictions, метрики, SHA, эксперименты и usage/этапы LLM, не меняя решений исходного агента. Все train/test/OOF/checkpoints — внеGit в /private/tmp/alfa-another-live. Документы/сводки — results/agent-loop/ и общий HYPOTHESES.md. Не запускать fit параллельно; не перезапускать completed или живые handles. Следующий шаг: контроль, затем конкретные изменения по измеренным проблемам, повтор с теми же условиями; не считать большее число Qwen вызовов улучшением само по себе.

Пилот с6итерациями завершилreg за175.7s (externalRMSE39.044473), но Qwenplanning0: corequeue длиннее. Это ошибочный protocol pilot, не primarycomparison. Сохранёнpilot-six-iterations.json и всеOOF. Егоclassphase прерван SIGINT толькоу проверенныхownedPID; пользовательскиепроцессыне трогались. Новыйcontrol-live60 — тотжеZIPefa97f сштатными60iterations/600sec; параcandidateс темижеусловиями. Повторbaselineобусловлен исправлениемлимита, не повторнойпроверкойстарыхASгипотез.

### Live checkpoint
Control-live60 completed: regression externalRMSE44.0296239669,277.97s,1356.8MB,3LLM calls/28688tokens (overshoot20k), accepted one LLM log1p, another failed raw-column contract. Classification externalMacroF1.63971148823,451.51s,zeroLLM; fallback queue exhausted time. These masked scores differ from old OOF29.34/.65349. Full receipts/results/OOF in /private/tmp/alfa-another-live/control-live60; summary results/agent-loop/control-live60.json. No live model processes remain from baseline.
Uncommitted AL1–3 modifications: ag_llm_client/io_contract, compact profiler/memory/planner/prompts, feature-code ordering in features/pipeline; benchmark harness experiments/agent_loop. Next: narrow tests, flat candidate ZIP, identical sequential paired live runs. No candidate trained yet. Frozen baseline /private/tmp/alfa-another-live/control.zip must remain.

AL1–3 completed and REJECTED as final delivery: regression44.0296239669 unchanged, classification.6004257559 versus control.6397114882. Qwen participated class(2requests) but displaced useful fallback; savings alone did not improve quality. Frozen ZIP017b3f… and receipts preserved.
AL1–6 candidate ea969e853318188ba34a8d23317fdbcaee7bf69bdd84b02c304e67f15a5af15e now running sequentially under same600s/task protocol; task dirs /private/tmp/alfa-another-live/candidate-al123456. AL4 exact matrix no-op skip, AL5 latest-temporal-fold selection, AL6 team raw-X duplicate grouped validation gated by feature-only test overlap.16targeted tests pass; baseline exp15 exact code preprocessing now passes full35000/18308rows (no newfit). Do not restart existing run; next inspect results/agent-loop/candidate-al123456.json, session handle in chat. No confirmed prediction gain yet, no new commit/push made.
Latest fetched origin/sergey_branch0efccdb intermediate scientific series; Ivanfe674da unchanged. Research report reports/AGENT_LOOP_2026-10-24.md; no external code copied.

Confirmed regression checkpoint: AL1–6 externalRMSE38.797193598402 versus44.029623966865, same outer/caps; actual495.77s versus277.97s. ZIPdist/solution_another_agent_loop_v2.zip SHAea969e…; measured source unchanged. Classification still running; not final readiness. Publishing checkpoint per user; no further experiments beyond this series planned before final technical verification.
