# Журнал исследовательской итерации — 24.10.2026

Ветка: `codex/research-optimized`. Новых исследовательских fit: **14**, суммарно **631.22 с (10 мин 31 с)**, 4 CPU на fit. После этой серии поиск остановлен. Это публичные validation-метрики, не Boosters score.

## Сохранение и база

Исходный checkout был `main@f5d7d87` с пользовательскими untracked baseline, requirements, PDF и публичными CSV. Создана собственная ветка от `eab478a`; main/ivan_branch/sergey_branch не изменялись. Оба прежних worktree и их незакоммиченные файлы оставлены на месте. Из `/Users/maxchibisov/Projects/HACKS/alfa-ultimate` перенесены рабочие модули, тесты, документы и public-v1…v5; уникальный rolling-study из другого worktree сохранён отдельно. Исходные `.gitignore` и requirements также оставлены в ignored `.intake` и `/private/tmp/alfa-intake-original`. [Инвентаризация](sources/research-optimized-intake.json).

## Существующие сопоставимые результаты

| Решение | Regression RMSE, outer 5 | Classification Macro F1, group outer |
| --- | ---: | ---: |
| Ivan H12 / H48 | 42.196700 | **0.649531162** |
| Sergey H73 / H75 fresh OOF | **34.520269** | 0.644722567 |
| Предыдущий AutoResearch public-v5, выбран по inner | 43.700590 | 0.637337946 |
| Текущий paired ET-контроль / сохранённый XGB-контроль | 42.282936 | 0.637337946 |

Regression: SHA train `096dac…`, `Year_Factor<5 → 5`, 35000/18308, split key `68ef0344…`. Classification: SHA train `171a89…`, первый StratifiedGroupKFold(5, shuffle=True, seed=42) по exact raw feature groups, 70529/17141, key `239200bc…`. Полные хеши/ключи: [Ivan](sources/ivan-latest-protocol.json), [Sergey](sources/sergey-latest-protocol.json), [v5](../results/public-v5/report.json). H75 взят из актуального HYPOTHESES.md Сергея; в таблице не используется для доказательства эффекта NV2. Старый random-stratified F1 Сергея 0.628620730 не смешивается с group split.

Не повторялись: mandatory full-v1 (ухудшение RMSE 43.837202 / F1 0.563787167), Tweedie, linear leaves, OOF smearing, focal/threshold механизмы, nearest-neighbor H52 и FM H62, causal panel H70…H74 и OOF stacking H48/H75. См. [историю](HYPOTHESES.md) и [старый журнал](EXPERIMENTS.md).

## Четыре новые гипотезы

Все варианты сохраняют исходные признаки; это ограниченные дополнительные представления, а не повтор обязательной Gaussian-нормализации. Rank/projection статистики обучаются только на fit; максимум 128 числовых полей, 12 проекций, фиксированный seed 1729. Raw+signed log меняет распределение случайных порогов ET, не отбрасывая абсолютную шкалу. Проекции проверяют косые границы после train-only median/IQR и clipping ±10. NV4 — заранее зафиксированная комбинация NV1/NV3.

Regression inner: окна `<4 → 4` и `<3 → 3`, веса 0.65/0.35, ключи `4b4a5483…` / `52607b5d…`; контроль ET200/depth24/leaf2/features0.8/log-target **46.971711**. Classification inner: вложенный group split 56707/13822, key `64e73424…`; контроль XGB600/depth6 + category frequencies + prior decision power0.3 **0.636708318**. Хеши данных и все индексы split сверены с public-v5; контроли inner переиспользованы без fit.

| Гипотеза | Inner RMSE ↓ | Inner Macro F1 ↑ | Решение |
| --- | ---: | ---: | --- |
| NV1 — empirical CDF + raw | 46.948047 | 0.610423228 | Не продвигать |
| NV2 — signed log + raw | 46.830086 | 0.615014811 | Кандидат улучшен на outer |
| NV3 — bounded robust random projections + raw | 47.061980 | 0.631712482 | Не продвигать |
| NV4 — ranks + projections + raw | 47.266679 | 0.633885594 | Не продвигать |

Выбор сохранён в JSON **до** outer: regression NV2, classification прежний контроль. Outer новых классификаторов не оценивался. Regression NV2: **42.012317102**; один необходимый paired fit текущего контроля: **42.282935868**, выигрыш **0.640%**. H12 использовал другую реализацию encoding/scaling, поэтому 42.196700 не выдаётся за парный контроль новой предобработки. [Все fit и split](../results/numeric-views-v1.json), [парный контроль](../results/numeric-control-audit.json), команды: `python experiments/numeric_views_study.py`, затем `python experiments/numeric_control_audit.py` (оба запрещают повтор при наличии результата).

## Что интегрировано и чего это не доказывает

В существующие canonical candidate, CandidateModel, offline scheduler и LLM schema добавлено дополнительное signed-log представление. Offline предлагает его только для RMSE, если текущий лидер — ET с log-target. Все решения сохраняют per-task inner selection, дедупликацию, process deadlines, ранний fallback и отдельный audit. Rank/projection режимы сохранены для воспроизводимости исследования, но не добавлены в offline/LLM очередь. Классификационный поиск сохранён.

Выигрыш доказан только для кандидата на одном исторически открытом outer; доверительный интервал/повторный seed не измерены. Итоговая политика AutoResearch после добавления кандидата заново на публичных данных не прогонялась: нельзя приписывать ей RMSE42.0123, поскольку inner может выбрать другой кандидат. Лидеры команды 34.5203/0.64953 не превзойдены. Отличие от веток команды: сохранённый nested controller + новые ограниченные числовые представления и условное предложение, без публичных имён колонок и frozen pretrained/meta-state. Не заявляется новая с нуля архитектура.

## Проверки, API и ограничения

Финальный suite: **67 passed**, включая train-only transforms, cache keys, schema, scheduler, API failure/timeout mocks, deadline/fallback. Первый запуск: 62 passed/4 sandbox bind failures; повтор модуля 7/7, затем финальный suite 67/67. Отчёты `results/unit-research-*.xml`. Clean-room ZIP — `results/cleanroom-research-optimized.json`, 9 сценариев Python3.12; точные версии — `results/research-environment.json`. Linux/официальный контейнер недоступен, hidden score неизвестен.

`.env.local` ignored, загружен только в окружение локального проверочного процесса. Ключ не печатался. В original baseline нет значений endpoint/model ID: используются OPENAI_* environment. Локально отсутствуют OPENAI_BASE_URL/OPENAI_MODEL; live API не вызывался и параметры не придуманы. [Статус](../results/local-api-check.json). Платформенный код не читает `.env.local`, не использует сторонний API, не требует dotenv/mcp; официальный requirements неизменён.

Далее: (1) независимая проверка signed-log ET и замороженной search policy на новом временном окне без выбора по старому outer; (2) универсальное выделение повторных объектов для causal panel через стабильность ключа, сравнение с H70…H74 вместо повторения их fit; (3) ортогональная H48/H75 проверка shrinkage-калибровки редких классов на disjoint OOF с фиксированным бюджетом и групповым split.
