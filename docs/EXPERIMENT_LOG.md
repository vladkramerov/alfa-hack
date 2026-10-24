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

## Восстановление public score — 24.10.2026

**ERRATUM к предыдущей итерации:** temporal search возвращал позиции внутри outer_fit без отображения в исходный train. В первом search-fit регрессии находились 3556 outer-строк года5, в search-valid499; во втором3041/515. Старые regression inner keys4b4a5483…/52607b5d… и заявления о независимости audit невалидны. Метрики реальных fit сохранены как exploratory. Исправленные keysf4bcbd8b…/b946d200… и составы годов: [диагноз](../results/score-recovery/temporal-index-diagnosis.json). Новый тест перемешивает строки и запрещает search/audit overlap. Классификационный mapping изначально был глобальным.

Score13.7676, место118/186, top10=26.6404 сообщены пользователем; версия загруженного ZIP неизвестна (пользователь подтвердил). [Локальные SHA](../results/score-recovery/submission-provenance.json) ни одному score не приписаны. Нельзя установить виновную задачу или объяснить конкретный платформенный запуск без его SHA/логов.

Доказанные ограничения кода: temporal-index ошибка; резервный CSV от более старого/меньшего окна; subsample LightGBM без subsample_freq; default12 trials при трёхчасовом бюджете; отсутствие автоматической causal history. Исправлены mapping, checkpoint крупнейшего fit и bagging; default36/hard64, сильный старт до LLM, до4 планов. Сбой метрики/submission/refit конкретного исторического запуска не доказан.

Публично изучены [MUCG9](https://github.com/MUCG9/alfa-hack-automl), [hilhool](https://github.com/hilhool/alfa-automl-agent), а также найденные GitHub-поиском [NolReaction](https://github.com/NolReaction/alfa-automl-agent), [ilove4fak](https://github.com/alfahack-ilove4fak/alfahack-automl). [Commit/public visibility/license](../results/score-recovery/external-sources.json): все публичны, явная лицензия не найдена, чужой код не переносился. Связь с указанными leaderboard-командами не подтверждена; целевые web/GitHub запросы не дали такой связи.

Пять применимых идей:
1. Train-only entity discovery и strictly-past history с cold fallback (hilhool/entity.py, NolReaction/history_discovery.py). Наша реализация выбирает ключ без test/target, по recurrence и collision; не фиксирует колонки H70…H74.
2. Metric-aware stopping на внутреннем grouped split и полный refit (MUCG9/model_utils.py, NolReaction/boosting_candidate.py). Внешний audit не выбирает iteration count.
3. Выравненная композиция probability caches (MUCG9/ensemble_cache.py). Наша новая абляция — фиксированное geometric pooling, без OOF coefficients или повторного H37 threshold search; точное равенство обычному refit проверено при0 новых fit.
4. Feedback-first контекст, reasoning-disable и JSON wrappers (MUCG9/agent_runner.py, hilhool/llm.py). Авторы сообщают reasoning-only/empty content и эффект reasoning.enabled=false; наш endpoint не проверен. Флаг запрашивается только для Qwen; один bounded400/422 retry без него. Мысли модели не сохраняются.
5. Обязательный сильный старт и бюджет финального fit (MUCG9 runner, ilove4fak/runtime/budget.py). Существующие deadline/fallback сохранены; сильные рецепты запускаются до LLM.

Локально ключ присутствует, BASE_URL/MODEL отсутствуют;0 платных вызовов. Чужие адрес/ID не подставлялись. [API status](../results/score-recovery/api-config.json).

Абляции:
- Исправленные period4/3, веса.65/.35: старый Cat/expert рецепт RMSE49.357414; ET log48.230237; auto-history LGB42.893677; auto-history ET40.499557. ET period4:44.950082→33.056728; period3:54.321953→54.321953 (ключ не найден, cold fallback). Frozen ET outer36.129126 против текущего ET42.282936; Sergey H73=34.520269 сильнее. [Данные](../results/score-recovery/panel.json):9 model fit,92.39с,peak873.4MB. Новые контроли необходимы: старые inner были несопоставимы.
- Classification key64e73424…: XGB frequency/prior.3 inner.636708318/outer.637337946. Native Cat+stopping inner.556644350 — отвергнут; strong LGB+stopping.640445243; arithmetic XGB/LGB.642861247; geometric.644824653. Frozen geometric outer.644093520; Ivan H48=.649531162 сильнее. [Данные](../results/score-recovery/classification.json):158.06с, включая внутренние stopping probes и outer. Локальные prediction/probability NPZ сохранены по путям в JSON, не включены в Git/ZIP.

75 tests passed;9 clean-room passed: [unit](../results/score-recovery/unit.xml), [package](../results/score-recovery/cleanroom.json). Frozen CLI E2E старой/новой политики на masked outer test и обеих исходных задачах выполняется; итог будет добавлен после завершения. На Boosters ничего не отправлялось.

Проверка финальной политики выявила пятое направление: nested source384a5a7 дал честный shadow RMSE41.599136 против44.810781 старого ZIP (одинаковые35000/18308,8CPU,1200с,cap16). Это лучше контроля, но хуже36.129126 отдельного кандидата: последний известный период был только audit и не участвовал в выборе, поиск смотрел на более старую историю. Поэтому оставшийся устаревший прогон остановлен, все checkpoints сохранены; неизменённый контроль повторно не fit. [Отчёт первой пары](../results/score-recovery/e2e.json).

Проверяется deployment-validation: выбор по последним доступным train-периодам, внешняя оценка по замаскированному test вне агента. Внутренние search-метрики теперь явно не independent audit; nested mode остаётся опцией. Это не использование меток настоящего test. Финальный frozen прогон — `e2e-deployment.json`; он переиспользует старый regression shadow control и отдельно проверяет обе исходные задачи. Нулевой RSS старого sandbox-монитора означает недоступное измерение, не нулевую память; новый запуск допускает чтение RSS.

Завершена честная regression shadow-пара по финальным submission.csv: старый ZIP44.810780971, deployment-policy37.578329562 RMSE (−16.13998%). Split68ef0344…,35000/18308; одинаковые8CPU,1200с,cap16; метки года5 отсутствовали в папках агента. Новая политика выбрала ансамбль80/20 из auto-history ET(log-target+signed-log) и auto-history LGB. Это подтверждает участие истории в платформенном пути, а не только абляции. Предыдущая nested-policy41.599136 сохраняется как промежуточный контроль. [Парный анализ](../results/score-recovery/regression-paired-audit.json): условный row-bootstrap интервал выигрыша13.49…18.94%, без учёта корреляции сущностей и неопределённости выбора/hidden test.

При обновлении инструкций E2E-parent был прерван, но CLI продолжил fit. Добавлен режим --resume: проверяет совпадение frozen source, подключается к существующему PID/готовому summary и пропускает завершённые arms. Завершённые fit не повторялись. RSS resumed regression-arm измерен только после подключения; полные следующие arms наблюдаются целиком. Финальный unit suite76/76, clean-room9/9: unit-deployment.xml, cleanroom-deployment.json.
