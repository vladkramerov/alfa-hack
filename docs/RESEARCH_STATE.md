# Состояние исследования — 24.10.2026

## Точка продолжения

Серия strongest-team завершена. Ветка maxim/autoresearch, контроль73d5aa4; этапы d04ec91/052ebc1/72ab5c6. Последний итоговый commit узнавать git log, не дублировать fit. Все результаты в results/strongest-team, подробный единый журнал docs/EXPERIMENT_LOG.md, понятная таблица results/strongest-team/README.md. Никаких субагентов/Superpowers, reset/clean/force push, изменений других веток и отправок Boosters без поручения.

Сильнейшая подтверждённая наша поставка: dist/alfa_autoresearch_submission.zip, SHA256 f3a24044b90fb638df00872ab82c7ee0a91dd7d7f1bdb20b35a2a90065150c88,49320bytes,10rootentries, official requirementsSHA9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6. Контрольe12d6913… оставлен dist/alfa_autoresearch_73d5aa4.zip и вGit73d5aa4. Первый ошибочный newcandidate/исправленнаяv2 сохранены; mainZIP соответствует v2, без подмены регрессионных arms по sourceequivalence.

## Что действительно достигнуто

Whole-agent masked regression37.578329562→36.951992372 (−1.66675%), Year<5→5,35000/18308,key68ef0344…. Classification.6494124789044244, прежний CSV52ea6af7… побайтно воспроизведён, exact-feature-group outer70529/17141,key239200bc…. Все4E2E armscompleted/finalrefitcompleted. Shadow8CPU/1200s/cap16,original1800s/defaulttrials. Actualoriginaloutputs22449/41166rows,1310.304/262.282sec,peak1046.03/1186.23MB. Originaltestlabelsunknown — это техническая проверка, не testscore. Прогнозы в local_data/strongest-team (ignored), манифест results/strongest-team/predictions.json; все/tmp также сохранены.

Это улучшение действующей политики, НЕ превосходство над team research: SergeyH73 RMSE34.520269; Ivanfe674da H107cachedfusion34.142888, H108blend33.979840; classH48.649531162/H78.659308414/H108blend.660279593. Windows/Mac и meta-in-sample ограничения отмечены; свежая runtimequality последних Ivan graphs не подтверждена. Версия ZIP для13.7676 неизвестна (пользователь подтвердил); по одному publicscore нельзя определить провалившуюся задачу. Старый regression index leak исправлен; прошлые innernumeric результаты не независимы. Эти outer многократно просмотрены, новой независимой гарантии нет.

## Код и закрытые направления

CausalHybridModel (automl_residual.py) реально выбран search/refit, доступны LLMschema/portfolio/build_pipeline. FIT-only generic key refinement, strictpast exponentially-decayed levels и LGBoriginal-unit changes на matchedhistoryrows, fixed50/50, stronghistoryET globalfallback. Без publiccolumnnames/данных/fittedweights. Classdefault не менялся. ST1/2 transport, ST3leafretrieval, ST4inferred driver, ST6posterior transfer(outer.60634), ST9groupOOF categorical TE(inner.631698) отклонены. ST8oneSEreplay не дал достаточногообоснования — не integrated. Не повторять эту серию под новымназванием/seed. Исследовательские automl_transfer.py/automl_decision.py не входят вZIP. Старыеworktrees иteambranches не менялись.

Верификация:81testsPASS results/strongest-team/unit.xml;9cleanroom и liveCLI receipts проверять в одноимённых JSON. Первый newcandidate dispatchошибка вcandidate_key исправленавbuild_pipeline, fallbackсохранил18308rows; e2e-first-error.json иoriginaldirectory сохранены. Два lateoriginalregtrialtimeouts штатно оставилиbestCSV и резервfinalrefit. Costaccounting не считать их0fit по seconds0 вагрегате: processlogs есть.

## API и среда

BASE_URL=https://changellenge.up.railway.app/v1 подтверждён пользователем. GET/models сего ключом вернул единственную qwen/qwen3.8-27b, PDF Qwen3.8-27B совпадает. Реальный propose tool-call2.368s/168outputtokens/0reasoning/1attempt; usagecost.0290771534716, единица не установлена; без ABspeedclaim. SyntheticliveCLI отдельно, полнаяonlineMLpolicy не оценивалась. Alternativeapi.alfahack-gateway26.ru сообщение длядругихтреков, неиспользовался. .env.localignored, keyнепечатается/неGit/неZIP; localprocessenv only, платформенный агентнечитаетenvfile. Локальнаякоманда воспроизведения: experiments/check_local_api.py --base-url CONFIRMED --model ID_FROM_MODELS --generate --output NEW_RECEIPT (платныйcall, не повторятьбезцели).

Python3.12 /private/tmp/alfa-research-venv/bin/python, pinnedMLpackages; DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/alfa-openmp/libomp/23.1.4/lib, MPLCONFIGDIR=/private/tmp/alfa-matplotlib. Linux/официальныйDocker не проверен. Данные drive-download-20261023T201455Z-1-001/{Регрессия,Многоклассовая классификация}; не коммитить. git/network/localhost/RSS требуют escalation. Полная новая серияостановлена; следующий расходтолькоотдельнымзаданием.

Далее: (1) controlledwhole-policyonlineQwenvsfrozenoffline, потому что внешнийiloveжурнал сообщает возможныйregretLLM/длинныхrun, причина унас неустановлена; (2) freshteamfusion с genericroles и равнымpreprocessing/OS, а не переносcachedweights; (3) сначалаизучить итогH109/110Ивана дляneuralcomplementarity, недублировать pendingfits. Дляплатформенногоsubmit нужна отдельнаякоманда, сохранитьSHAиper-tasklogs.


## Checkpoint перед compact — 24.10.2026

По поручению пользователя выполнено только сохранение состояния: новых экспериментов, API-вызовов, обучений и compact не запускать. Никакие процессы не прерывались.

### Git и пользовательская работа

- Репозиторий: `/Users/maxchibisov/Projects/HACKS/alfa-hack`, ветка `maxim/autoresearch`.
- Код, готовый ZIP и итог исследования зафиксированы и опубликованы коммитом **ced5e3229701550e8ec1a15df87cd3318956e40a**. Все runtime-файлы ZIP также побайтно соответствуют `052ebc1e4bfe5247c0fb5a69bfa94bc7ac7518ad`; это проверено в `results/strongest-team/final-audit.json`. Коммит текущего checkpoint содержит только AGENTS.md и этот файл, не новую поставку; его SHA смотреть в `git log -1`.
- Перед обновлением памяти основная рабочая копия была чистой и синхронизированной с `origin/maxim/autoresearch`: посторонних tracked/untracked изменений в неё не включать. Игнорируемые `.env.local`, данные, ZIP-backups и `local_data/strongest-team` сохранить.
- `/Users/maxchibisov/.codex/worktrees/ultimate-autoresearch-20261023/alfa-hack`: `codex/ultimate-autoresearch-20261023`, HEAD `eab478a039d1efd02c001f56f9984f1f623bad2d`, 33 записи dirty status; SHA256 статуса `8fccea66062f204529d8a6bb48d00e20b75fb193597d287d790611214b4e8ed8`.
- `/Users/maxchibisov/Projects/HACKS/alfa-ultimate`: `codex/ultimate-autoresearch-20261024`, тот же HEAD, 33 записи dirty status; SHA256 статуса `aac6cf6a2f756bd22e3dca86037f6a78d525b07fc2a75560f8bc84038a780de3`.
- Оба статуса совпали с финальным аудитом предыдущего этапа. Подробная исходная инвентаризация: `docs/sources/research-optimized-intake.json`; не переносить/не сбрасывать их изменения автоматически. `main`, `ivan_branch`, `sergey_branch` и старый remote `codex/research-optimized` сохранены.

### Процессы и checkpoints

Повышенная read-only проверка Python/pytest-процессов по cwd и путям проекта/окружения не обнаружила активных процессов проекта. Все ранее запущенные E2E, clean-room и live API CLI завершены. Не перезапускать завершённые sessions ради получения вывода и не пытаться присоединиться к уже завершённым PID из process-state.json.

Основные checkpoints: `73d5aa4` — прежняя поставка; `d04ec91` — начало серии/память; `052ebc1` — runtime нового гибрида; `72ab5c6` — промежуточные доказательства; `ced5e32` — итоговая поставка. Ошибочный первый E2E сохранён в `/private/tmp/alfa-strongest-e2e`, успешный frozen v2 — `/private/tmp/alfa-strongest-e2e-v2`; scientific caches — `/private/tmp/alfa-strongest-team`, чистая распаковка — `/private/tmp/alfa-strongest-cleanroom`, живой CLI — `/private/tmp/alfa-strongest-live-cli`. Долговременные копии predictions лежат в ignored `local_data/strongest-team`; перечень и хеши — `results/strongest-team/predictions.json`. Ни один из этих каталогов не чистить.

### Подтверждённая поставка и ограничения

Лучший проверенный **наш whole-agent**, а не рекорд team research: `dist/alfa_autoresearch_submission.zip`, **49320 байт**, SHA256 **f3a24044b90fb638df00872ab82c7ee0a91dd7d7f1bdb20b35a2a90065150c88**. Повторная проверка SHA перед checkpoint совпала. Архив: 9 корневых .py и неизменённый официальный requirements; старый ZIP e12… сохранён. Архитектура: solution → engine; data/schema/splits; models/DSL; panel discovery; residual hybrid; изолированные worker/runtime и bounded LLM. Платформа Python3.12/8CPU/32GB/3ч, `python3 solution.py task.md`, submission рядом с task, никаких обязательных вложенных модулей/внешних pretrained weights/сторонних LLM API.

Сопоставимые whole-policy результаты: regression **37.578329562 → 36.951992372 RMSE (−1.66675%)**, Year<5→5,35000/18308; classification **.6494124789044244**, group outer70529/17141, byte-identical control CSV. Shadow:8CPU,1200sec,cap16, API отключён ради парного ML-сравнения. Оба outer многократно просмотрены, это не независимое подтверждение и не public/hidden score. Original E2E:1800sec/default trials, final refit completed,22449/41166 строк; меток originaltest нет. Все4arms исполнили один и тот же frozen ZIP. Лучшие team research33.979840/.660279593 не превзойдены; их cached/cross-OS/meta in-sample ограничения описаны выше. Связь public13.7676 с версией неизвестна.

Ранее выполнены **81 unit test**, **9 clean-room сценариев**, **4 публичных CLI/refit** и **1 живой synthetic CLI** с Qwen:47 строк,7.619sec,1 LLM-предложение реально обучено. Receipts: unit.xml, cleanroom.json, e2e-v2.json, e2e-summary.json, live-cli.json, package-security.json, final-audit.json в `results/strongest-team/`. Эти проверки сейчас не повторялись. Linux/официальный Docker, hidden score и ML-качество полной public policy с живой Qwen не проверены.

Закрытые направления: ST1/2 перенос доноров и multivariate transport, ST3 leaf retrieval, ST4 inferred driver, ST6 posterior transfer, ST9 groupOOF categorical TE отклонены. ST5b/ST7 дали лишь малый single-candidate gain, но fixed hybrid улучшил whole-policy и включён; ST8 one-SE replay не интегрирован. Подробные отрицательные исходы/параметры/стоимость в `docs/EXPERIMENT_LOG.md` и JSON; не переименовывать старые гипотезы в новые, не повторять fit без содержательной причины. Старый regression index leak исправлен; прежние inner numeric оценки не независимы.

### API и договорённости

Основной endpoint `https://changellenge.up.railway.app/v1`, model ID `qwen/qwen3.8-27b` найден через authenticated /models и соответствует Qwen из PDF. Два реальных generation за завершённую серию; оба с0reasoning tokens, tool-call формат работал. В checkpoint API не вызывается. `.env.local` исключён из Git (проверено), секрет не печатается/не включается в отчёты/ZIP. Локально ключ только в окружении тестового процесса; агент на Boosters получает OPENAI_BASE_URL/KEY/MODEL только из среды и не читает .env.local. Альтернативный endpoint для других треков не использовался.

Работаем только в `maxim/autoresearch`. Пользователь разрешил перенос лучших механизмов из командных веток, сохраняя исходные ветки, историю и пользовательские изменения. Каждый подтверждённый прирост сохранять коммитом и пушить в эту ветку репозитория `https://github.com/vladkramerov/alfa-hack`. Без reset --hard/clean/force push; без субагентов/Superpowers. На Boosters ничего не отправлять без отдельного поручения.

### Восстановление после compact

1. Прочитать AGENTS.md, затем этот checkpoint; выполнить только `git status --short --branch`, `git log -3 --oneline` и read-only проверку процессов. Не переключать ветку при посторонних изменениях.
2. Сверить текущий ZIP/manifest с SHA f3a… и прочитать короткий `results/strongest-team/README.md`. За деталями обращаться точечно к e2e-summary.json, final-audit.json и журналу, не перечитывать всю историю и не запускать fit заново.
3. Сейчас незавершённого обучения нет. Исследовательская серия остановлена; ждать отдельного задания. Открытые направления: controlled online/offline Qwen whole-policy, Linux-контейнер, свежая проверка team fusion и результатов H109/110 без переноса public fitted weights. Это следующий этап, не автоматически разрешённый запуск после compact.

Команды на будущее (сейчас не запускались): `/private/tmp/alfa-research-venv/bin/python -m pytest -p no:cacheprovider -q tests`; `python experiments/package_solution.py`; `python experiments/verify_package.py --output NEW_PATH`. Native macOS: DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/alfa-openmp/libomp/23.1.4/lib. Учитывать, что package/verify пересобирают основной ZIP; API smoke расходует реальный бюджет, а E2E — обучение. Сам compact пользователь пока не поручал.
