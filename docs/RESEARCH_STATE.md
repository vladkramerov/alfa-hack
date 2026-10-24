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
