# Alfa Hack AutoResearch — рабочая память

**Strongest-team завершена; точка продолжения — docs/RESEARCH_STATE.md. Ветка maxim/autoresearch. Подтверждённый ZIP f3a24044b90fb638df00872ab82c7ee0a91dd7d7f1bdb20b35a2a90065150c88,49320B. Повторять готовые fit нельзя. Результаты results/strongest-team/, единый журнал docs/EXPERIMENT_LOG.md.**

Общение по-русски. Минимальные изменения; без субагентов/Superpowers. Не трогать main/ivan_branch/sergey_branch; без reset/clean/force push. Ключи, данные и fitted models не включать в Git/ZIP. Без отдельного поручения ничего не отправлять на Boosters.

## Состояние — 24.10.2026

Контроль73d5aa4 и ZIPe12d6913… сохранены. Новая серия переносит принцип Сергея stable memory + annual residual dynamics на автоматически найденные FIT-only ключи, без публичных имён полей/fitted state. Новый CausalHybridModel реально включён в regression portfolio и выбран в обоих E2E. Classification default сохранён. Старый remote codex/research-optimized и оба прежних dirty worktree не менялись. Последние изученные team refs: Ivanfe674da, Sergey24e3311.

**ERRATUM:** прежние regression inner использовали локальные позиции как глобальные, в search попадал outer год5. Исправлено384a5a7; старые numeric/public inner не независимы. Доказательство results/score-recovery/temporal-index-diagnosis.json. Многократно просмотренные outer сейчас только exploratory, даже при честных fit-позициях.

Score13.7676 и ориентир26.6404 сообщил пользователь; версия ZIP неизвестна (подтверждено). Ни одному локальному архиву score не приписывать.

## Архитектура / контракт

solution.py → automl_engine.py: атомарный ранний fallback, до36 regression/16 classification trials,4LLM calls, резерв refit. Regression rolling latest train periods; classification nested legacy. automl_data.py — schema/split/metric/submission; automl_models.py — модели/DSL/ES/blends; automl_panel.py — FIT-only entity discovery и strict-past features; automl_residual.py — refined stable keys, оригинальный residual LGB только на history-matched rows, fixed50/50 stable-memory/dynamics с global fallback. automl_worker.py — изолированный fit/cache; automl_runtime.py — deadline/memory; automl_llm.py — platform API/JSON/reasoning fallback.

Python3.12,8CPU,32GB,до3ч; python3 solution.py task.md, CSV рядом с task. ZIP≤100MB:10корневых файлов (9py+неизменённый официальный requirements). Без обязательных вложенных модулей, внешних pretrained state, известных public полей и сторонних LLM API. Research automl_transfer.py/automl_decision.py и experiments не входят в ZIP.

## Сопоставимые измерения

- Whole-policy masked regression `<5→5`,35000/18308,key68ef0344…: контроль37.578329562 → **36.951992372 RMSE (−1.66675%)**. Actual final CSV, одинаковые train/test CSV,8CPU/1200s/cap16. Это улучшение поставки; лучшие team research не превзойдены.
- Whole-policy classification group outer70529/17141,key239200bc…: **.6494124789044244**, CSV byte-identical control SHA52ea6af7….
- Лучшие team research на тех же outer: SergeyH73 RMSE34.520269; IvanH10734.142888, H108blend33.979840; classH48.649531162, H78.659308414, H108blend.660279593. Иван: Windows/cached mixture/meta in-sample caveats; свежий runtime этих последних графов не оценён. Не выдавать за качество действующего агента/Boosters.
- ST1/2 donor+multivariate transport, ST3 supervised leaf retrieval, ST4 inferred-driver transport, ST6 posterior transfer, ST9 groupOOF categorical TE отклонены. ST5b/ST7 tiny individual gain, но fixed hybrid дал policy gain. ST8 paired1SE replay не интегрирован. Все варианты/время/ключи в JSON и журнале; не переименовывать в новые гипотезы.

## Проверки / API / продолжение

81 tests passed; чистая распаковка — results/strongest-team/cleanroom.json. Четыре frozen CLI: results/strongest-team/e2e-v2.json, все final_refit completed. Original test22449/41166 строк;1310.30/262.28sec,peak1.05/1.19GB. Source ZIP один и тот же f3a… во всех4arms. Прогнозы долговременно сохранены в ignored local_data/strongest-team, receipt predictions.json. Предыдущие результаты/tmp не удалять. Linux/официальный Docker/hidden score не проверены.

API подтверждён: BASE_URL=https://changellenge.up.railway.app/v1; authenticated GET/models вернул qwen/qwen3.8-27b, совпадает с Qwen в PDF. Tool-call2.368s/168outputtokens/0reasoning/1attempt; receipts api-*.json. Live synthetic CLI — live-cli.json. Полная public ML-политика оценена offline, выигрыш с живой Qwen на этих задачах не измерен. .env.local ignored, автоматически платформенным агентом не читается; key только в локальном процессе. На платформе все3OPENAI_* из среды.

Python /private/tmp/alfa-research-venv/bin/python; native macOS libs DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/alfa-openmp/libomp/23.1.4/lib.

- python -m pytest -p no:cacheprovider -q tests (localhost mocks требуют bind).
- python experiments/package_solution.py; python experiments/verify_package.py --output NEW_PATH.
- experiments/strongest_e2e.py -v2 умеет пропускать завершённые arms; НЕ запускать ради повторения.

Далее: один связанный с SHA Boosters-submit только по поручению; ограниченное whole-policy online/offline сравнение Qwen; свежая проверка team fusion/TabM H109/110 без копирования public fitted weights. Не начинать новую серию автоматически.
