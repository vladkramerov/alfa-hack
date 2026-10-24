# Состояние исследования — 24.10.2026

## Точка продолжения strongest-team

Активная просьба: превзойти лучшие решения команды по regression и multiclass, интегрировать и проверить всю политику, исследуя также первичные статьи/GitHub. Без отправки Boosters. Без субагентов/Superpowers, без изменения других веток и потери пользовательских файлов. Новая серия начинается с чистого maxim/autoresearch@73d5aa4d87ae4c90d61ec2268bdeb71fab2aa98b. Fetch показал новый origin/ivan_branch fe674da (в том числе fusion с Sergey и большой sprint), origin/sergey_branch24e3311. Предыдущие worktrees не трогать. Сначала точечно изучить их новые отчёты/код, не распечатывать огромные git diff.

Контрольная поставка: dist/alfa_autoresearch_submission.zip, SHA256 e12d691382bc280fae5b2063e2a28ca2d21a18847f1c23f786522dafd464a641,46651B; исходники73d5aa4. Оставить backup перед изменением. Официальный requirements SHA9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6.

Подтверждённые whole-agent результаты (results/score-recovery/e2e-final.json): regression masked год<5→5 44.810781→37.578330 RMSE; classification group outer .649412479 (CSV byte-identical control). Исторический лучший Сергей H73=34.520269, Иван H48=.649531162. Эти outer многократно просмотрены; НЕ независимые и НЕ Boosters score. Версия ZIP для public13.7676 неизвестна (подтверждено пользователем). Новые branch результаты ещё не изучены.

Старый regression nested mapping давал утечку outer, исправлен384a5a7; старые inner numeric-study не независимы. Диагноз results/score-recovery/temporal-index-diagnosis.json. Последние regression policy окна — последние известные train периоды; classification — старый nested маршрут (новый boosting/geometric default проиграл E2E и откатан). Дополнительные causal history/ES/cache механизмы сохранены. 77 tests+9 cleanroom passed. Полные результаты и rejected направления в docs/EXPERIMENT_LOG.md, results/score-recovery/. Не повторять их fit без содержательной причины.

## API / среда / следующие действия

Организаторы через пользователя подтвердили OPENAI_BASE_URL=https://changellenge.up.railway.app/v1. .env.local содержит ключ и ignored; агент автоматически не загружает его. Точный разрешённый OPENAI_MODEL пока не найден: официальная PDF называет Qwen3.8-27B, baseline берёт ID из среды. Чужой default не подтверждение нашего track. Найти точную инструкцию/список, иначе спросить пользователя и продолжать ML. Live calls в предыдущей серии0. Секрет не печатать/коммитить/класть в ZIP; local API тест только отдельным процессом.

Python3.12: /private/tmp/alfa-research-venv/bin/python. macOS env DYLD_FALLBACK_LIBRARY_PATH=/private/tmp/alfa-openmp/libomp/23.1.4/lib; MPLCONFIGDIR=/private/tmp/alfa-matplotlib. Данные drive-download-20261023T201455Z-1-001/{Регрессия,Многоклассовая классификация}; не публиковать данные/predictions. Native dependencies exact official pins. Linux Docker не проверен. git/network/localhost/RSS могут требовать escalation.

Дальше: короткий branch/results inventory, точная проверка model ID, литература и novel mechanism screening, конечная серия с сохранением прогресса/recipe до outer. Перед fit проверить HYPOTHESES и существующий кэш; сравнения разделить на exploratory outer, честный model-selection и whole-agent delivery. После серии сохранить лучший проверенный агент, ZIP/manifest/SHA, tests, журнал, commit и push maxim/autoresearch. Предыдущий полный журнал не переписывать.
