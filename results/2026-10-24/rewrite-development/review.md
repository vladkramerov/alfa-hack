# SPEC010: проверка реализации

Дата2026-10-24. Три независимых реализации: neighbors, FM, data; отдельные controller/final reviewer. Реальные научные fits выполнял только frozen experiment runner, не reviewers.

- NN:8 тестов, scalar brute-force oracle; независимый reviewer дополнительно сверил60 случайных сочетаний размеров/k/blocks/missing/ties.
- FM:9 тестов, независимый попарный oracle и finite differences; дополнительная максимальная ошибка производных8.67e-9. Held-out synthetic nonlinear signal изучен; это не победа на публичных данных.
- Data:15 тестов, nested/renamed/opaque inputs, train-only numerical statistics, group/time splits и probability submissions.
- Memory/controller: hash before fit, started persisted, seed-only repeat rejected, strict15%/10% boundaries, retries total deadline, valid submission retained on resume и reuse без новогоfit.
- Первые controller RED наблюдения: отсутствующие модули до реализации. Отдельная техническая диагностика обнаружила запрет записи в default sandbox TEMP; tests используют workspace/tmp, не меняя продукт ради теста.
- Исправлены по review: probability-column order, binary ROC/logloss output dimensions, local fallback при API failure, проверка остатка времени, восстановление CSV вместо повторного fit, frozen worker path.
- Исправлены в read-only verifier без повторного обучения: per-run fit/valid связаны с frozen stage; продвижение запрещено при любом failed audit; actual fit fingerprints связаны с исходными training X/y/config.

Root полная проверка: **257/257 tests OK** ([лог](final-suite.log)); после создания workspace/tmp в независимых fixtures дополнительно32/32 target tests OK. [Чистый CLI](../rewrite-clean-cli/report.md):4/4 synthetic tasks,8 actual synthetic evaluate/refit, без API/ZIP. Requirements byte-exact.

Известные ограничения: реальный LLM endpoint недоступен без platformenv; private quality не измерено; feature dtype определяется schema inference по train, но статистики/imputation/encoding learners обучаются только на fit. Дополнительный public split не считается полностью независимым private test. Новый исследовательский код не заменяет прежнюю поставку до прохождения метрического gate.
