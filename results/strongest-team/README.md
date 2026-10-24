# Strongest-team: исследования 24.10.2026

Контроль — maxim/autoresearch73d5aa4, ZIPe12d6913…; команды origin/ivan_branchfe674da, Sergey24e3311. Секреты/данные/predictions внеGit. Все outer давно просмотрены; независимого или Boosters подтверждения нет. Версия ZIPдля13.7676 неизвестна.

| Уровень доказательства / задача | Версия | RMSE↓ / MacroF1↑ | Условия |
|---|---|---:|---|
| Whole-agent control, regression | Maxim73d5aa4 |37.578330|maskedCLI `<5→5`,35000/18308,8CPU,1200sec,cap16|
| Whole-agent control, classification | Maxim73d5aa4 |.649412479|maskedCLI group outer70529/17141,8CPU,1200sec,cap16|
| Research reg, team |SergeyH73|34.520269|same temporal outer,Mac; fixed key|
| Research reg, team |IvanH107|34.142888|sameouter,cachedcross-OS mixture,no fresh runtimequality|
| Research reg, team |IvanH108blend|33.979840|sameouter,cachedH107+NumPyTabM-mini; not promoted|
| Research class, team |IvanH48|.649531162|samegroupouter,stackedmodel|
| Research class, team |IvanH78|.659308414|sameouter,meta in-sample teacher caveat|
| Research class, team |IvanH108blend|.660279593|sameouter,cachedH78+NumPyTabM-mini; not promoted|

Разные inner между командами не объединены. Team recipes переносятся как механизмы, публичные fitted веса/состояния не входят в платформу. Новые исследования:

| Проверка | Сопоставимый inner | Outer, если допущен | Решение |
|---|---:|---:|---|
|ST1 direct donor|40.499557→38.822009|36.129126→36.763312|отклонить|
|ST2 multivariate original-loss transport|39.196620|не запускался|хужеST1|
|ST3 supervised leaf retrieval|.636708318→.635627272 лучшая положительная смесь|не запускался|все positiveweights проиграли|
|ST4 inferred driver log/isotonic|40.464435/38.832568|не запускался|хужеST1|
|ST5 residual dynamics loose keys|40.499557→39.250721|36.129126→36.206620|отклонить|
|ST5b stable keys +residual|40.499557→39.147331|36.129126→36.096956|слишком малый gain сам по себе|
|ST6 H78 mechanism/current teacher|.636708318→.648375385|.637337946→.606339730|отклонить; переноса нет|
|ST7 fixed stable/dynamic hybrid|period4 33.056728→30.716224|36.129126→36.111884|только кандидат дляwhole-policy,не объявлять победой|
|ST7 inner-only simplex|period4 30.654919|36.083734|publicweightsне встраивать|

JSON содержат ключи/хеши/параметры/время/RSS. Предсказания: /private/tmp/alfa-strongest-team, замороженные новые CLI: /private/tmp/alfa-strongest-e2e-v2. Часть ST1/2/4/pool — только обучение статистик/линейных/изотонических поправок; oldbase learnerfits0. ST6 needed2 fresh teachers дляtrainprobabilities (их нет в старых кэшах), внутреннийcontrol воспроизвёл cache побайтно/maxdiff0. Первая интеграционная ошибкаdispatch сохранена e2e-first-error.json; fallbackCSV валиден, qualityclaimнет.

Проверенный API: api-model-discovery.json и api-generation.json, modelID отauthenticated/models. Один платный generation,2.368sec,1441tokens,0reasoning; cost единица провайдера не установлена. Альтернативный URL для других треков не использовался. На платформе только переменные окружения, безавтозагрузкиenvfile.

Литература: [TabM](https://arxiv.org/abs/2410.24210) — уже исследуетсяИваном, недублировали; [TabR](https://arxiv.org/abs/2307.14338) — мотивирует supervisedretrieval, нашаST3 неTabR; [TabReD](https://arxiv.org/abs/2406.19380) — time-split иindustrialshift важнееIID; [TabDPT](https://arxiv.org/abs/2410.18164) — pretrainedstate противограничений. Публичные MUCG9/hilhool/NolReaction/ilove4fak проверены в прошлойсерии; no license/no foreigncodeimport, актуальные выводы results/score-recovery/external-sources.json.

Статусwhole-policy: ожидается e2e-v2.json; только после него принимать илиоткатывать новыйdefault. КонтрольZIPсохранён.
