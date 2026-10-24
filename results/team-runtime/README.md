# Сопоставимый runtime-контроль и переносы

24.10.2026. Исходный код команды Ivanfe674da, `deliveries/sergey_ivan_fusion_v1`, source hashes в control*.json. Никаких изменений других веток. Все outer исторически многократно просмотрены, независимого/Boosters score нет. Связь старого13.7676 с ZIP неизвестна.

| Уровень / задача | Наша поставка f3a… | Свежая командная delivery | Условия |
|---|---:|---:|---|
| Whole-agent regression RMSE↓ |36.951992372|**34.129124076**|одни maskedCSV,35000/18308,Year<5→5,key68ef0344…|
| Whole-agent classification MacroF1↑ |**.649412479**|.627615829|одни maskedCSV,70529/17141,exact-Xgroup,key239200bc…|

Одинаковые бюджеты1200sec/cap16/offline,8CPU ceiling; командный worker фиксированно4threads, наш8. Командная reg110.44s/1459.75MB; class408.76s/4799.22MB. Classification первый run без imblearn — техническая диагностика, отдельно сохранён, не сильнейший контроль. После установки official0.12.4 stacking завершился, но политика выбрала macro_f1_calibrated с тем же итоговымCSV. Промежуточный stack outer.639741679, inner.602242764; выбранный calibration inner.641384928/outer.627615829. Selection regret доказан в этом run, его причина/связь сBoosters неизвестна. Сравнивать с историческими team33.979840/.660279593 как эквивалентными whole-agent нельзя.

| Перенос / гипотеза | Inner, одинаковый baseline | Outer | Решение |
|---|---|---|---|
|HT1 H48 groupedOOF meta на ownXGB/ET, вместо ST6 in-sample|.606104897→.599992354|не запускался|отклонить|
|HT2 finite-pair discovery, percent-support, groupedH83isotonic|33.056728→30.365662|36.129126→36.043242|candidate gain.238%, неwholepolicy, неинтегрирован|
|HT3 H43/H73 pooledgroupfallback внутриgenericHybrid|30.518055→31.515886|не запускался|отклонить|

HT1 retry89.02s/1108.86MB,8teacher+1meta; первый8teacher+1meta потерян приошибкеlist-indexing, времянеизвестно и не считаетсянулём. Checkpoints добавлены доmetrics и послекаждогоfold, первыйошибочныйreceipt сохранён. HT2 outer5.08s/382.39MB; первыйinnerreceipt np.bool_ serialization упал после сохраненияpredictions, inner восстановлен безrefit; время/audit неизвестны. HT3 48.39s/681.45MB,3ET+1LGBfit. Полные ключи,параметры,протоколы в JSON и docs/EXPERIMENT_LOG.md.

Error-slices: warmunique12242rows team19.92464/own21.99422; cold/ambiguous6066 team52.09925/own56.07893. Это posthoc диагностика, не независимый тест и не правило выбора поouter.

Платформенный runtime и ZIP не изменены, новых подтверждённыхwhole-policyулучшений нет. Narrowtest genericratio1passed, py_compile и gitdiffcheckpassed. APIcalls0. Прогнозы ignored local_data/team-runtime и /private/tmp/alfa-team-runtime-control*. Данные/модели/секреты вGit не включать. Следующий шаг: сначала совместимое neural complement наinner, с конечнымбюджетом и без повторения завершённыхH108.
