# SPEC-009: первая волна альтернативных подходов

Дата: 2026-10-24, Москва. Статус: первая волна завершена с одним техническим ограничением. Основание: пользователь поручил ещё20 новых идей, продолжить работу и найти принципиально иной подход с целью десятикратного улучшения платформенного результата. Это цель исследования, не гарантия. Сообщённый1376.762 относится по уточнению пользователя к solution_improved_metrics.zip; runID/точнаяформула/SHA платформы неизвестны.

Итог: десять терминальных исходов, девять численных оценок, H53 без метрики из-за отсутствующего worker output. 65 независимых сверок прошли,22 новых fit без повторов. Единственный новый победитель H69: RMSE39.39219382766771, класс H48 остался лучшим с Macro-F1.6495311620074445. H65 выявила и отдельной правкой устранила потерю строковых ID/classes; публичные X/y/dtypes/split строго неизменны. 206 unit-тестов,5mock API сценариев,1сквозной lexical-сценарий,2полных refit из чистого ZIP прошли. [Приёмка](../../results/2026-10-24/ALTERNATIVE_DELIVERY.json). REQ-907 выполнен для9полученныхscore; технический H53 не объявлен отрицательной модельной проверкой. Цель10x и hidden score не подтверждены.

## Контракт

- REQ-901: текущие проверенные H43RMSE40.972997661684175/H48MacroF1.6495311620074445 и ZIP solution_combined_metrics.zip сохранены. Только строгое улучшение на том же outer и независимая сверка допускают новый baseline.
- REQ-902: двадцать proposed H49…68 записаны отдельно; первая исполняемая волна — H49,H51,H53,H54,H55,H56,H58,H60 (оба task), а также новая H69temporalprofilememory. Это9 ID/10 оценок. Не выдавать оставшиеся идеи за выполненные. H68диагностика отдельна от модельных метрик.
- REQ-903: публичные SHA/outer из2026-10-23 неизменны, X не содержитtarget/id. Selection по outerlabels запрещён; все перечисленные параметры фиксированы, дополнительного поиска нет. No old baselinevalidationrefit; совместимый cache обязателен для reusedкомпонентов. Новые stateful fits и estimatorfits читают полный ledger и получают persistentstarted/fingerprint до fit.
- REQ-904: перед новой оценкой отдельная карточка/report/request/sourceSHA/dataSHA/split. После каждого исхода немедленный append в HYPOTHESES.md/results/README. Таймаут/ошибка не равны отрицательной научной проверке. 4threads,1800sec/case, data/test/validation не удалять.

## Фиксированные методы до первого fit

| ID | Изменение и параметры |
| --- | --- |
| H49 | Soft distillation: q=.7onehot+.3teacher; teacher probabilities из уже сохранённых raw/SMOTE/frequency baseOOFs и frozenH48meta. Это in-samplemetaonbaseOOF, не независимый teacherOOF score. ОдинXGBcustomCE600/depth6/lr.05/lambda1/histCPU. Grad=p−q, Hessian diagonalupperbound2p(1−p), floor1e−6. Безteacherrefit. |
| H51 | Balancedsoftmax traininglogits+log(trainingclasscounts), обычныйsoftmaxдляinference; XGBcustom600/6/.05/lambda1/hist, numclasses толькоfit. Grad/Hessian сначала numericaltests. |
| H53 | Fit-only categoryrare mapping cap128, median/OHE/scaling; NyströmRBF256landmarks seed42, gamma=1/medianpositivesquareddistance среди<=512fitrows; LR C1/lbfgs/maxiter1000, безclassweights. |
| H54 | Numericmedian→SplineTransformer5quantileknots/degree3/constantoutside; categoricalOHE rarecap128; Ridgealpha10/sparse_cg, yoriginal. |
| H55 | ПятьLGBMquantile alpha.1/.3/.5/.7/.9,200trees/lr.05/leaves31/min20/lambda1; nativecategories/unscaled, yoriginal. Среднее5прогнозов, никакихouterweights. |
| H56 | H43 savedouterprediction/model reuse; NEW2temporalH43OOF, residualy−OOFprediction наcoveredrows; residualLGB200/depth3/leaves7/lr.03/min20/lambda10/nativeunscaled, coefficient1. Нет независимой оценки residualfit на тех же trainingtargets. |
| H58 | Trainingdomain=(Year==maxfitYear), Year исключён изdomainX; StratifiedKFold3/seed42; LGBbinary100/leaves15/lr.05/min20/lambda1. OOFratio=clip(p/(1−p)*nold/nlatest,.2,5), normalize mean1; новыйrawET200/24/leaf2/features.8, logy, weighted loss. Это веса поX, не H36 по одному году. |
| H60 | Append1copy каждойfitстроки, маскировать5%numericполей rng42; защититьYear_Factor икатегориальныеключи. yнеизменный. RegressionET200/24/leaf2/.8/logy; classXGB600/6/.05/lambda1/hist; median/OHE/unscaled. Толькоfitaugmentation. |
| H69 | Вместо универсального регрессора — lookup прошлыхy по physicalprofile [State_Factor,building_class,facility_type,floor_area,year_built,ELEVATION]. НиID, ниannualrating/weather не ключ. Для каждогоqueryYear использовать толькоtrainYear<queryYear; decay.5^(latest−pastYear), trainingpastmean; alpha=min(.9,n_past/(n_past+1))*exp(−.25*max(0,queryYear−latestYear−1)). pred=alpha*memorymean+(1−alpha)*H43. Неизвестныйпрофиль→H43. Профиль не доказаннаяidentityздания, коллизии показать в аудите. Нет подбораключей/альфы поoutery. |

- REQ-905: для H69 отдельно проверить future exclusion, missing keys, неприменимостьбезявноговремени, unseenfallback, совпадениечисловыхключеймеждуdtype. В featureaudit толькоX, targets не используютдлявыбораключей. Это не standalonecat encodingH15: lookupстрогопрошлыхлетпофиксированномусоставномупрофилю и перенос истории по годам.
- REQ-906: newproductionmodules не импортируют experiments. Experimental H49/H56 соsavedteachers/basecontext могут оцениваться offline; перенос разрешён только после реализации идентичной train-only логики finalrefit. Нетэкспериментальноговыигрыша→старыйZIPбезизменений.
- REQ-907: независимо пересчитать10primaryscores изpredictionиоригинальныхy, проверитьpositions/classes/probabilities/fits/cache/source; sourcecopies не менять. При выигрыше полный регресс/flatZIPпроверка повторяется с новымархивом, текущийнеперезаписывать.

## Ограничения

Macro-F1 ограничен1, увеличение текущегоF1 в10раз невозможно. Дляплатформенногобалла1376.762 цель10x не переводитсяавтоматическивRMSE4.0973: составзадач/формула неизвестны. Повторно просмотренныйouter остаётся разведочным, поэтому выигрышlocal не равен privatequality. Нельзя улучшать score удалениемсложныхvalidationстрок, использованиемtestlabels или нарушениемтребованийплатформы.
