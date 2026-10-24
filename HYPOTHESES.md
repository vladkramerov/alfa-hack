# Единый журнал гипотез

SPEC016 HYP-109 registered ДО fit: [fullTabM regression](docs/hypotheses/HYP-109-full-tabm-regression.md), adapters r/s/bias EVERY hidden layer + learned numeric PWL8bins/16d, k8/width96/blocks2/drop.1/60epochs/AdamW.001/CPU4; originalstandardizedMSE. HYP-110 registered ДО fit: [fullTabM classCE+traininverse-sqrtweights](docs/hypotheses/HYP-110-full-tabm-classification.md), та же новаяархитектура; не H108/backendrename. ≤120sec/neuralfit/900sec perhyp. Actual previousZIP9a3924a controls (не H107/H78cachedquality), однократный baseline ZIP comparison прямо разрешён текущим поручением. Innerfirst, weightfreeze beforeouter, terminal/report сразу; публиковать только measured runtime gain и clean flat delivery.

HYP-108 итог: 4/4completed и no-fit verification PASS; outer blend RMSE33.97984036360298 против34.142888130110755 (0.4775%gain), Macro-F1.6602795926985132 против.6593084142037399 (0.1473%gain). Standalone обеим контролям проиграл. [Отчёт](results/2026-10-24/series-tabm-mini-v1/report.md). Только exploratory cached comparison; runtime/hidden не проверены, baseline/ZIP/agent schema не менялись. Все4fit закрыты для повторов, сменаbackend/seed не новаягипотеза.

SPEC015 HYP-108 registered ДО fit: [TabM-mini](docs/hypotheses/HYP-108-tabm-mini.md), NumPy input adapters/random signs + shared2x64ReLU +8independentheads; AdamW.003/wd.0001/batch256/60epochs, ≤240sec/fit/900sec total. Не H72/H94 teacher-residual MLP и не H62FM. Fit-only numeric median/missing/quantile-normal+OHE64; regression originalMSE/classCE. Cached H107/H78 controls, pure+inner-only frozen blend, outer только strict inner gain; no oldfits. Torch locally missing/network blocked; это самостоятельный research spike, production/ZIP не меняются.

SPEC011 доrealfit HYP-099 registered: [globalone-to-onerecordlinking](docs/hypotheses/HYP-099-global-entity-assignment.md), Hungarian/exactH70reserve/latestpastyear/alpha.35/cost≤1/dummy1.000001/capN*(M+N)4m, no H70beta refit. Этоnewjointallocationcohortconstraint, неH81localnearestparamповтор; querycohortdependent/noqueryy/physicalidentitynotproven. CachedH70fallback/stateSHA/source/config/data/splitsfreeze, firstinner, everyoutcomepreserved.

SPEC011 доrealfit explicitregistered HYP-098: [exactempirical probabilityoffsets](docs/hypotheses/HYP-098-exact-empirical-f1-offsets.md),3rarefirstpasses/no tiedgroups/directactualF1onTRUEOOF RAWteacher(nonewteachers), неH37multipliers/H75expectedsoftcounts. HYP-100: [frozenH78+empiricaloffsets](docs/hypotheses/HYP-100-frozen-posterior-empirical-composition.md), noH78refit, newpostdecision offsets; innerproxytraininginsample/outerH48meta-insample explicitlyaudited, неindependentOOF. Новыеinputstate/source/config/data/splitsSHAfreeze; innerfirst, everyresultpreserved.

SPEC011 доrealfit HYP-095 registered: [noise-robust GCE](docs/hypotheses/HYP-095-generalized-cross-entropy.md), q.7 exactgradient иinverse-sqrtTRAINclassweightsmean1; LGB300/63/lr.05/lambda5/2CPU/positiveCEsurrogate; TRUEOOFteacherреюзH88, нольteacherrefit. ОтличиеH50focal противоположноеdynamicdownweightневероятныхlabels, неH39staticweights/H49distillation/H51balancedsoftmax; всеlabels/rowsсохранены. Историяпрочитана/source/config/data/splitteacherSHAfreeze, innerfirst/≤900sec.

SPEC011 доrealfit explicitregistered HYP-096 и HYP-097: [honestposterior decisions](docs/hypotheses/HYP-096-097-honest-posterior-decisions.md), прежниеH78properCE+F1/H75affineF1 с НОВЫМH88TRUEOOFtrainingteacher, неin-sampleматрицейстарыхfit. Параметрынеизменны/нольnewteacherfits/всеисточникиcacheпроверены. Source/config/data/split/teacherSHAfreeze, innerfirst; это новаякомбинация процедуры, не новыйseed/имяидентичноговхода.

HYP-093 и HYP-094 явно registered доfit: таже frozenкарточка HYP-092-094, выше сокращённая запись HYP-092/093/094 не прошла literal-IDguard для093/094. Две preflight остановки без созданияfitclaim/безобучения сохранены отдельно; исправляется только записьID, научныеconfigнеизменны. HYP-092уже допущена. Историяперечитана передтехническимпродолжением.

SPEC011 доrealfit registered [HYP-092/093/094](docs/hypotheses/HYP-092-094-honest-objectives.md): unchangedexactsoftF1LGB/LovaszLGB/softF1MLP с PROPEROOFteacher вместоinsampleпредыдущихH80/H77/H72. H88innerсделал3nestedteachers, все дальнейшие гипотезыТОЛЬКОreuseпроверенныхnpz+oldfrozensource/config/dataSHA, нольteacherrefit. ФиксированнаяinitialgridH92rounds300/900,H93rounds200,H94epochs40/150,2CPUeach,≤900sec/hypвсеварианты+outer. Иннерпервый/выборouterдоскоринга; историяпрочитана, source/config/data/splitsfrozen, отдельныеoutcomesresults.

SPEC011 доrealfit registered [HYP-090](docs/hypotheses/HYP-090-observable-missingness-expert.md): observablemissingratinggateonly, originalRMSECat800/depth6/lr.05/l2=20/2CPU, subsetFeature29/30/31, min300, fixed.7expert+.3cachedH70; knownratingbitunchanged. ActualsubsetlearnerMemoryclaimдоfit+mask/indicesaudit, noH70refit; это неH35buildingclasslogy/H45targettailgate/H41wholetaskrouter. Историяпрочитана/source/config/data/splitsfrozen; firstinner andalloutcomesresults.

SPEC011 доrealfit registered [HYP-088](docs/hypotheses/HYP-088-honest-context.md): rawteachertraining TRUEgroupedOOF via3NEWnested31kXGBsubsets inner (каждыйMemoryclaimдоfit); originalouter rawOOF/H11preds изготовогоcache,0oldrefits. H74fixedCat400/depth5 обучаетсянаOOFteacher, referenceouterпрежнийH48! [HYP-089](docs/hypotheses/HYP-089-panel-differences.md): withinentity Δy Ridge100 наdeltaweather/ratinginteraction+donory/age+facilityOHE, meanimputeStandardScalerfitonlypairs, boundedtransport; nooldfit. Историяпрочитана, source/config/data/splitsfrozen; innerfirst/≤900secsallcomponents.

SPEC011 до вычислениявесов registered [HYP-091](docs/hypotheses/HYP-091-new-regression-mechanisms-pool.md): fixednewcomponentsetH70/H83/H85, simplexSLSQPinneronly200/1e-12; не повторH79H70/H76. Cachedinnerpredonly, weightfreeze beforeH85outer; ≥2active+strictinnergain required, nooldfit.

SPEC011 доrealfit зарегистрирована [HYP-086](docs/hypotheses/HYP-086-state-space-history.md): causalKalman history normalized logy−frozenH70beta*q(rating), Q.01/R.04 fixed, reconstructconditionalmean withvariance, alpha.95/support.25/age.25; noH70betarefit, cachedH70fallback. Unknownqueryratingunchanged. Новыйstate-space filter всехpastobservations, неH69weightedrawmean/H70latestdonor/H83isotonicratio. Source/data/config/splitfreeze; firstinner/≤900sec.

HYP-063 feature-only audit архивирован: [report](results/2026-10-24/HYP-063-01-feature-only-audit/report.md), counts.json. Нольmodelfits, нольqualitymetrics; сходство не доказывает buildingidentity/утечку, исходныйJSONTypeErrorиуспешноепродолжение сохранены.

SPEC011 доfit зарегистрированы [HYP-084](docs/hypotheses/HYP-084-signature-posteriors.md): hierarchy numericpacket signature counts→Dirichletposterior, q=(counts+20pi)/(n+20), support≥8, alpha=.5n/(n+50), convexteacherpool (не posteriorproduct); [HYP-085](docs/hypotheses/HYP-085-conditional-lognormal.md): CatBoost800/depth6 RMSEWithUncertainty log1p(y), conditionalmean exp(mu+.5variance)-1 variancecap2, features29/30/31; [HYP-087](docs/hypotheses/HYP-087-query-cohort-f1.md): unlabeled querycohortprior expectedMacroF1 with H78cachedcalibration, notbatchinvariant/noquerylabels. Историяпрочитана, новыеsource/config/data/splitsfrozenдоfit; innerfirst/no oldrefits, отдельныеотчётывключаянегативные.

SPEC011 доrealfit зарегистрирована [HYP-083](docs/hypotheses/HYP-083-original-loss-transport.md): withinprofile monotone ratio(current_y/donor_y) отdelta-rating, exact originalMSE via donor_y² weights, constrained isotonicexp±1.5 (noobservedratio clipping), facilitymin100/shrink200; alpha.95support.25age.25. ОтличиеH70логloss/H71globalratingnormalization. CachedH69fallback/nooldfits; фиксированыsource/config/data/split, strictpast/collisionreject, firstinner.

SPEC011 доrealfit зарегистрирована [HYP-082](docs/hypotheses/HYP-082-network-semantics.md): networksemantic preprocessing — удалитьstcpb/dtcpb, TTLupperbucket/hopfraction, directionalimbalances/loss-per-packet/RTTconsistency, без повторныхzeroindicatorH26/H27. H74CatBoost400/depth5fixed/cachedteacher, rawrowsunchanged, 2CPU. Innerпервый/новыеsource/config/dataSHA/splitfrozen; не повторH5fourratios/H16categoryjoint/H17frequency.

SPEC011 доrealfit HYP-050 переведена изproposed вregistered: [unweighted gamma2 focal](docs/hypotheses/HYP-050.md), exactgradient/LGB200/63leaves/lr.05/lambda5/2CPU/positiveCEapproxcurvature, cachedteacherinit. Не H39staticweights/H51balancedsoftmax. TDDfiniteDiff6/6, новыеsource/config/dataSHA/splitfrozen доfit; innerfirst, nooldlearnerrefit.

SPEC011 зарегистрирована доfit [HYP-081](docs/hypotheses/HYP-081-fuzzy-temporal.md): консервативный fuzzytemporaloverlay поверх cachedH70(alpha.95), alpha.5/area.001/year±1/margin.5/age.25/support.25; no refitH70, перенос beta изсохранённогоH70audit. Новое приближённое сопоставление только ранее unmatched; feature-onlyH63audit не являлся metricfit и не доказывает identity. Строгоinnerfirst, source/config/dataSHA frozen; исход и отрицательные результаты отдельноresults.

SPEC011 доrealfit HYP-080 зарегистрирована: [exact coupled softF1 residual LightGBM300](docs/hypotheses/HYP-080-soft-f1-residual-boosting.md). Комбинация H72objective с H77representation, вместоLovaszrank/MLP; параметрыfixed300rounds/63leaves/2CPU/positiveapproxhessian. Никакихстарыхfit; innerfirst, source/config/datafreeze.

SPEC011 зарегистрирована до вычисления весов HYP-079: [новая комбинация H70+H76](docs/hypotheses/HYP-079-transport-panel-blend.md), analytical convexweight fromcachedinnersameYear4, только еслиinnerстроголучшеH70; wfreezebeforeouter, reuseH70outer/nooldfits. Это отдельная проверка комплементарности двух новых методов.

SPEC011 вторая волна, зарегистрировано ДО fit: [HYP-076](docs/hypotheses/HYP-076-panel-transition.md) — CatBoost800/depth6/lr.05/l2=10 оригинальный y на strict-past entitypairs+lagtarget/weatherdelta, alpha.9/минимум32pairs; [HYP-077](docs/hypotheses/HYP-077-ambitious-classifier.md) — LovaszSoftmax rank gradient+.5CE, LGB200rounds/63leaves/2CPU, teacher init logits; [HYP-078](docs/hypotheses/HYP-078-posterior-macro-decision.md) — proper CEtemperature/classbias calibration перед expected affineMacroF1. Новые механизмы, nooldfits, frozen source/data/split/config, ≤900sec/hyp; каждое исследование сначалаinner, выборouterдоегоscore. SPEC011дополнение отмечает дальнейший адаптивный поиск после H70outer.

SPEC011 зарегистрирована до fit HYP-075: аффинные a*p+b expected-MacroF1 decisions,25 iterations/damping.5/eps1e-8, training priors only, без inverse-prior correction; [карточка](docs/hypotheses/HYP-075-affine-macro-decision.md). Отличие от H37 — additive offsets и expected confusion objective. Inner cached raw teacher, без старых fit; outer только после выбора финалиста.

SPEC009 актуальныйитог:10terminal9численныхоценок(9ID),22уникальныхfit,65независимыхсверок. H69memory — новыйbaselineRMSE39.39219382766771; H48MacroF1.6495311620074445сохранён. H53техническаяошибкабезscore, не отрицательнаянаучнаяпроверка. H65семантическаяпроверка6/8выявила2ошибки, затемисправлена8/8безpublicfit и с точнойнеизменностьюданных/split; H68platformprovenanceнедостаточноданных. [Отчёт](results/2026-10-24/series-alternatives-v1/ALTERNATIVES_REPORT.md). Не повторять завершённыеmodelконфигурации; остальныеproposedIDещёнезапущены.

SPEC-008 завершена: семь новых комбинаций,63/63 независимых проверок,24 уникальныхfit,0 старых baselinevalidationrefit. Включены H43RMSE40.972997661684175 и H48MacroF1.6495311620074445; H42/H44/H46 проиграли, H45/H47 улучшили прежний incumbent, но уступили победителям. Все7 закрыты для повторов. [Отчёт](results/2026-10-23/series-combinations-v1/COMBINATIONS_REPORT.md).

Обновлён: 2026-10-23. **Перед КАЖДОЙ гипотезой прочитать этот файл целиком; проверенное не повторять.**
Новую гипотезу и критерий записать до запуска в docs/hypotheses/, затем внести сюда статус «Запланирована»; после проверки сразу сохранить исход и ссылку на отдельный results/. Новый seed/имя/техническая правка не создают новую гипотезу.
Сравнивать данные (SHA256), признаки/предобработку, модель/параметры, split и смысл изменения. Технический full refit для итоговых предсказаний не является новой проверкой и не даёт новой метрики.
На платформе эта же память — постоянный hypotheses.json рядом с task.md: агент читает его непосредственно перед каждым допуском кандидата, записывает started до fit и результат после. История разделена по SHA данных/split/метрике и читается при следующем запуске.

## Данные и протоколы

- Регрессия train SHA256: `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`; outer: Year_Factor<5→5 (35000/18308).
- Классификация train SHA256: `171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`; outer: SGKF5 first fold, seed42, группировка одинаковых сырых признаков (70529/17141).
- Старые классификационные random-split scores несопоставимы с grouped. Остальные SHA и параметры: [история](results/history/2026-10-21/historical-experiments.json), evidence.json/metrics.json каждого run.

## Уже проверенные направления — закрыты для повторов

| ID | Изменение | Статус / вывод | Подробности |
| --- | --- | --- | --- |
| HYP-001 | Парные Pearson/Spearman корреляции, порог0.95 | Проверена; улучшения зависят от split | [гипотеза](docs/hypotheses/HYP-001.md) |
| HYP-002 | VIF>10, joint collinearity | Проверена; сильное ухудшение regression outer | [гипотеза](docs/hypotheses/HYP-002.md) |
| HYP-003 | StandardScaler после отбора | Проверена; систематической пользы деревьям нет | [гипотеза](docs/hypotheses/HYP-003.md) |
| HYP-004 | Yeo–Johnson для непрерывных столбцов | Проверена; нормальность всех данных не гарантирована | [гипотеза](docs/hypotheses/HYP-004.md) |
| HYP-005 | Полный cumulative preprocessing + engineering | Проверена; не лучший outer | [гипотеза](docs/hypotheses/HYP-005.md) |
| HYP-006 | CatBoost l2_leaf_reg1/3/10/30 | Проверена; inner выбрал1, не гарантирует лучший outer | [гипотеза](docs/hypotheses/HYP-006.md) |
| HYP-007 | L1/L2: Lasso/Ridge и LogisticRegression | Проверена; L1 лучше в проверенной сетке, classification разница мала | [гипотеза](docs/hypotheses/HYP-007.md) |
| HYP-008 | Engineering без pruning/scaling/Gaussian | Проверена; classification RF0.577365824 | [гипотеза](docs/hypotheses/HYP-008.md) |
| HYP-009 | Нативный категориальный LightGBM500/.05/31/20/lambda1 | Проверена; новый лучший grouped classifier0.583855959 | [гипотеза](docs/hypotheses/HYP-009.md) |
| HYP-010 | ExtraTrees200/depth24/leaf2/features.8 | Проверена; regression43.567567738, class0.580495276 | [гипотеза](docs/hypotheses/HYP-010.md) |

Также проверены в старой истории: линейные/RF/CatBoost baselines, Optuna16 trials, balanced RF, нативные категории CatBoost, missing flags, year_built0→NaN, четыре flow-ratios с+1, равные ансамбли и другие splits. Их не повторять. Все записи ниже импортированы, без новых fit.

## Исторические оценки — 33 записи и 16 Optuna trials

| ID | Задача | Изменение / параметры | Split | Метрика | Источник |
| --- | --- | --- | --- | --- | --- |
| constant-regression-reference | Регрессия | Постоянное среднее train; strategy=mean | Year_Factor < 5 train; == 5 validation | 56.79943183188593 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| constant-multiclass-reference | Классификация | Всегда самый частый класс train; strategy=most_frequent | stratified 80/20; random_state=42 | 0.048413590386444195 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| regression-20261021-195542 | Регрессия | Первый линейный ориентир; Ridge; alpha=1; solver=lsqr; tol=1e-6 (восстановлено по коду первого этапа) | Year_Factor < 5 train; == 5 validation | 50.079635494578994 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| multiclass-20261021-195652 | Классификация | Первый линейный классификатор; solver=lbfgs; max_iter=3000; random_state=42 (по коду первого этапа) | stratified 80/20; random_state=42 | 0.4288518821725383 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| regression-forest-20261021-stage2 | Регрессия | Замена линейной модели на Random Forest; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8 | Year_Factor < 5 train; == 5 validation | 47.470727845698555 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| multiclass-forest-20261021-stage2 | Классификация | Замена линейной модели на Random Forest; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | 0.628538470153851 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| regression-boosting-20261021-stage3 | Регрессия | Замена модели на CatBoost; категории OneHotEncoder; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 45.93583634208973 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| multiclass-boosting-20261021-stage3 | Классификация | Замена модели на CatBoost; категории OneHotEncoder; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | stratified 80/20; random_state=42 | 0.5340999478722075 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| regression-optuna-20261021-stage4-candidate | Регрессия | Лучший по внутренней проверке кандидат Optuna, до решения о сохранении; {"depth": 8, "iterations": 700, "l2_leaf_reg": 3.7888252014533643, "learning_rate": 0.10390991237859405} | Year_Factor < 5 train; == 5 validation | 46.924137940373335 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| regression-optuna-20261021-stage4 | Регрессия | Итог подбора: сохранена прежняя модель; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 45.93583634208973 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| multiclass-optuna-20261021-stage4-candidate | Классификация | Лучший по внутренней проверке кандидат Optuna, до решения о сохранении; {"max_depth": 28, "max_features": 0.8, "min_samples_leaf": 4, "n_estimators": 100} | stratified 80/20; random_state=42 | 0.6203523622860305 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| multiclass-optuna-20261021-stage4 | Классификация | Итог подбора: сохранена прежняя модель; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | 0.628538470153851 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| C00-forest-control | Классификация | Повтор прежнего Random Forest; контроль воспроизводимости; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | 0.628538470153851 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H1-balanced-forest | Классификация | Только class_weight=balanced; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=balanced | stratified 80/20; random_state=42 | 0.6133163856531635 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| R00-catboost-control | Регрессия | Повтор CatBoost с OneHotEncoder; контроль; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 45.93583634208973 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H2-native-regression | Регрессия | Нативные категории CatBoost вместо OneHotEncoder; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 51.31822671778022 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| C00-catboost-control | Классификация | Повтор CatBoost с OneHotEncoder; контроль; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | stratified 80/20; random_state=42 | 0.5340999478722075 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H2-native-multiclass | Классификация | Нативные категории CatBoost вместо OneHotEncoder; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | stratified 80/20; random_state=42 | 0.5417317719703455 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H3-missing-flags | Регрессия | Шесть индикаторов пропусков; прочее без изменений; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 46.80246419128978 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H4-zero-year | Регрессия | Только year_built=0 заменён на пропуск; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 43.0046386479306 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H5-flow-ratios | Классификация | Четыре отношения объёмов/пакетов, знаменатель +1; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | 0.628620730127727 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| R00-forest-control | Регрессия | Повтор прежнего Random Forest для ансамбля; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8 | Year_Factor < 5 train; == 5 validation | 47.470727845698555 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H6-ensemble-regression | Регрессия | Среднее прогнозов исходных леса и CatBoost, веса 50/50; equal_weights=[0.5, 0.5] | Year_Factor < 5 train; == 5 validation | 44.61680775576098 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| H6-ensemble-multiclass | Классификация | Среднее вероятностей исходных леса и CatBoost, веса 50/50; equal_weights=[0.5, 0.5] | stratified 80/20; random_state=42 | 0.5788370836114318 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| S-regression-control | Регрессия | Контроль на дополнительном разделении; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 4 train; == 4 validation | 45.51876878464039 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| S-regression-candidate | Регрессия | Кандидат H4-zero-year на дополнительном разделении; iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 4 train; == 4 validation | 45.31673513065655 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| S-multiclass-control | Классификация | Контроль на дополнительном разделении; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=2026 | 0.6082047994072377 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| S-multiclass-candidate | Классификация | Кандидат H5-flow-ratios на дополнительном разделении; n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=2026 | 0.6013008205545127 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| diagnostic-ols-regression | Регрессия | LinearRegression; LinearRegression(); настройки по умолчанию, диагностический запуск | Year_Factor < 5 train; == 5 validation | 10615239987723.564 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| technical-repository-check-multiclass | Классификация | Первый линейный классификатор; solver=lbfgs; max_iter=3000; random_state=42 (по коду первого этапа) | stratified 80/20; random_state=42 | 0.4288518821725383 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| technical-repository-check-regression | Регрессия | Первый линейный ориентир; Ridge; alpha=1; solver=lsqr; tol=1e-6 (восстановлено по коду первого этапа) | Year_Factor < 5 train; == 5 validation | 50.079635494578994 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| technical-verified-multiclass | Классификация | Первый линейный классификатор; solver=lbfgs; max_iter=3000; random_state=42 (по коду первого этапа) | stratified 80/20; random_state=42 | 0.4288518821725383 | [JSON](results/history/2026-10-21/historical-experiments.json) |
| technical-verified-regression | Регрессия | Первый линейный ориентир; Ridge; alpha=1; solver=lsqr; tol=1e-6 (восстановлено по коду первого этапа) | Year_Factor < 5 train; == 5 validation | 50.079635494578994 | [JSON](results/history/2026-10-21/historical-experiments.json) |

Optuna: все16 параметрических попыток сохранены в разделе «Optuna» [истории](results/history/2026-10-21/historical-experiments.json); новое название trial не разрешает повтор.

## SPEC-002: все завершённые и неуспешные оценки

| Run | Конфигурация | Часть | Метрики | Статус | Результат |
| --- | --- | --- | --- | --- | --- |
| HYP-001-01-regression-inner-baseline-catboost | {"family": "catboost", "recipe": "baseline"} | inner | {"rmse": 45.518715477760836} | completed | [отчёт](results/2026-10-23/HYP-001-01-regression-inner-baseline-catboost/report.md) |
| HYP-001-02-regression-inner-correlation-catboost | {"family": "catboost", "recipe": "correlation"} | inner | {"rmse": 45.32052344905627} | completed | [отчёт](results/2026-10-23/HYP-001-02-regression-inner-correlation-catboost/report.md) |
| HYP-001-03-regression-inner-baseline-forest | {"family": "forest", "recipe": "baseline"} | inner | {"rmse": 46.1639294892355} | completed | [отчёт](results/2026-10-23/HYP-001-03-regression-inner-baseline-forest/report.md) |
| HYP-001-04-regression-inner-correlation-forest | {"family": "forest", "recipe": "correlation"} | inner | {"rmse": 46.36248309741696} | completed | [отчёт](results/2026-10-23/HYP-001-04-regression-inner-correlation-forest/report.md) |
| HYP-001-05-multiclass-inner-baseline-catboost | {"family": "catboost", "recipe": "baseline"} | inner | {"macro_f1": 0.5283017428570587, "accuracy": 0.8088028367197614, "weighted_f1": 0.7860891259009041} | completed | [отчёт](results/2026-10-23/HYP-001-05-multiclass-inner-baseline-catboost/report.md) |
| HYP-001-06-regression-outer-baseline-catboost | {"family": "catboost", "recipe": "baseline"} | outer | {"rmse": 45.93583634208973} | completed | [отчёт](results/2026-10-23/HYP-001-06-regression-outer-baseline-catboost/report.md) |
| HYP-001-07-regression-outer-baseline-forest | {"family": "forest", "recipe": "baseline"} | outer | {"rmse": 48.565980842466} | completed | [отчёт](results/2026-10-23/HYP-001-07-regression-outer-baseline-forest/report.md) |
| HYP-001-08-multiclass-inner-correlation-catboost | {"family": "catboost", "recipe": "correlation"} | inner | {"macro_f1": 0.5132340783634113, "accuracy": 0.8030618562503518, "weighted_f1": 0.7798450938735185} | completed | [отчёт](results/2026-10-23/HYP-001-08-multiclass-inner-correlation-catboost/report.md) |
| HYP-001-09-multiclass-inner-baseline-forest | {"family": "forest", "recipe": "baseline"} | inner | {"macro_f1": 0.6023505992999411, "accuracy": 0.8206225023920752, "weighted_f1": 0.8055785391931077} | completed | [отчёт](results/2026-10-23/HYP-001-09-multiclass-inner-baseline-forest/report.md) |
| HYP-001-10-multiclass-inner-correlation-forest | {"family": "forest", "recipe": "correlation"} | inner | {"macro_f1": 0.5614586103126855, "accuracy": 0.8122361681769573, "weighted_f1": 0.796936329997139} | completed | [отчёт](results/2026-10-23/HYP-001-10-multiclass-inner-correlation-forest/report.md) |
| HYP-001-11-multiclass-outer-baseline-catboost | {"family": "catboost", "recipe": "baseline"} | outer | {"macro_f1": 0.5239534323156312, "accuracy": 0.8157633743655562, "weighted_f1": 0.7922194332917473} | completed | [отчёт](results/2026-10-23/HYP-001-11-multiclass-outer-baseline-catboost/report.md) |
| HYP-001-12-multiclass-outer-baseline-forest | {"family": "forest", "recipe": "baseline"} | outer | {"macro_f1": 0.572183507286843, "accuracy": 0.8278980222857476, "weighted_f1": 0.8157534962058391} | completed | [отчёт](results/2026-10-23/HYP-001-12-multiclass-outer-baseline-forest/report.md) |
| HYP-002-01-regression-inner-vif-catboost | {"family": "catboost", "recipe": "vif"} | inner | {"rmse": 45.000826848085644} | completed | [отчёт](results/2026-10-23/HYP-002-01-regression-inner-vif-catboost/report.md) |
| HYP-002-02-regression-inner-vif-forest | {"family": "forest", "recipe": "vif"} | inner | {"rmse": 46.68533131938694} | completed | [отчёт](results/2026-10-23/HYP-002-02-regression-inner-vif-forest/report.md) |
| HYP-002-03-regression-outer-vif-catboost | {"family": "catboost", "recipe": "vif"} | outer | {"rmse": 54.62842812078608} | completed | [отчёт](results/2026-10-23/HYP-002-03-regression-outer-vif-catboost/report.md) |
| HYP-002-04-multiclass-inner-vif-catboost | {"family": "catboost", "recipe": "vif"} | inner | {"macro_f1": 0.5132340783634113, "accuracy": 0.8030618562503518, "weighted_f1": 0.7798450938735185} | completed | [отчёт](results/2026-10-23/HYP-002-04-multiclass-inner-vif-catboost/report.md) |
| HYP-002-05-multiclass-inner-vif-forest | {"family": "forest", "recipe": "vif"} | inner | {"macro_f1": 0.5614586103126855, "accuracy": 0.8122361681769573, "weighted_f1": 0.796936329997139} | completed | [отчёт](results/2026-10-23/HYP-002-05-multiclass-inner-vif-forest/report.md) |
| HYP-003-01-regression-inner-scaled-catboost | {"family": "catboost", "recipe": "scaled"} | inner | {"rmse": 45.00084665392861} | completed | [отчёт](results/2026-10-23/HYP-003-01-regression-inner-scaled-catboost/report.md) |
| HYP-003-02-regression-inner-scaled-forest | {"family": "forest", "recipe": "scaled"} | inner | {"rmse": 46.67970603881967} | completed | [отчёт](results/2026-10-23/HYP-003-02-regression-inner-scaled-forest/report.md) |
| HYP-003-03-multiclass-inner-scaled-catboost | {"family": "catboost", "recipe": "scaled"} | inner | {"macro_f1": 0.5107662999640583, "accuracy": 0.802724151516857, "weighted_f1": 0.7787386827217957} | completed | [отчёт](results/2026-10-23/HYP-003-03-multiclass-inner-scaled-catboost/report.md) |
| HYP-003-04-multiclass-inner-scaled-forest | {"family": "forest", "recipe": "scaled"} | inner | {"macro_f1": 0.554820917280919, "accuracy": 0.8147689536781674, "weighted_f1": 0.7987640272318586} | completed | [отчёт](results/2026-10-23/HYP-003-04-multiclass-inner-scaled-forest/report.md) |
| HYP-004-01-regression-inner-gaussian-catboost | {"family": "catboost", "recipe": "gaussian"} | inner | {"rmse": 45.12283523338561} | completed | [отчёт](results/2026-10-23/HYP-004-01-regression-inner-gaussian-catboost/report.md) |
| HYP-004-02-regression-inner-gaussian-forest | {"family": "forest", "recipe": "gaussian"} | inner | {"rmse": 46.71183260529205} | completed | [отчёт](results/2026-10-23/HYP-004-02-regression-inner-gaussian-forest/report.md) |
| HYP-004-03-multiclass-inner-gaussian-catboost | {"family": "catboost", "recipe": "gaussian"} | inner | {"macro_f1": 0.5128290768670659, "accuracy": 0.8033432768615973, "weighted_f1": 0.7796028566047394} | completed | [отчёт](results/2026-10-23/HYP-004-03-multiclass-inner-gaussian-catboost/report.md) |
| HYP-004-04-multiclass-inner-gaussian-forest | {"family": "forest", "recipe": "gaussian"} | inner | {"macro_f1": 0.5629527561008232, "accuracy": 0.81460010131142, "weighted_f1": 0.7985038277657371} | completed | [отчёт](results/2026-10-23/HYP-004-04-multiclass-inner-gaussian-forest/report.md) |
| HYP-005-01-regression-inner-engineered-catboost | {"family": "catboost", "recipe": "engineered"} | inner | {"rmse": 45.26398406382808} | completed | [отчёт](results/2026-10-23/HYP-005-01-regression-inner-engineered-catboost/report.md) |
| HYP-005-02-regression-inner-engineered-forest | {"family": "forest", "recipe": "engineered"} | inner | {"rmse": 46.94636373340831} | completed | [отчёт](results/2026-10-23/HYP-005-02-regression-inner-engineered-forest/report.md) |
| HYP-005-03-regression-outer-engineered-catboost | {"family": "catboost", "recipe": "engineered"} | outer | {"rmse": 46.99974168722143} | completed | [отчёт](results/2026-10-23/HYP-005-03-regression-outer-engineered-catboost/report.md) |
| HYP-005-04-regression-outer-engineered-forest | {"family": "forest", "recipe": "engineered"} | outer | {"rmse": 46.87784630178003} | completed | [отчёт](results/2026-10-23/HYP-005-04-regression-outer-engineered-forest/report.md) |
| HYP-005-06-multiclass-inner-engineered-catboost | {"family": "catboost", "recipe": "engineered"} | inner | {"macro_f1": 0.5128290768670659, "accuracy": 0.8033432768615973, "weighted_f1": 0.7796028566047394} | completed | [отчёт](results/2026-10-23/HYP-005-06-multiclass-inner-engineered-catboost/report.md) |
| HYP-005-07-multiclass-inner-engineered-forest | {"family": "forest", "recipe": "engineered"} | inner | {"macro_f1": 0.5629527561008232, "accuracy": 0.81460010131142, "weighted_f1": 0.7985038277657371} | completed | [отчёт](results/2026-10-23/HYP-005-07-multiclass-inner-engineered-forest/report.md) |
| HYP-005-09-multiclass-outer-engineered-catboost | {"family": "catboost", "recipe": "engineered"} | outer | {"macro_f1": 0.5075268759704035, "accuracy": 0.8052038970888513, "weighted_f1": 0.7803248081892851} | completed | [отчёт](results/2026-10-23/HYP-005-09-multiclass-outer-engineered-catboost/report.md) |
| HYP-005-10-multiclass-outer-engineered-forest | {"family": "forest", "recipe": "engineered"} | outer | {"macro_f1": 0.5594984742922693, "accuracy": 0.8220057172860393, "weighted_f1": 0.8088471303088757} | completed | [отчёт](results/2026-10-23/HYP-005-10-multiclass-outer-engineered-forest/report.md) |
| HYP-006-01-regression-inner-vif-catboost-1 | {"family": "catboost", "recipe": "vif", "strength": 1} | inner | {"rmse": 44.55882835878972} | completed | [отчёт](results/2026-10-23/HYP-006-01-regression-inner-vif-catboost-1/report.md) |
| HYP-006-02-regression-inner-vif-catboost-10 | {"family": "catboost", "recipe": "vif", "strength": 10} | inner | {"rmse": 45.603991717078685} | completed | [отчёт](results/2026-10-23/HYP-006-02-regression-inner-vif-catboost-10/report.md) |
| HYP-006-03-regression-inner-vif-catboost-30 | {"family": "catboost", "recipe": "vif", "strength": 30} | inner | {"rmse": 46.330352267082525} | completed | [отчёт](results/2026-10-23/HYP-006-03-regression-inner-vif-catboost-30/report.md) |
| HYP-006-04-regression-outer-vif-catboost-1 | {"family": "catboost", "recipe": "vif", "strength": 1} | outer | {"rmse": 52.94239384982671} | completed | [отчёт](results/2026-10-23/HYP-006-04-regression-outer-vif-catboost-1/report.md) |
| HYP-006-05-multiclass-inner-baseline-catboost-1 | {"family": "catboost", "recipe": "baseline", "strength": 1} | inner | {"macro_f1": 0.5418208400891047, "accuracy": 0.8097033826757472, "weighted_f1": 0.788208632780246} | completed | [отчёт](results/2026-10-23/HYP-006-05-multiclass-inner-baseline-catboost-1/report.md) |
| HYP-006-06-multiclass-inner-baseline-catboost-10 | {"family": "catboost", "recipe": "baseline", "strength": 10} | inner | {"macro_f1": 0.5095018586287922, "accuracy": 0.8075083019080317, "weighted_f1": 0.7822626380885134} | completed | [отчёт](results/2026-10-23/HYP-006-06-multiclass-inner-baseline-catboost-10/report.md) |
| HYP-006-07-multiclass-inner-baseline-catboost-30 | {"family": "catboost", "recipe": "baseline", "strength": 30} | inner | {"macro_f1": 0.48214773380255344, "accuracy": 0.8042438228175831, "weighted_f1": 0.7737147046898298} | completed | [отчёт](results/2026-10-23/HYP-006-07-multiclass-inner-baseline-catboost-30/report.md) |
| HYP-006-08-multiclass-outer-baseline-catboost-1 | {"family": "catboost", "recipe": "baseline", "strength": 1} | outer | {"macro_f1": 0.5270879065570541, "accuracy": 0.8150632985240067, "weighted_f1": 0.7912883328576368} | completed | [отчёт](results/2026-10-23/HYP-006-08-multiclass-outer-baseline-catboost-1/report.md) |
| HYP-007-01-regression-inner-engineered-l1-0.01 | {"family": "l1", "recipe": "engineered", "strength": 0.01} | inner | {"rmse": 611.1450350874579} | completed | [отчёт](results/2026-10-23/HYP-007-01-regression-inner-engineered-l1-0.01/report.md) |
| HYP-007-02-regression-inner-engineered-l1-0.1 | {"family": "l1", "recipe": "engineered", "strength": 0.1} | inner | {"rmse": 517.4284040167722} | completed | [отчёт](results/2026-10-23/HYP-007-02-regression-inner-engineered-l1-0.1/report.md) |
| HYP-007-03-regression-inner-engineered-l1-1 | {"family": "l1", "recipe": "engineered", "strength": 1} | inner | {"rmse": 54.19713785600558} | completed | [отчёт](results/2026-10-23/HYP-007-03-regression-inner-engineered-l1-1/report.md) |
| HYP-007-04-regression-inner-engineered-l2-0.1 | {"family": "l2", "recipe": "engineered", "strength": 0.1} | inner | {"rmse": 878.1770374661244} | completed | [отчёт](results/2026-10-23/HYP-007-04-regression-inner-engineered-l2-0.1/report.md) |
| HYP-007-05-regression-inner-engineered-l2-1 | {"family": "l2", "recipe": "engineered", "strength": 1} | inner | {"rmse": 884.6729602261103} | completed | [отчёт](results/2026-10-23/HYP-007-05-regression-inner-engineered-l2-1/report.md) |
| HYP-007-06-regression-inner-engineered-l2-10 | {"family": "l2", "recipe": "engineered", "strength": 10} | inner | {"rmse": 719.0247942438682} | completed | [отчёт](results/2026-10-23/HYP-007-06-regression-inner-engineered-l2-10/report.md) |
| HYP-007-07-regression-outer-engineered-l1-1 | {"family": "l1", "recipe": "engineered", "strength": 1} | outer | {"rmse": 58.664647652489585} | completed | [отчёт](results/2026-10-23/HYP-007-07-regression-outer-engineered-l1-1/report.md) |
| HYP-007-08-regression-outer-engineered-l2-10 | {"family": "l2", "recipe": "engineered", "strength": 10} | outer | {"rmse": 65.452603887892} | completed | [отчёт](results/2026-10-23/HYP-007-08-regression-outer-engineered-l2-10/report.md) |
| HYP-007-09-multiclass-inner-engineered-l1-0.01 | {"family": "l1", "recipe": "engineered", "strength": 0.01} | inner | {"macro_f1": 0.37638188670830286, "accuracy": 0.7522935779816514, "weighted_f1": 0.7286123191429368} | completed | [отчёт](results/2026-10-23/HYP-007-09-multiclass-inner-engineered-l1-0.01/report.md) |
| HYP-007-10-multiclass-inner-engineered-l1-0.1 | {"family": "l1", "recipe": "engineered", "strength": 0.1} | inner | {"macro_f1": 0.3755980101778375, "accuracy": 0.759047672651545, "weighted_f1": 0.7312061160286761} | completed | [отчёт](results/2026-10-23/HYP-007-10-multiclass-inner-engineered-l1-0.1/report.md) |
| HYP-007-11-multiclass-inner-engineered-l1-1 | {"family": "l1", "recipe": "engineered", "strength": 1} | inner | {"macro_f1": 0.3775319090469025, "accuracy": 0.759047672651545, "weighted_f1": 0.7325445954190413} | completed | [отчёт](results/2026-10-23/HYP-007-11-multiclass-inner-engineered-l1-1/report.md) |
| HYP-007-12-multiclass-inner-engineered-l2-0.01 | {"family": "l2", "recipe": "engineered", "strength": 0.01} | inner | {"macro_f1": 0.3762205998776406, "accuracy": 0.7576968537175662, "weighted_f1": 0.7311719750658902} | completed | [отчёт](results/2026-10-23/HYP-007-12-multiclass-inner-engineered-l2-0.01/report.md) |
| HYP-007-13-multiclass-inner-engineered-l2-0.1 | {"family": "l2", "recipe": "engineered", "strength": 0.1} | inner | {"macro_f1": 0.37751467232812563, "accuracy": 0.759047672651545, "weighted_f1": 0.7325440492227051} | completed | [отчёт](results/2026-10-23/HYP-007-13-multiclass-inner-engineered-l2-0.1/report.md) |
| HYP-007-14-multiclass-inner-engineered-l2-1 | {"family": "l2", "recipe": "engineered", "strength": 1} | inner | {"macro_f1": 0.3775030371857732, "accuracy": 0.759047672651545, "weighted_f1": 0.732542202308974} | completed | [отчёт](results/2026-10-23/HYP-007-14-multiclass-inner-engineered-l2-1/report.md) |
| HYP-007-15-multiclass-outer-engineered-l1-1 | {"family": "l1", "recipe": "engineered", "strength": 1} | outer | {"macro_f1": 0.3716414811191194, "accuracy": 0.7544483985765125, "weighted_f1": 0.7316911360704531} | completed | [отчёт](results/2026-10-23/HYP-007-15-multiclass-outer-engineered-l1-1/report.md) |
| HYP-007-16-multiclass-outer-engineered-l2-0.1 | {"family": "l2", "recipe": "engineered", "strength": 0.1} | outer | {"macro_f1": 0.3715427929049082, "accuracy": 0.7543317192695875, "weighted_f1": 0.731579052317665} | completed | [отчёт](results/2026-10-23/HYP-007-16-multiclass-outer-engineered-l2-0.1/report.md) |
| HYP-008-01-regression-inner-engineering_only-catboost | {"family": "catboost", "recipe": "engineering_only"} | inner | {"rmse": 44.976674584166} | completed | [отчёт](results/2026-10-23/HYP-008-01-regression-inner-engineering_only-catboost/report.md) |
| HYP-008-02-regression-inner-engineering_only-forest | {"family": "forest", "recipe": "engineering_only"} | inner | {"rmse": 46.12352152733562} | completed | [отчёт](results/2026-10-23/HYP-008-02-regression-inner-engineering_only-forest/report.md) |
| HYP-008-03-regression-outer-engineering_only-catboost | {"family": "catboost", "recipe": "engineering_only"} | outer | {"rmse": 45.204914878207624} | completed | [отчёт](results/2026-10-23/HYP-008-03-regression-outer-engineering_only-catboost/report.md) |
| HYP-008-04-regression-outer-engineering_only-forest | {"family": "forest", "recipe": "engineering_only"} | outer | {"rmse": 46.70829342223722} | completed | [отчёт](results/2026-10-23/HYP-008-04-regression-outer-engineering_only-forest/report.md) |
| HYP-008-05-multiclass-inner-engineering_only-catboost | {"family": "catboost", "recipe": "engineering_only"} | inner | {"macro_f1": 0.5265158705817645, "accuracy": 0.8094219620645016, "weighted_f1": 0.7851927470223338} | completed | [отчёт](results/2026-10-23/HYP-008-05-multiclass-inner-engineering_only-catboost/report.md) |
| HYP-008-06-multiclass-inner-engineering_only-forest | {"family": "forest", "recipe": "engineering_only"} | inner | {"macro_f1": 0.594619984096699, "accuracy": 0.8203410817808297, "weighted_f1": 0.8055492959128074} | completed | [отчёт](results/2026-10-23/HYP-008-06-multiclass-inner-engineering_only-forest/report.md) |
| HYP-008-07-multiclass-outer-engineering_only-catboost | {"family": "catboost", "recipe": "engineering_only"} | outer | {"macro_f1": 0.5239293879168748, "accuracy": 0.8153549967913191, "weighted_f1": 0.7901538667092455} | completed | [отчёт](results/2026-10-23/HYP-008-07-multiclass-outer-engineering_only-catboost/report.md) |
| HYP-008-08-multiclass-outer-engineering_only-forest | {"family": "forest", "recipe": "engineering_only"} | outer | {"macro_f1": 0.5773658239122667, "accuracy": 0.8277813429788227, "weighted_f1": 0.8163534309813834} | completed | [отчёт](results/2026-10-23/HYP-008-08-multiclass-outer-engineering_only-forest/report.md) |

## SPEC-003: только новые семейства — четыре проверки

| Run | Данные | Модель и параметры | Метрика | Статус | Результат |
| --- | --- | --- | --- | --- | --- |
| HYP-009-01-regression-lightgbm | regression | lightgbm: 500trees, lr.05, leaves31, min_child20, lambda1; native categories | rmse=60.492688256314 | completed | [отчёт](results/2026-10-23/HYP-009-01-regression-lightgbm/report.md) |
| HYP-010-01-regression-extra_trees | regression | extra_trees: 200trees, depth24, min_leaf2, features.8; median/OHE | rmse=43.567567738320 | completed | [отчёт](results/2026-10-23/HYP-010-01-regression-extra_trees/report.md) |
| HYP-009-02-multiclass-lightgbm | multiclass | lightgbm: 500trees, lr.05, leaves31, min_child20, lambda1; native categories | macro_f1=0.583855959368 | completed | [отчёт](results/2026-10-23/HYP-009-02-multiclass-lightgbm/report.md) |
| HYP-010-02-multiclass-extra_trees | multiclass | extra_trees: 200trees, depth24, min_leaf2, features.8; median/OHE | macro_f1=0.580495275694 | completed | [отчёт](results/2026-10-23/HYP-010-02-multiclass-extra_trees/report.md) |

## Текущий выбор

Исторический выбор SPEC-003: CatBoost H4 RMSE43.004639 / LightGBM F1 0.583856. Позже SPEC-004 улучшила их до42.196700/0.591451; обязательная full-v1 SPEC-005 дала43.837202/0.563787 (ухудшение, сохраняется по прямому требованию обработки). Старые validation-проверки не переобучались.
[Новые выводы](results/2026-10-23/NEW_EXPERIMENT_REPORT.md), [основная серия](results/2026-10-23/EXPERIMENT_REPORT.md), [журнал артефактов](results/README.md).

## Следующие гипотезы

**HYP-021** была запланирована до новых fit и теперь проверена: [полная mandatory full-v1 обработка и до5% training-выбросов](docs/hypotheses/HYP-021.md), [SPEC-005](docs/specs/SPEC-005-agent-preprocessing.md). Две составные проверки против сохранённых победителей HYP-012/HYP-011, без повторов baseline.

Пользователь поручил проверить все десять предложений. До первого fit зафиксированы [SPEC-004](docs/specs/SPEC-004-ten-new-hypotheses.md) и [PLAN-004](docs/plans/PLAN-004-ten-new-hypotheses.md); это11 новых evaluation, старые контроли не переобучаются. Подробности и критерии — [HYP-011…020](docs/hypotheses/HYP-011-020-proposals.md). Перед каждым опытом runner перечитывает этот журнал и реестр попыток; результаты дописываются ниже сразу после завершения.

| ID | Предложенный новый механизм | Статус |
| --- | --- | --- |
| HYP-011 | XGBoost histogram, обе задачи | Проверена: SPEC-004; результаты ниже |
| HYP-012 | log1p таргета регрессии и обратное преобразование | Проверена: SPEC-004; результаты ниже |
| HYP-013 | Tweedie objective для регрессии | Проверена: SPEC-004; результаты ниже |
| HYP-014 | Линейные листья LightGBM | Проверена: SPEC-004; результаты ниже |
| HYP-015 | Out-of-fold target encoding категорий | Проверена: SPEC-004; результаты ниже |
| HYP-016 | Категориальная комбинация proto×service×state | Проверена: SPEC-004; результаты ниже |
| HYP-017 | Train-only частотные признаки сетевых категорий | Проверена: SPEC-004; результаты ниже |
| HYP-018 | Multiclass one-vs-rest | Проверена: SPEC-004; результаты ниже |
| HYP-019 | Иерархия «норма / атака → тип атаки» | Проверена: SPEC-004; результаты ниже |
| HYP-020 | Веса1/размер группы точных X-дубликатов | Проверена: SPEC-004; результаты ниже |

- SPEC-004 2026-10-23T21:44:51.993508+03:00: **HYP-011/regression**, completed, rmse=43.510161481164815; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-011-01-regression-none/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:46:16.240403+03:00: **HYP-011/multiclass**, completed, macro_f1=0.5914513709191194; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-011-02-multiclass-none/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:46:45.777312+03:00: **HYP-012/regression**, completed, rmse=42.19670002719412; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-012-01-regression-log1p_target/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:46:51.297017+03:00: **HYP-013/regression**, completed, rmse=47.18831919578777; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-013-01-regression-none/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:46:58.232957+03:00: **HYP-014/regression**, completed, rmse=178.08498786616641; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-014-01-regression-linear_scaled/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:47:15.245642+03:00: **HYP-015/regression**, completed, rmse=44.402896602005654; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-015-01-regression-target_encode/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:47:46.834150+03:00: **HYP-016/multiclass**, completed, macro_f1=0.5807635957092588; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-016-01-multiclass-categorical_cross/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:48:06.952170+03:00: **HYP-017/multiclass**, completed, macro_f1=0.5909684499124109; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-017-01-multiclass-frequency/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:48:26.272481+03:00: **HYP-018/multiclass**, completed, macro_f1=0.578639858804918; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-018-01-multiclass-none/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:48:48.104576+03:00: **HYP-019/multiclass**, completed, macro_f1=0.5901110420481382; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-019-01-multiclass-hierarchical/report.md). Повтор не запускать.

- SPEC-004 2026-10-23T21:49:20.373857+03:00: **HYP-020/multiclass**, completed, macro_f1=0.5829349072076985; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-020-01-multiclass-inverse_group_weights/report.md). Повтор не запускать.

- HYP-021/regression: completed, rmse=43.837201829105254; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-021-01-regression-full-v1/report.md). Повтор не запускать.

- HYP-021/multiclass: completed, macro_f1=0.5637871673383791; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-021-02-multiclass-full-v1/report.md). Повтор не запускать.

## HYP-022…041 — двадцать новых гипотез, завершены

2026-10-23: пользователь попросил20 новых гипотез и рассмотреть отказ от вредной обработки/другой подход. Прочитаны журнал, история и ранее реализованные признаки; предложения и границы новизны сохранены в [HYP-022…041](docs/hypotheses/HYP-022-041-proposals.md). Статус каждой: **Предложена; не запускалась**; вывод: **Не оценивалась**. Поручения на обучение этой серии пока нет. Это не новые результаты и не изменение действующего ZIP.

| ID | Новый механизм | Задача |
| --- | --- | --- |
| HYP-022 | Winsorization X вместо удаления строк | обе |
| HYP-023 | Многомерная IterativeImputer imputation | регрессия |
| HYP-024 | Заполнение пропусков медианами сопоставимых групп | регрессия |
| HYP-025 | Gaussian transform условно по proto | классификация |
| HYP-026 | Нулевая масса отдельно от Gaussian ненулевой части | классификация |
| HYP-027 | Структурная неприменимость TCP-полей | классификация |
| HYP-028 | Устойчивый отбор по inner permutation importance | обе |
| HYP-029 | sin/cos направлений ветра | регрессия |
| HYP-030 | Гармоники сезонной температурной кривой | регрессия |
| HYP-031 | Отклонение X от медианы/IQR сопоставимых зданий | регрессия |
| HYP-032 | Взаимодействия тепловой нагрузки, площади и типа | регрессия |
| HYP-033 | Временная OOF smearing-коррекция log-target | регрессия |
| HYP-034 | Pseudo-Huber loss вместо squared error | регрессия |
| HYP-035 | Частично объединённые эксперты по building_class | регрессия |
| HYP-036 | Экспоненциальные веса по давности года | регрессия |
| HYP-037 | Коэффициенты решающего правила под OOF Macro-F1 | классификация |
| HYP-038 | Ограниченная SMOTENC синтетика mixed-type классов | классификация |
| HYP-039 | Веса по индивидуальной OOF-трудности | классификация |
| HYP-040 | Локальный эксперт для путаемой пары атак | классификация |
| HYP-041 | Inner-выбор preprocessing×model вместо mandatory full-v1 | обе |

Цель нового рекорда: RMSE<42.19670002719412 / grouped Macro-F1>0.5914513709191194. Восстановление качества лишь относительно ухудшившейся full-v1 не выдавать за победу над лучшими гипотезами. Конкретный anchor/configuration/бюджет зафиксировать в отдельной карточке и плане до будущего fit; все learned choices только на inner, outer неизменён, старые контроли не переобучать.

Уточнение2026-10-23: после предложений пользователь прямо поручил проверить **все20** и обязательно записывать в проверенные; прежнее замечание «поручения пока нет» выше историческое. Разрешены23оценки по [SPEC-006](docs/specs/SPEC-006-twenty-hypotheses.md) и [PLAN-006](docs/plans/PLAN-006-twenty-hypotheses.md). До появления результата статус «Запланирована», а не «Проверена». Нормализация/фильтрация могут быть вариантами новойэкспериментальнойполитики; действующийZIP/productionagentнеизменён. Каждыйисходбудетдописансразуниже,включаяошибки.

- SPEC-006 2026-10-23T22:49:41.469788+03:00: **HYP-022/regression**: completed, rmse=42.203267904780695; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-022-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T22:50:22.578970+03:00: **HYP-022/multiclass**: completed, macro_f1=0.5615525355321143; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-022-02-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T22:50:44.666196+03:00: **HYP-023/regression**: completed, rmse=43.19210947337134; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-023-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T22:51:20.837139+03:00: **HYP-024/regression**: completed, rmse=42.18354628369281; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-024-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T22:52:22.947236+03:00: **HYP-025/multiclass**: completed, macro_f1=0.5771726314994396; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-025-01-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T22:53:13.774215+03:00: **HYP-026/multiclass**: completed, macro_f1=0.5793987048589155; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-026-01-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T22:54:02.507795+03:00: **HYP-027/multiclass**: completed, macro_f1=0.5857385909681803; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-027-01-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T22:54:50.535524+03:00: **HYP-028/regression**: failed, None=None; Недостаточно данных; [отчёт](results/2026-10-23/HYP-028-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:02:14.578232+03:00: **HYP-028/multiclass**: failed, None=None; Недостаточно данных; [отчёт](results/2026-10-23/HYP-028-02-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:02:39.422870+03:00: **HYP-029/regression**: completed, rmse=42.08610226703874; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-029-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:03:02.093548+03:00: **HYP-030/regression**: completed, rmse=41.993466003366024; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-030-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:03:30.279103+03:00: **HYP-031/regression**: completed, rmse=42.00460393727476; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-031-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:04:03.798131+03:00: **HYP-032/regression**: completed, rmse=43.810373477573165; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-032-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:04:07.387733+03:00: **HYP-033/regression**: completed, rmse=42.974266298635236; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-033-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:04:17.991503+03:00: **HYP-034/regression**: completed, rmse=44.42393495521875; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-034-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:04:29.420898+03:00: **HYP-035/regression**: completed, rmse=41.480963027907194; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-035-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:04:49.455906+03:00: **HYP-036/regression**: completed, rmse=42.25509131431627; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-036-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:05:11.200813+03:00: **HYP-037/multiclass**: completed, macro_f1=0.6286047899141216; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-037-01-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:06:51.554022+03:00: **HYP-038/multiclass**: completed, macro_f1=0.595178021245499; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-038-01-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:07:41.727174+03:00: **HYP-039/multiclass**: completed, macro_f1=0.5795834428470248; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-039-01-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:07:51.965916+03:00: **HYP-040/multiclass**: completed, macro_f1=0.5922402460598002; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-040-01-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:08:21.196287+03:00: **HYP-041/regression**: completed, rmse=42.19670002719412; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-041-01-regression-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 2026-10-23T23:11:01.808913+03:00: **HYP-041/multiclass**: completed, macro_f1=0.5914513709191194; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-041-02-multiclass-spec006/report.md). Проверка завершена; повтор не запускать.

- SPEC-006 HYP-028/regression, техническое продолжение безповтораinner: completed, rmse=42.245303736669484; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-028-03-regression-guard-continuation/report.md). Повтор не запускать.

- SPEC-006 HYP-028/multiclass, техническое продолжение безповтораinner: completed, macro_f1=0.5676156935900548; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-028-04-multiclass-guard-continuation/report.md). Повтор не запускать.

## Итог SPEC-006 — проверенные двадцать гипотез

Все20 ID HYP-022…041 проверены:23сопоставимыхоценки,25запусковсучётом2техническихостановокH28. Обеостановкисохранены; продолженияиспользоваликэшбезповторныхinnerfit. Независимаясверка539/539, модельныхfit39, повторовfingerprint0. ЛучшиеH35:RMSE41.480963027907194 и H37:MacroF1.6286047899141216. [Полныйотчёт](results/2026-10-23/TWENTY_HYPOTHESES_REPORT.md), [CSV](results/2026-10-23/twenty-hypotheses-comparison.csv). Проверенноенеповторять.

По следующему поручению пользователя собирается улучшенныйZIP: [SPEC-007](docs/specs/SPEC-007-improved-delivery.md). Это перенос измеренныхстратегий и техническиеfullrefit, не новыеvalidationгипотезы.

## Поставка лучших измеренных стратегий — SPEC-007 завершена

Готов [solution_improved_metrics.zip](dist/solution_improved_metrics.zip): H35RMSE41.480963027907194 и H37Macro-F1.6286047899141216.126тестов,5mockсценариев,2fullrefitбезповтораvalidation. Все20проверокостаютсявистории; [проверкаZIP](results/2026-10-23/IMPROVED_DELIVERY.json).

## SPEC-008: семь новых составных гипотез — запланированы

Пользователь поручил радикальнее менять методы, объединить некоторые гипотезы и включать только улучшения в baseline; отрицательные записывать и не повторять. Доfitпрочитанаистория, [HYP-042…048](docs/hypotheses/HYP-042-048-combinations.md) и точныепараметры [SPEC-008](docs/specs/SPEC-008-combinations.md) зафиксированы. Все7пока **Запланированы / Не оценивались**, не являютсяповторамиstandaloneH29/30/31/35/37/38/17. Контроль41.480963027907194/.6286047899141216переиспользуетсябезbaselinefit.

| ID | Новое составное изменение | Задача |
| --- | --- | --- |
| HYP-042 | H35experts + H30season | regression |
| HYP-043 | H35experts + H29wind + H30season + H31relative | regression |
| HYP-044 | TemporalOOFconvexblend4сохранённыхкомпонентов | regression |
| HYP-045 | Target-tailgate + lowlog/highoriginal mixture | regression |
| HYP-046 | SMOTENC + OOFMacroF1decisionrule | multiclass |
| HYP-047 | XGB/frequencyLGB50:50 + OOFMacroF1decisionrule | multiclass |
| HYP-048 | OOFlogprobaweightedLRstackXGB/SMOTE/frequency | multiclass |

Исходы дописываются после каждого запуска; исходные outer остаются разведочными, качество на платформе неизвестно. При отсутствии новогоverifiedрекорда текущийbaseline не менять.

- SPEC-008 2026-10-23T23:40:36.833955+03:00: **HYP-042/regression**: completed, rmse=41.5048413147343; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-042-01-regression-combinations/report.md). Повтор не запускать.

- SPEC-008 2026-10-23T23:41:25.401839+03:00: **HYP-043/regression**: completed, rmse=40.972997661684175; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-043-01-regression-combinations/report.md). Повтор не запускать.

- SPEC-008 2026-10-23T23:42:15.506840+03:00: **HYP-044/regression**: completed, rmse=41.94416116050523; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-044-01-regression-combinations/report.md). Повтор не запускать.

- SPEC-008 2026-10-23T23:42:37.837530+03:00: **HYP-045/regression**: completed, rmse=41.406800898690484; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-045-01-regression-combinations/report.md). Повтор не запускать.

- SPEC-008 2026-10-23T23:45:52.471011+03:00: **HYP-046/multiclass**: completed, macro_f1=0.6154359833457845; Не подтверждена по заданному критерию; [отчёт](results/2026-10-23/HYP-046-01-multiclass-combinations/report.md). Повтор не запускать.

- SPEC-008 2026-10-23T23:47:00.603774+03:00: **HYP-047/multiclass**: completed, macro_f1=0.6336560111990485; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-047-01-multiclass-combinations/report.md). Повтор не запускать.

- SPEC-008 2026-10-23T23:47:36.058135+03:00: **HYP-048/multiclass**: completed, macro_f1=0.6495311620074445; Подтверждена в проверенных условиях; [отчёт](results/2026-10-23/HYP-048-01-multiclass-combinations/report.md). Повтор не запускать.

## SPEC009: альтернативы, первая волна

20новых proposed H49…68: [идеи](docs/hypotheses/HYP-049-068-proposals.md), пока не проверены. По поручению продолжитьиотойтиотbaseline исполняется [SPEC009](docs/specs/SPEC-009-alternative-models.md): HYP-049/051/053/054/055/056/058/060(двезадачи), плюс новаяHYP-069 строго-прошлаявременнаяпамятьphysicalprofilesбезID. Всего9ID/10оценок; остальныеидеиостаютсяproposed. Baseline40.972997661684175/.6495311620074445, sourcefreezeдоfit. H68диагностика отдельно: пользовательуточнилscore1376.762дляsolution_improved_metrics.zip, формула/runID/SHA платформынеизвестны.

- SPEC009 2026-10-24T00:10:22.039448+03:00: **HYP-069/regression**: completed, rmse=39.39219382766771; Подтверждена в проверенных условиях; [отчёт](results/2026-10-24/HYP-069-01-regression-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:11:33.702910+03:00: **HYP-049/multiclass**: completed, macro_f1=0.6149071035851954; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-049-01-multiclass-alternatives/report.md). Повтор запрещён.

- HYP-068: локальный диагностический разбор выполнен, model fits0, новых метрик0; Недостаточно данных, нетrunID/формулы/SHA платформы. [Отчёт](results/2026-10-24/HYP-068-01-platform-provenance/report.md). Не считатьопровержением модели; дополнятьлишьпри новыхсведениях.

### Явный реестр H49…68 (техническое уточнение ID)

Диапазон/сокращённая запись не заменяют явный ID для защиты. Статусы ниже — proposal на момент регистрации; фактические исходы смотреть в запусках/последующих строках.
- HYP-049: Сжатие H48 в один классификатор; предложена, исход только из отдельного отчёта.
- HYP-050: Focal loss в одной многоклассовой модели; предложена, исход только из отдельного отчёта.
- HYP-051: Balanced Softmax вместо синтетики; предложена, исход только из отдельного отчёта.
- HYP-052: Классификация по локальному смешанному расстоянию; предложена, исход только из отдельного отчёта.
- HYP-053: Нелинейный kernel без глубокого ансамбля; предложена, исход только из отдельного отчёта.
- HYP-054: Аддитивная регрессия на сплайнах; предложена, исход только из отдельного отчёта.
- HYP-055: Условные квантили для восстановления среднего; предложена, исход только из отдельного отчёта.
- HYP-056: Условная поправка остатка в исходных единицах; предложена, исход только из отдельного отчёта.
- HYP-057: Семантически ограниченная монотонная модель; предложена, исход только из отдельного отчёта.
- HYP-058: Веса переноса по распределению X; предложена, исход только из отдельного отчёта.
- HYP-059: Одна модель с защитой худшей среды; предложена, исход только из отдельного отчёта.
- HYP-060: Обучение с пропусками как модель отказа измерений; предложена, исход только из отдельного отчёта.
- HYP-061: Сглаживание меток по соседям без синтетических X; предложена, исход только из отдельного отчёта.
- HYP-062: Низкоранговые взаимодействия признаков; предложена, исход только из отдельного отчёта.
- HYP-063: Аудит близких, но не точных копий; предложена, исход только из отдельного отчёта.
- HYP-064: Выбор семейства на уровне новой задачи; предложена, исход только из отдельного отчёта.
- HYP-065: Инвариантность к транспорту схемы и единиц; предложена, исход только из отдельного отчёта.
- HYP-066: Исполнение с гарантированным резервом на результат; предложена, исход только из отдельного отчёта.
- HYP-067: Защита предсказаний вне обученной области; предложена, исход только из отдельного отчёта.
- HYP-068: Разложение платформенного результата по происхождению и отказам; предложена, исход только из отдельного отчёта.

- SPEC009 2026-10-24T00:13:32.490371+03:00: **HYP-051/multiclass**: completed, macro_f1=0.5632041224667611; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-051-01-multiclass-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:13:39.500202+03:00: **HYP-053/multiclass**: failed, None=None; Недостаточно данных; [отчёт](results/2026-10-24/HYP-053-01-multiclass-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:13:45.425693+03:00: **HYP-054/regression**: completed, rmse=44.388898407039605; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-054-01-regression-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:13:53.049617+03:00: **HYP-055/regression**: completed, rmse=47.44104483605545; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-055-01-regression-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:14:39.223158+03:00: **HYP-056/regression**: completed, rmse=43.063770695804266; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-056-01-regression-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:15:05.841449+03:00: **HYP-058/regression**: completed, rmse=42.36424466952146; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-058-01-regression-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:16:13.607711+03:00: **HYP-060/regression**: completed, rmse=42.46179807398836; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-060-01-regression-alternatives/report.md). Повтор запрещён.

- SPEC009 2026-10-24T00:17:48.826368+03:00: **HYP-060/multiclass**: completed, macro_f1=0.567987406329944; Не подтверждена по заданному критерию; [отчёт](results/2026-10-24/HYP-060-02-multiclass-alternatives/report.md). Повтор запрещён.

- HYP-065 диагностика:6/8выявиладвеошибки; послеотдельногоCSVcodecfix8/8green,publicdata/dtypes/splitстрогонеизменны,publicfits0. [Первыйотчёт](results/2026-10-24/HYP-065-01-schema-semantics/report.md), [fixverification](results/2026-10-24/HYP-065-02-schema-codec-fix/verification.json). Не новаямодельнаяметрика.

- Grill-me, 2026-10-24: завершён read-only аудит текущего решения и срезов сохранённых ошибок; 0 новых fit, H52/H62/H61 не запускались. [Отчёт и прямые решения пользователя Q1–Q5](results/2026-10-24/grill-current-solution/report.md). Лимит новой гипотезы 900 секунд, предобработка по данным/модели; разрешено одно дополнительное сравнение финалистов с текущим лидером. При компромиссе выигрыш строго >15%, ухудшение другой метрики строго <10%, проценты относительные; без просадки достаточно подтверждённого выигрыша. Конкретный план ожидает итогового подтверждения в запрошенном grilling.

- SPEC010, 2026-10-24, **до новых fit**: пользователь явно подтвердил итоговый план grill. Зарегистрированы [HYP-052](docs/hypotheses/HYP-052.md) и [HYP-062](docs/hypotheses/HYP-062.md) для новых frozen inner/confirmation протоколов. H52: exact25NN mixed capped L1+categorical; H62: rank8 FM squaredloss originalscale,20epochs Adam. На каждую до900сек; старый outer не переобучается. Однократный контроль H48/H69 на дополнительном confirmation прямо разрешён, без frozen meta-state чужого fold. Исходы пока не получены; перепись агента не считается победой.

- SPEC010 итог2026-10-24: **HYP-052 и HYP-062 не подтверждены по критерию улучшения**, повтор запрещён. На сопоставимых новых confirmation: H52 Macro-F1.4803098632106223 vs H48.6103340454456273 (−21.3038%); H62 RMSE58.58032538035888 vs H6942.46005247026025 (+37.9657%ошибка).113/113сверок,257unit,4cleanCLI; научные4newlearnerfits+17authorizedcontrolcomponents, старыйouter0refits. Прежний baseline/ZIPсохранён; новое исследовательское решение не продвинуто. [Итог](results/2026-10-24/series-clean-rewrite-v1/REPORT.md).

- SPEC010 2026-10-24T00:56:45.500264+03:00: **HYP-062/regression/inner**: completed, rmse=61.25577980475285; [отчёт](results/2026-10-24/HYP-062-01-regression-inner-rewrite/report.md). Новый механизм; параметры не выбирались по confirmation. Повтор этой проверки запрещён.

- SPEC010 2026-10-24T00:56:49.641920+03:00: **HYP-062/regression/confirmation**: completed, rmse=58.58032538035888; [отчёт](results/2026-10-24/HYP-062-02-regression-confirmation-rewrite/report.md). Новый механизм; параметры не выбирались по confirmation. Повтор этой проверки запрещён.

- SPEC010 2026-10-24T00:57:14.468637+03:00: **HYP-062/regression/control**: completed, rmse=42.46005247026025; [отчёт](results/2026-10-24/HYP-062-03-regression-control-rewrite/report.md). Это разрешённая сравнительная проверка старого лидера на новом confirmation, без frozen public meta-state. Повтор этой проверки запрещён.

- SPEC010 2026-10-24T00:59:06.521233+03:00: **HYP-052/multiclass/inner**: completed, macro_f1=0.459744100268895; [отчёт](results/2026-10-24/HYP-052-01-multiclass-inner-rewrite/report.md). Новый механизм; параметры не выбирались по confirmation. Повтор этой проверки запрещён.

- SPEC010 2026-10-24T01:01:14.673014+03:00: **HYP-052/multiclass/confirmation**: completed, macro_f1=0.4803098632106223; [отчёт](results/2026-10-24/HYP-052-02-multiclass-confirmation-rewrite/report.md). Новый механизм; параметры не выбирались по confirmation. Повтор этой проверки запрещён.

- SPEC010 2026-10-24T01:08:03.686011+03:00: **HYP-052/multiclass/control**: completed, macro_f1=0.6103340454456273; [отчёт](results/2026-10-24/HYP-052-03-multiclass-control-rewrite/report.md). Это разрешённая сравнительная проверка старого лидера на новом confirmation, без frozen public meta-state. Повтор этой проверки запрещён.

## SPEC011 зарегистрировано до realfit,2026-10-24

Пользователь дал3часа до04:15:07MSK и потребовал ОБЕметрики20%: RMSE≤31.51375506213417, MacroF1≥.7794373944089334. HYP-070 (temporalentitylinker+ratingdelta;inneralpha.5/.8/.95), HYP-071 (ratingtargetnormalizer+g²weightedET), HYP-072 (teacherresidualMLP exactsoftF1), HYP-073 (15-bitECOC), HYP-074 (contextualCatBoostresiduallogits) — новые механизмы, пока proposed/registered. [SPEC011](docs/specs/SPEC-011-three-hour-improvement.md), [regcards](docs/hypotheses/HYP-070-regression-research.md), [classcards](docs/hypotheses/HYP-072-classification-research.md). До900сек/hyp, source/data/splits/configfrozen, nooldbaselinefits. Innerclass RAWcachedfold0 — proxy, не fullH48validation; oldouteralreadyviewedexploratory. H52/H62 не повторять.

- SPEC011 2026-10-24T01:53:24.653979+03:00: **HYP-070/inner/low** completed; rmse=34.20894464756146; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-070-01-inner-low-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T01:53:28.936682+03:00: **HYP-070/inner/mid** completed; rmse=31.323933523598928; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-070-02-inner-mid-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T01:53:33.271713+03:00: **HYP-070/inner/high** completed; rmse=30.663255379972725; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-070-03-inner-high-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T01:53:49.523518+03:00: **HYP-072/inner/default** completed; macro_f1=0.6090378831239697; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-072-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T01:55:12.276645+03:00: **HYP-071/inner/default** completed; rmse=49.19259595219514; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-071-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T01:55:41.170632+03:00: **HYP-073/inner/default** completed; macro_f1=0.5803974034255268; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-073-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T01:56:31.254170+03:00: **HYP-070/outer/high** completed; rmse=34.318489927675174; reference=39.39219382766771; [отчёт](results/2026-10-24/HYP-070-04-outer-high-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T01:58:44.111277+03:00: **HYP-074/inner/default** completed; macro_f1=0.6143385510752261; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-074-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:01:05.171536+03:00: **HYP-075/inner/default** completed; macro_f1=0.6212181165985486; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-075-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:04:32.332095+03:00: **HYP-076/inner/default** completed; rmse=32.91081846583218; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-076-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:05:02.512911+03:00: **HYP-077/inner/default** completed; macro_f1=0.6110510854211656; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-077-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:05:53.164561+03:00: **HYP-078/inner/default** completed; macro_f1=0.6288481661155734; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-078-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

HYP-079 inner completed: RMSE=30.663255379972725 vsH70=30.663255379972725; w70=1.0; [report](results/2026-10-24/HYP-079-01-inner-combination-sprint/report.md). Повтор не запускать.

- SPEC011 2026-10-24T02:07:00.052064+03:00: **HYP-078/outer/default** completed; macro_f1=0.6593084142037399; reference=0.6495311620074445; [отчёт](results/2026-10-24/HYP-078-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:09:12.809950+03:00: **HYP-080/inner/default** completed; macro_f1=0.6140917642777984; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-080-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:11:52.194994+03:00: **HYP-081/inner/default** completed; rmse=30.689409165993794; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-081-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
HYP-081 inner incremental НЕ подтверждена:30.68940916599 против H70inner30.66325537997, прирост кH69 унаследован. Outer не запускать, повтор запрещён.

- SPEC011 2026-10-24T02:13:39.220401+03:00: **HYP-050/inner/default** completed; macro_f1=0.5991047549783951; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-050-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:17:33.467742+03:00: **HYP-083/inner/default** completed; rmse=30.651578052812432; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-083-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:18:48.720601+03:00: **HYP-082/inner/default** completed; macro_f1=0.6150544797423486; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-082-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:20:21.025521+03:00: **HYP-087/inner/default** completed; macro_f1=0.6254419869564557; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-087-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:20:22.661130+03:00: **HYP-084/inner/default** completed; macro_f1=0.5580393390877207; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-084-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:20:38.588116+03:00: **HYP-085/inner/default** completed; rmse=46.02178005161315; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-085-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:23:18.973630+03:00: **HYP-086/inner/default** completed; rmse=31.12117379943106; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-086-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:24:31.976838+03:00: **HYP-083/outer/default** completed; rmse=34.32053895104047; reference=39.39219382766771; [отчёт](results/2026-10-24/HYP-083-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.

HYP-091/inner completed RMSE=30.636363541581975 vsH70=30.663255379972725 weights=[0.43908725498434387, 0.5446376329320275, 0.016275112083628654] [report](results/2026-10-24/HYP-091-01-inner-multiblend-sprint/report.md). Повтор не запускать.

- SPEC011 2026-10-24T02:27:12.581611+03:00: **HYP-085/outer/default** completed; rmse=44.12424608516781; reference=39.39219382766771; [отчёт](results/2026-10-24/HYP-085-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.

HYP-091/outer completed RMSE=34.21382789041039 vsH70=34.318489927675174 weights=[0.43908725498434387, 0.5446376329320275, 0.016275112083628654] [report](results/2026-10-24/HYP-091-02-outer-multiblend-sprint/report.md). Повтор не запускать.

- SPEC011 2026-10-24T02:28:38.828414+03:00: **HYP-089/inner/default** completed; rmse=31.05521238320071; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-089-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:29:25.099941+03:00: **HYP-090/inner/default** completed; rmse=34.86994782367874; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-090-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:33:42.665450+03:00: **HYP-088/inner/default** completed; macro_f1=0.5978384913324375; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-088-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:36:47.076490+03:00: **HYP-092/inner/default** completed; macro_f1=0.6075155330908212; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-092-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:36:54.448689+03:00: **HYP-094/inner/default** completed; macro_f1=0.6192492910174916; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-094-02-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:37:21.769093+03:00: **HYP-093/inner/default** completed; macro_f1=0.6002777710371685; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-093-02-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:38:09.656982+03:00: **HYP-094/inner/long** completed; macro_f1=0.6223414898157842; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-094-03-inner-long-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:40:16.964501+03:00: **HYP-092/inner/extended** completed; macro_f1=0.599340154183666; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-092-02-inner-extended-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:40:18.832303+03:00: **HYP-096/inner/default** completed; macro_f1=0.6291379699977411; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-096-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:40:57.430905+03:00: **HYP-097/inner/default** completed; macro_f1=0.6334156298867437; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-097-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:41:27.865741+03:00: **HYP-095/inner/default** completed; macro_f1=0.6227569652507932; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-095-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:48:40.764829+03:00: **HYP-100/inner/default** completed; macro_f1=0.550835584943435; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-100-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:48:47.448496+03:00: **HYP-098/inner/default** completed; macro_f1=0.6208270606519142; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-098-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:50:50.561132+03:00: **HYP-099/inner/default** completed; rmse=31.099625876123866; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-099-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:51:04.647022+03:00: **HYP-097/outer/default** completed; macro_f1=0.6384122468214593; reference=0.6495311620074445; [отчёт](results/2026-10-24/HYP-097-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 03:00MSK: HYP-057 proposed→registered: known-rating native LightGBM700 monotone raw rating=-1, features29/30, overlay .7 only H70-unmatched; cachedH70 no refit; [card](docs/hypotheses/HYP-057.md). HYP-101 registered: shared vector-leaf XGBoost300 exact softF1+.5CE, trueH88cachedOOF,2CPU; [card](docs/hypotheses/HYP-101-vector-leaf-f1.md). Different model representation from scalar H92; frozen before fit; no known-repeat.

- SPEC011 2026-10-24T02:58:56.387990+03:00: **HYP-057/inner/default** completed; rmse=30.00420378193599; reference=42.46005247026025; [отчёт](results/2026-10-24/HYP-057-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:59:42.157690+03:00: **HYP-057/outer/default** completed; rmse=43.038591928820146; reference=39.39219382766771; [отчёт](results/2026-10-24/HYP-057-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.

- SPEC011 2026-10-24T02:59:56.887724+03:00: **HYP-101/inner/default** completed; macro_f1=0.6178371238873426; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-101-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 HYP-103 registered before fit: Ordered boosting on unchanged H88 contextmodel400depth5, trueH88OOFreuse, no teacherfits; [card](docs/hypotheses/HYP-103-ordered-context.md). Novelty only OrderedvsPlain, source/configfixed,2synthetictestsPASS.

SPEC011 HYP-102 registered before parameter fit: nonconvex amplitude calibration of cachedH91 relativeH69, lambda in[0,2] by original MSE on inner only, max0 inference; [card](docs/hypotheses/HYP-102-confidence-amplitude.md). Source/config/inputsha frozen; no learner fits.

HYP-102/inner completed RMSE=30.449416831901488 vsH91=30.636363541581975 amplitude=1.1289189550164425 [report](results/2026-10-24/HYP-102-01-inner-amplitude-sprint/report.md). Повтор не запускать.

HYP-102/outer completed RMSE=34.50690731292415 vsH91=34.21382789041039 amplitude=1.1289189550164425 [report](results/2026-10-24/HYP-102-02-outer-amplitude-sprint/report.md). Повтор не запускать.

- SPEC011 2026-10-24T03:12:54.239971+03:00: **HYP-103/inner/default** completed; macro_f1=0.597392348204526; reference=0.5873380697311004; [отчёт](results/2026-10-24/HYP-103-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

Дополнение 2026-10-24 03:14 MSK: в строке регистрации H57/H101 ручная метка 03:00 приблизительна; фактическое started H57 02:58:56.387990+03:00 фиксирует launcher. Регистрация была записана командой до fit. Убран буквальный префикс backtick-n и нормализированы окончания строк; научные исходы не менялись. H50/H57 завершены, H63 только feature-only диагностика; прежний список proposed описывает исходное предложение, текущие запреты повторов определяются последующими outcome.

SPEC011 HYP-104 registered before fit: DART scalar multiclass trees300 with dropout rate.1/skip.5, exact.5CE+.5softF1, trueH88cachedOOF reused. H92extended degradation is exploratory motivation, not proof of overfitting. [card](docs/hypotheses/HYP-104-dropout-f1.md). Four synthetic tests pass; configuration/source/data/splits frozen before real fit. Strict inner admission>.6334156298867437, otherwise no outer. Deadline reserve remains30minutes.

- SPEC011 2026-10-24T03:33:58.022556+03:00: **HYP-104/inner/default** timeout; None=None; reference=None; [отчёт](results/2026-10-24/HYP-104-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

2026-10-24 SPEC012: прочитаны журналы Ivan иSergey24e3311. Новых гипотез/публичныхfit0. [Сравнение](results/2026-10-24/branch-comparison-sergey-24e3311-v3/report.md): branch-scopedH70..75Sergey не смешиватьсIvanH70..75; nominalgains0.8877%/2.2623%vsSergey, differentOS/frozenmeta. H104timeout900secнеимеетметрики; не повторять. API-keyнехранится, endpointнеизвестен.

SPEC013 HYP-105 registered before real fit: exactlags1/2/3+causalstate+fit-selected3changingfields, nativeNaNLightGBM1200/.03/63/minchild20/col.8/bag.8freq1/lambda5/seed42/4CPU. AllrowsoriginaltargetRMSE; notH76latestdonorCat/SergeyH72EWresidual. Cards/docs/tests frozen before900sec cumulative inner+outer. [card](docs/hypotheses/HYP-105-multilag-native-boosting.md). Two fixed inferencearms pure/blend75Sergey, outeronlybestinner<30.636363541581975; no oldrefits.

HYP-105/inner: {"status": "completed", "hypothesis": "HYP-105", "stage": "inner", "metric": "rmse", "score": 32.01672432373479, "arm_scores": {"pure": 34.56623580026114, "blend75": 32.01672432373479}, "selected_arm": "blend75", "control_score": 31.04663635241995, "selected_on": "inner_only", "learner_fits": 1, "old_model_refits": 0, "fit_rows": 22054, "valid_rows": 12946, "eligible_for_outer": false, "seconds": 8.094000000011874, "fingerprint": "c0523370162349ee4ad3a8f474ce60306f6f816758b3a35777060ad20b0228de"}; [report](results/2026-10-24/HYP-105-01-inner-external-multilag/report.md). Не повторять.

SPEC013 HYP-106 registered before keyfit: fit-only greedy≤6feature entitykey scorecoverage/meanrowsperiod, proxyquerylastFITperiod, coverage≥.2/std ratio≤.5, strictpasthalf-life1history+cachedSergeyfallback. NotfixedkeyH69/H70; nodependentqueryselection, nolearnerfits. CardHYP-106-training-key-discovery,6synthetictestsPASS;900sec cumulative, sameinnergate<30.636364 andfixedpure/blend75arms.

HYP-106/inner: {"status": "completed", "hypothesis": "HYP-106", "stage": "inner", "metric": "rmse", "score": 30.963253972816442, "arm_scores": {"pure": 31.15263063723188, "blend75": 30.963253972816442}, "selected_arm": "blend75", "control_score": 31.04663635241995, "selected_on": "inner_only", "learner_fits": 0, "parameter_fits": 1, "old_model_refits": 0, "fit_rows": 22054, "valid_rows": 12946, "eligible_for_outer": false, "seconds": 11.90600000001723, "fingerprint": "58cced6f80dc02bd615b7ca46705d6be831d11aa1377f735e38c63356c19de61"}; [report](results/2026-10-24/HYP-106-01-inner-external-key/report.md). Не повторять.

SPEC013closed: H105/H106 обе завершены наinner,2новыхpipelinechecks/4fixedarmmetrics/1learnerfit+1keyparameterfit,0oldrefits. H105выигрыша нет, H106только0.268571%vsSergey и не прошла preregisteredoutergate30.636364. Outerнеповторятьрадипросмотра; [review+nofitverification](results/2026-10-24/external-repositories-review-v1/report.md).562unitPASS; liveplatformscoreнепроверен. НоваяпоставкаНЕсоздана,Сергейостаетсяосновой.

SPEC014 HYP-107 registered до parameter fit: новая композиция Sergey H73 + Ivan H70/H83/H85, simplex SLSQP200/ftol1e-12 на Year4 inner; веса freeze до outerload. Gate inner<30.636363541581975, outer<34.21382789041039; >=2active. Cards/SPEC014/PLAN014/tests3PASS; oldlearnerfits0,900sec. Не повтор H91 без Sergey или H102 amplitude.

HYP-107/inner: RMSE=30.59160391991867; Sergey=31.04663635241995; Ivan=30.636363541581975; passes_gate=True; [report](results/2026-10-24/HYP-107-01-inner-sergey-fusion/report.md). Проверка завершена, не повторять.

HYP-107/outer: RMSE=34.142888130110755; Sergey=34.520268654585415; Ivan=34.21382789041039; passes_gate=True; [report](results/2026-10-24/HYP-107-02-outer-sergey-fusion/report.md). Проверка завершена, не повторять.

SPEC014 H107 закрыта: cached inner30.59160391991867/outer34.142888130110755, gain1.093214%vsSergey и0.207342%vsIvanH91. Одна настройка весов/0новыхscientificlearnerfits. Отдельнаяtask-localпоставка, technicalfullrefit17unique fit stages/10learner stages,295.578sec,22449testpredictions;568unit+271base,cleanZIP/replay/CLI PASS. Runtime-graphquality отдельнонеизмерено,20%bothнеполучены. [Отчёт](results/2026-10-24/series-sergey-ivan-fusion-v1/report.md). Не повторятьH107.

Read-only обзор maxim/autoresearch@be9d9ae: NV1raw+ranks, NV2raw+signed_log, NV3raw+robustprojections, NV4raw+ranks+projections уже проверены Максом; не выдавать тот же контекст за новую гипотезу. Reported outerNV2RMSE42.012317102 и classifierMacroF1.637337946; наших моделей не обучали. [Обзор](results/2026-10-24/publication-and-max-review-v1/report.md).

HYP-108 regression/inner: {"status": "completed", "hypothesis": "HYP-108", "kind": "regression", "stage": "inner", "metric": "rmse", "score": 30.285884162436915, "pure_score": 51.908456385388476, "control_score": 30.59160391991867, "selected_weight": 0.0928330654623246, "arm_scores": [[0.0, 30.59160391991867], [1.0, 51.908456385388476], [0.0928330654623246, 30.285884162436915]], "eligible_for_outer": true, "fit_rows": 22054, "valid_rows": 12946, "new_learner_fits": 1, "old_model_refits": 0, "seconds": 22.078000000008615, "fingerprint": "a8693ed5c731db38153494a3515b7b07d9089ac106c3661809032a9b12153c95"}; Подтверждена в проверенных условиях; [report](results/2026-10-24/HYP-108-01-regression-inner-tabm-mini/report.md). Не повторять.

HYP-108 classification/inner: {"status": "completed", "hypothesis": "HYP-108", "kind": "classification", "stage": "inner", "metric": "macro_f1", "score": 0.628945309907142, "pure_score": 0.516402108331563, "control_score": 0.6288481661155734, "selected_weight": 0.1, "arm_scores": [[0.0, 0.6288481661155734], [0.1, 0.628945309907142], [0.25, 0.6280732998577576], [0.5, 0.6275732038748059], [0.75, 0.5795224999605784], [1.0, 0.516402108331563]], "eligible_for_outer": true, "fit_rows": 47075, "valid_rows": 23454, "new_learner_fits": 1, "old_model_refits": 0, "seconds": 112.73399999999674, "fingerprint": "3e84a88ffd5e8ab248016817e68359b370fb34597eeca51e505ea3b36d980839"}; Подтверждена в проверенных условиях; [report](results/2026-10-24/HYP-108-02-classification-inner-tabm-mini/report.md). Не повторять.

HYP-108 regression/outer: {"status": "completed", "hypothesis": "HYP-108", "kind": "regression", "stage": "outer", "metric": "rmse", "score": 33.97984036360298, "pure_score": 51.85189626940788, "control_score": 34.142888130110755, "selected_weight": 0.0928330654623246, "arm_scores": [[0.0928330654623246, 33.97984036360298]], "eligible_for_outer": true, "fit_rows": 35000, "valid_rows": 18308, "new_learner_fits": 1, "old_model_refits": 0, "seconds": 29.703000000008615, "fingerprint": "25d738cbcd0c15a587fe2865dcb6d1f73a1b362f2eea10a0fe95d6ad15c5150d"}; Подтверждена в проверенных условиях; [report](results/2026-10-24/HYP-108-03-regression-outer-tabm-mini/report.md). Не повторять.

HYP-108 classification/outer: {"status": "completed", "hypothesis": "HYP-108", "kind": "classification", "stage": "outer", "metric": "macro_f1", "score": 0.6602795926985132, "pure_score": 0.5099944452622676, "control_score": 0.6593084142037399, "selected_weight": 0.1, "arm_scores": [[0.1, 0.6602795926985132]], "eligible_for_outer": true, "fit_rows": 70529, "valid_rows": 17141, "new_learner_fits": 1, "old_model_refits": 0, "seconds": 180.85899999999674, "fingerprint": "c2c69474d79a3fd304178c8bf26539e6051ee93e589791b76061c9a7edebabf9"}; Подтверждена в проверенных условиях; [report](results/2026-10-24/HYP-108-04-classification-outer-tabm-mini/report.md). Не повторять.

SPEC016 explicit ZIP control regression/inner: {"status": "completed", "kind": "regression", "stage": "inner", "role": "explicit_previous_zip_control", "metric": "rmse", "score": 30.656905391248326, "fit_rows": 22054, "valid_rows": 12946, "seconds": 146.59399999998277, "zip_sha256": "9a3924a53f6bf6191d27dd0717c44afc4137757557e469efe77c3eac28d1756a", "actual_archived_factory": true, "previous_quality_trials_replayed": false}; [report](results/2026-10-24/ZIP-CONTROL-regression-inner-torch-v1/report.md). Не повторять.
