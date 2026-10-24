# Точечная доработка senior-agent — 24 октября 2026

Прямое поручение: только another_solution, исходник сохранить до правок, измерить, проверить всего2–3точечных гипотезы, подготовить ZIP/MDтаблицу/push; объяснение до400слов. Архитектуру не переписывать. Оригинал сохранён commit568fc9879e677371b75fb879d49011cc6c2377f6, ZIP SHA256cbf5de846d8a4a4f760e9574a60d47c42c1adeb7b340fd655f5d296b41589547.

## Протокол до запусков

Оригинал запускается безLLM на двух публичных задачах. TIME_LIMIT300с,MAX_ITERATIONS3,TUNE_TRIALS0,N_JOBS4, одинаковыйseed42. Это ограниченный локальный контроль, не результат3часового запуска/платформы. Исходники извлекаются в отдельный clean taskdir побайтно; harness только сохраняет уже рассчитанныеOOF/testпрогнозы и итоговое решение, не меняет алгоритм. Сохраняются всеfolds,A/B,task/metric,source/dataSHA, failures. ПредыдущаяSPEC012 имеет другойагент/протокол и не является новымbaseline.

## Всего3гипотезы, до вычисления

- AS001: ограничитьэкстремальность multiclasspostprocess весов классов — мягкаяboundedcoordinatecalibration[A], принимаемнаB. Уточнённые fixedbounds/stepsдоопыта. Model/treefitsнеповторяются, толькоsavedOOF.
- AS002: точноеconvexsimplexrefinement весов уже найденногоregressionensemble подRMSE наA, с небольшимshrinkкстарымвесам. Исходныйensemble остаётсякандидатом, новыйпринимаетсянаB темижепорогамиисходника.
- AS003: macro-awareloglossproxy дляжадногоclassificationensemble, sqrtinverseclasscountsweights толькоизA. Дляпрочихметрикповедениеисходное; конечноеBрешениепоисходнойцелевойметрике.

Новаяправкa внедряетсятолькопри фактическомвыигрышеB и проверкеостальныхметрик/формата; отрицательныесохранить. ВыигрышиOOF/B разведочные, независимогоскрытогоскоранет. РеальнуюLLMпораньшеданномууказаниюнепроверять.

## Доставка

Толькоroot.py+originalrequirements.txt, nofolders/data/weights/secrets/pip, <=100MB. НовыйZIPпроверитьCRC/точныйсписок/sourceSHA/импорты/cleanCLI. Original requirements сравнитьсспискоморганизаторов.8CPU32GB3h, n_jobs<=8. Чёткоотмечатьнезапущенныепроверки. Commit/pushтолькоanother_solution.

### Точные фиксированные варианты до вычисления

AS001: абсолютные classweightsgrid[.5,.75,1,1.25,1.5,2],3coordinatepassesнаA, сохранитьлишьеслиBлучшеidentity. AS002:SLSQP simplexNNLSRMSE onA+ridge1e-4towardgreedyweights, finalw=.5old+.5opt;max100steps/ftol1e-10;приниматьпротивстарогоансамблялишьBrelative>=.0005 иbootstrap200win>=.8(порогикаквисходнике). AS003: greedy25steps/max8models сsqrtinverseclasscounts(A) weightedlogloss;Bвыборкакисходный. Модели/базовыеOOFнеперезапускаются. Все3variantsexactcodeвwork/another_solution/refinements.py,hashвresults/evidence.

## Завершено

Исходниксохранён568fc98; baseline2задачи/10trainingruns;3hypotheses/0newtreefits; внедренаAS002;6unit/38independent/2cleanZIPCLI;ZIProot30/136784bytes. Всеизмеренияиограничениявreports/ANOTHER_SOLUTION_RESULTS_2026-10-24.md.
