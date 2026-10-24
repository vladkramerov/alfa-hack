# Three-hour improvement Implementation Plan

**Goal:** проверить новые механизмы, стремясь к−20%RMSE/+20%MacroF1 одновременно; сохранить честный лучший результат.
**Spec:** [SPEC011](../specs/SPEC-011-three-hour-improvement.md).
**Architecture:** отдельные portable sprint_regression.py/sprint_classification.py, offline runner/cache guard, неизменные старые predictions. Подключение в агент только после отбора/проверки.
**Deadline:**04:15:07MSK,2026-10-24. Research≤2h, finaleval/integration≥30min резерв. Научные fits≤900sec/hyp,≤4threads,≤8общихCPU.

- [ ] T1 прочитатьисторию, записатьмеханизмы/численныецели/запретповторов; доfitсоздатькарточки/manifest/sourcefreeze и budgetledger.
- [ ] T2 regressionworker: H70 temporalentitylinking/ratingadjustment; H71targetnormalizer; tests first, compatiblebasecacheonly.
- [ ] T3 classworker: H72exactsoftF1 derivatives/neuralorfunctionalgradient выбрандоfit; H73ECOC; H74contextualmetaonlyaftercacheaudit. Tests/code зафиксировать до запуска.
- [ ] T4 runner: независимые claims/outcomes perjob, source/data/split/cache hashes,deadline/no-replayguard, innerselection record; не редактировать код во времяfrozenfits.
- [ ] T5 финалисты: один/task,originalouter и optionalknownsecondary, nooldcontrolrefit; независимые scores/lineage verifier, отдельная оценка20%threshold.
- [ ] T6 внедрить лучший подтверждённый вариант, preflightflatZIP/source/tests, publishivan_branch; меньшийвыигрышневыдаватьза20%.

## Progress

02:32MSK: T1–T4 исполнены для H70…H91/H50, H63 feature-onlyarchive. H70outer34.31849, H78outer.6593084; H91 weightsfrominner[.4390873,.5446376,.0162751]→outer34.21382789, текущийbestRMSE. H83outer34.32054чутьхужеH70; H85alone44.1242непродвигать. H71/H73/H79/H81/H84/H85/H86/H87/H89/H90 не дали incrementalgain; остальныеinnerприросты не обошлиH78. H88nestedrawOOF3new31kfits+contextualfix выполняется; новыеH92–94 будутиспользоватьТОЛЬКОготовыеnestedprobasбезteacherrefits. Goal20%bothещёнедостигнут. ProductionadaptersH70/H78nofitreplayouterverified, новыйpooladapterH91подготовка. T5/T6необъявлятьзавершённымибезfinalaudit/cleanZIP/fullrefit.

Вторая волна после H70outer12.88%: H75affine expectedMacroF1, H76supervised conditional temporal transitions, H77ranked Lovasz-softmaxboosting. Цель20%пообеим сохраняется; SPEC011дополнение явно отмечает адаптивное многократное использование outer. Выбор в каждой волне по inner до outer, без старых refit.

01:15MSK начат бюджет. Read-only audits completed, no newscientificfits: regression support broadphysicalkey69% vs18.5%; trainingclass ambiguous1167groups11984rows, optimisticrelaxed empiricalF1bound.821057 (nottestbound). T1implementationunderway; методывыбираеткоординаторврамкахпрямогоширокогопорученияпользователя, безновогогейтаобщегоразрешения.
