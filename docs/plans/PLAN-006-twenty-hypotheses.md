# Проверка двадцати гипотез — Implementation Plan

> For agentic workers: use superpowers:executing-plans with parallel independent feature/model implementation. User authorized execution and subagents; proceed without another approval round.

**Goal:** проверить HYP-022…041 и сохранить23честных результата без повторения прежнихexperiments.
**Architecture:** отдельные stateful DataFrame transforms, model wrapper сtrain-only preprocessing и registrar, advancedinnerпроцедуры сpersistentcache, единственный runner/journal writer и независимый verifier.
**Tech Stack:** Python3.12, numpy/pandas/sklearn1.5.2, ExtraTrees/XGBoost2.1.4, imblearn0.12.4 из предоставленногоrequirements.
**Spec:** [SPEC-006](../specs/SPEC-006-twenty-hypotheses.md).

## Global constraints

Все20ID/23оценки; исходныеSHA/splits; <=4threads; timeout1800s/оценку; старыеouterbaselinesбезrefit; критерии42.19670002719412/.5914513709191194; результатыатомарно; всеinnerfitfingerprintsдоfit, повторыизcache.

## Review focus

Группы одинаковыхrawXдоинженерии/синтетики не пересекаются; временныеinner не видят будущее; cachedclassorder проверяется; пустой отбор переиспользуетbaseline; replay/timeoutнеповторяетstarted.

## Tasks

- [x] 1. Создать failing meaningfultests для transforms в tests/test_twenty_features.py, common/registry в tests/test_twenty_runner.py и corrections/cache в tests/test_twenty_advanced.py; выполнитьRED.
- [x] 2. Реализовать experiments/twenty_features.py: make_features(number) -> fit/transform DataFrame,indexpreserved; независимыйсубагент, GREEN tests.
- [x] 3. Реализовать experiments/twenty_common.py: make_model(task,policy,feature,family,target_transform,parameters) -> fit/predict/proba+audit; registrar исключаетодинаковыйtrainingdata/configfingerprint.
- [x] 4. Реализовать experiments/twenty_advanced.py: evaluate_advanced(number,task,X,y,Xvalid,context)->prediction/proba/classes/audit, безoutery; grouped/timeinnercache; independentсубагент, GREEN tests.
- [x] 5. Реализовать experiments/run_twenty_hypotheses.py и twenty_worker.py: заморозить23jobs, карточки/снимки/registrystartedдоsubprocess; resumeпропускаетattempted; finishкаждыйисходв3журнала. H41повториспользуетизмеренныеouterroutes.
- [x] 6. Выполнитьunit/preflight; запуститьвсе23оценки, сохранятьпрогресс; научныеотрицательныерезультаты не подбиратьдоуспеха.
- [x] 7. experiments/verify_twenty_results.py: независимоrecompute/positions/classes/groupboundaries/fingerprints; свежийreview, итоговыйотчёт/CSV, обновитьстатусы.

Ни одной поведенческой правки productionagent или новогоZIP; экспериментальныйисход предшествует решению о внедрении.

Ruling до завершения серии: HYP-028/regression остановлен guard до нового outer-fit, потому что guard сравнивал rows/family/parameters, но не фактический набор X. Inner модели и importance сохранены; выбран54из62столбцов. Сначала завершить frozenv1-серию безправки еёисходников, затем исправить guard на сравнение точногоrawXfingerprint, выполнитьRED→GREEN `tests/test_twenty_guard.py`, сохранитьv2source и техническипродолжитьтольконеобученныйreducedoutercandidateчерез `continue_twenty_blocked_selection.py`. Перваяошибка остаётся отдельнымrun; innerfit не повторять, cache reuse обязателен. Аналогично обработать classification, если guard остановит его. Это продолжение неполученнойоценки, не повтор провереннойгипотезы.

Уточнение до первогоfit: H41 joint routes legacy+ETlog/XGB; full+ETlog/XGB; light+feature24/26+LightGBM. Review существенные замечания устранены; preflight108testsOK (results/2026-10-23/spec006-preflight-tests.log). imbalanced-learn0.12.4 установлен в .venv-ml из разрешённого requirements; productionrequirementsнеизменён.

Завершено:20ID/23метрики/25историческихпопыток;539checksOK,110unitstestsOKдоSPEC007. H28продолжениявыполнилитолькопоодномуранееостановленномуouterfit; ниодногоinnerповтора. [Отчёт](../../results/2026-10-23/TWENTY_HYPOTHESES_REPORT.md).
