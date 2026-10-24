# SPEC-012: научно обоснованная новая серия Сергея

Статус: исполнение. Основание: прямое поручение пользователя искать современные статьи, проверять новые гипотезы, подготовить улучшенный ZIP и push только sergey_branch. Топ-10 — цель команды, не проверяемый локальный критерий; пользователь подтвердил отсутствие подробных hidden scores. Реальную LLM/API не проверять по ранее данному указанию.

## Требования

- REQ-001: читать полный HYPOTHESES перед каждой новой гипотезой; заранее фиксировать механизм/бюджет/данные/split. H76 нормированная сходимая метамодель; H77 affine Macro-F1; H78 focal differentiable MCC; H79 PLE-MLP; H80 BatchEnsemble-inspired MLP. Все чужие pretrained веса/данные запрещены. На ID до900с локальной модельной проверки; техническое формирование отсутствующего OOF кэша отдельно фиксируется, один раз на протокол.
- REQ-002: сравнение с текущими H73 RMSE34.520268654585415/31.04663635241995 и H75 F1.64472256711304/.6075743572072007. Старые validation/full-branch fits переиспользовать. Нет сохранённого base OOF: восстановить 9 component fits на каждый из двух outer-fit протоколов ровно один раз, кешировать, это необходимые calibration inputs для новых механизмов, а не новая baseline-гипотеза. Точный список повторяемых component fits раскрыть. Никакие outer labels не влияют на fit/гиперпараметры/эпохи/калибровку.
- REQ-003: обучить метаслой и его решающее правило только на grouped training OOF; калибровка второго уровня только по cross-fitted meta predictions. Нейросети: inner grouped holdout для эпох, затем full outer-fit refit с frozen epochs, outer только оценка. Primary и secondary уже многократно просмотрены, не называть их независимыми скрытыми.
- REQ-004: внедрять подтверждённый новый прирост без просадки другой задачи; сохранить H73 если регрессия не меняется. Отрицательные и технические исходы сохранить немедленно. Не обещать кратный выигрыш, если измеренного нет.
- REQ-005: улучшить переносимость агента: model-appropriate preprocessing без обязательного удаления строк; компактная диагностика ошибок для AutoResearch. Это инженерный перенос разрешённого поведения, не повтор H41 под новым ID и не автоматический прирост метрик. Предложенные нейросети в портфеле только при принятии по валидации и под бюджетом.
- REQ-006: новый ZIP root-only solution.py+original requirements.txt+нужные .py, <=100000000 bytes, без данных/весов/конфигов/секретов; source SHA, CRC, compilation, clean execution и full default offline portfolio проверены. Реального API/платформенной отправки нет.
- REQ-007: MD командный отчёт, полная таблица всех новых исходов, связи со старым журналом и точные дельты. Коммит/push только sergey_branch; ivan_branch неизменна; исходные данные/ZIP не менять.

## Приёмка

Новый измеренный выигрыш на обоих зафиксированных протоколах классификации либо иной доказанный выигрыш без регресса; независимый пересчёт по сохранённым прогнозам. Все численные/нечисленные исходы, source/data/split hashes, условия времени и памяти в results/. Полный offline lifecycle без искусственного ограничения одного кандидата + mock LLM сценарии; тест формата каждого zip entry и actual root-copy запуска.

## Первичные источники

TabM ICLR2025: https://arxiv.org/abs/2410.24210 ; Numerical embeddings NeurIPS2022: https://papers.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html ; TabPack ICML2026: https://arxiv.org/abs/2607.05380 ; JMLR2024 complex metrics: https://jmlr.org/papers/v25/22-1137.html ; OMMA ICML2024: https://proceedings.mlr.press/v235/kotlowski24a.html ; MCC tabular preprint2026: https://arxiv.org/abs/2604.21252 . Применяемые упрощения обозначать адаптациями, не точными репликациями.

## Расширение H81 до fit

Проверка cross-family complementarity H79 и treebranchOOF: HYP081. Основание пользовательпоручилпродолжатьновыемеханизмыдоулучшения; новоеproductscopeне добавлено. Budget900s,totaltwoouter; точныйпротоколвкарточке.

## Расширение H82 до fit

НовыйlearnedmetricsoftneighborsмеханизмизModernNCA2025; отличаетсяотзакрытогоfixedH52. Условияиbudget900sвкарточкеH82.

## Расширение H83/H84 до fit

Новые causal covariate differences дляповторныхphysicalprofiles; H83dynamicforecaster/H84fixedblendwithH73. Не повторH72: модель получаетстрогопрошлыеXиихизменение, новаяцелеваяпеременнаяannualdeltafromlastyear. Budget900sec/ID, обаtemporalprotocols, cachedH43/H73.

## Расширение H50 до первого fit

РанееproposedноненачатаяH50focallossбудетпроверенаодинраз:classbalancedfocalXGB, новыйtrainingobjective. Fixedgamma2,positivecurvaturesurrogate,900s,дваouter.
