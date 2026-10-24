# SPEC010: независимый агент и короткие проверки H52/H62

Дата: 2026-10-24, Москва. Статус: реализация проверена, метрический допуск не пройден. Основание: прямой ответ «Подтверждаю, реализовать и проверить» на итоговый план [grill-me](../../results/2026-10-24/grill-current-solution/report.md).

## Итог2026-10-24

Реализация и согласованная исследовательская серия завершены; цель улучшения метрик **не достигнута**. H62 RMSE58.58032538035888 против H69контроля42.46005247026025 на одном новом temporal split; H52 Macro-F1.4803098632106223 против H48контроля.6103340454456273 на одном новом grouped split. Оба механизма не подтверждены по критерию выигрыша; повтор не запускать. Прежние baseline/solution.py/ZIP сохранены, новый архив не создан.

113/113 независимых сверок,257unitтестов,4clean-flatCLI сценария/8синтетическихfit. Научные данные:4новыхfitNN/FM и17контрольныхкомпонентов в двух явно разрешённых сравнительных процедурах;6top-levelclaims. Лимит900сек на гипотезу соблюдён. [Отчёт](../../results/2026-10-24/series-clean-rewrite-v1/REPORT.md), [приёмкаCLI](../../results/2026-10-24/rewrite-clean-cli/verification.json). Реальный API и private-quality не проверены; реализация не объявляется улучшенной платформенной поставкой.

## Цель

Написать отдельный универсальный ML-агент с диагностикой ошибок и выбором предобработки по модели; проверить два ещё не запускавшихся механизма. Переписывание не объявляется улучшением без измерений. Прежние solution.py, baseline.json и ZIP сохраняются до подтверждённого выигрыша.

## Требования

- REQ-1001: новые research_*.py не импортируют automl_*, не содержат SHA/коэффициентов публичных победителей; единый CLI принимает task.md и работает на неизвестных именах target/ID/путей.
- REQ-1002: чтение train/test/dictionary/sample из описания или однозначного вложенного расположения; сохранение лексических ID/классов; target/ID исключены из X. Атомарный submission.csv рядом с task.md, проверка числа/порядка строк, столбцов, конечности чисел/допустимости классов.
- REQ-1003: тип/primary metric/направление из task.md; train-only imputation/scaling/encoding по модели. Не навязывать Gaussian и удаление строк. Validation/test не фильтруются.
- REQ-1004: research_inner отделён от дополнительного confirmation до fit; временная регрессия past→future, классификация группирует исходные raw-X дубликаты. LLM получает диагностику inner, не labels/score confirmation. Исходный многократно просмотренный outer — только исторический контекст, не дополнительный подбор.
- REQ-1005: H52 — exact blocked 25NN по смешанному расстоянию: numeric median/IQR fit-only, abs differences capped at 3, среднее по numeric; categorical mismatch, среднее по categorical; среднее доступных двух типов. Unknown/missing категории сохраняют отдельную лексическую метку. Равные голоса, ничья по training class order. Numeric constant scale=1; numeric missing median + отдельные missing flags. Не материализовать N×N.
- REQ-1006: H62 — FM второго порядка, rank8, исходный y; numeric median/standardization и OHE fit-only, original missing flags. Минимизировать squared loss с детерминированным minibatch Adam, numpy/scipy; target центрировать/масштабировать train-only для устойчивости, обратно в исходную шкалу при predict. rank8, epochs20, batch512, learning_rate0.005, L2=0.001, factor init std0.01, seed42. Нет torch/новых зависимостей. Производные сверить finite differences до реального fit.
- REQ-1007: локальная гипотеза целиком <=900сек, включая inner/evaluate; отдельный subprocess, threads<=4, timeout записать как технический исход. Полный HYPOTHESES.md читать до каждого fit, зарегистрировать started/fingerprint до обучения. Freeze params/dataSHA/splits/source до запуска; никаких повторов известных контролей на старом протоколе.
- REQ-1008: только один дополнительный сравнительный запуск финалиста и текущего лидера на заранее фиксированном новом протоколе на задачу (прямое исключение пользователя). Старый лидер обучается без frozen meta-state с иного training fold. Условия data/preprocessing/model каждого fit записать. Не выбирать параметры по confirmation. Если новый кандидат явно проиграл inner, дополнительное сравнение можно не запускать; отметить этот статус, не выдавать за проверенную победу.
- REQ-1009: baseline выбран на сопоставимых scores: без просадки достаточно строгого улучшения; при компромиссе относительный выигрыш строго >15%, просадка другой метрики строго <10%. RMSE gain=(control-new)/control, F1 gain=(new-control)/control. Отдельный split не даёт права сравнивать с прежним числом другого split.
- REQ-1010: LLM proposals используют только platform OPENAI_BASE_URL/API_KEY/MODEL; пустые/обрезанные/ошибочные ответы не разрушают результат. Контекст включает per-class confusion или residual slices и прежние отрицательные исходы. Кандидаты типизированы и проверяются до fit. При сбое LLM работает локальный bounded search.
- REQ-1011: сохранять отдельные results/<date>/<hypothesis>-NN-*/ для каждой проверки, журнал started/outcome/fingerprints, source/split/metrics/predictions/log/report; ошибки не считать опровержением. Финальный рефит только выбранного кандидата, ранний валидный fallback и резерв времени.
- REQ-1012: Python3.12, предоставленный requirements byte-exact, flatZIP<=100MB,<=8CPU/32GB/3h на платформе. Новый ZIP публиковать только после REQ1009 и preflight; без выигрыша сохранить новый исследовательский код и отчёты, прежний архив не переименовывать в новый.

## Приёмка

Синтетические тесты: train-only transforms/unknown categories, finite difference FM, обучение аддитивного+парного сигнала, predict repeat invariance, exact NN против brute force, opaque IDs/классы, nested paths/renamed target, metric direction/promotion boundaries, duplicate/time separation, history reject seed-only repeat, empty/truncated API, clean CLI fallback/output. Новые гипотезы проверяются только после этих тестов; source/config freeze и независимый пересчёт сохранённых прогнозов обязательны. Никаких утверждений о private/10x без платформенных измерений.

## Rulings

- Пользователь задал ivan_branch; работа ведётся в этой ветке с новыми отдельными файлами, без нового worktree/переноса незакоммиченных audit-документов.
- Дополнительный confirmation: регрессия Year_Factor4 (fit годы<4); исходный год5 не используется для новых решений. Классификация — другой заранее фиксированный SGKF fold по raw-X groups. Выбор fold не зависит от score.
- Исследовательские inner splits строятся только внутри confirmation-fit. Новые кандидаты не получают confirmation-y до финального сравнения.
