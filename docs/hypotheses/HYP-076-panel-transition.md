# HYP-076: условная модель временного перехода энергопотребления

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: реализация и синтетические проверки; **реальные данные ещё не обучались**.
- Основание: поручение координатора реализовать отдельный conditional temporal transition, заморозить один конфиг; регистрацию и realfit выполняет координатор.

## Утверждение и новизна

Модель условного среднего текущего энергопотребления, обученная на парах уникальных записей одного физического профиля, может учитывать совместное влияние прошлого потребления, изменения рейтинга и погоды лучше параметрического переноса H70. Это supervised regression **current y | current raw X, past y, past X**: не поправка остатка baseline H56 и не нормализатор рейтинга H71. H69/H70 используют донорское значение с заданным правилом, здесь отдельная CatBoost-модель обучает условный переход.

## Зафиксированный метод до realfit

Данные: `data/regression/train.csv`, SHA256 `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`. Inner: Year_Factor<4 →4 внутри исходного outer-fit; совместимый сохранённый H69 RMSE42.46005247026025. Outer: Year_Factor<5 →5; сохранённый H69 RMSE39.39219382766771. Модель не использует validation y и не переобучает baseline. Перед запуском координатор фиксирует SHA кода, данных, индексов и configuration().

1. Ключ строго равен `State_Factor/building_class/facility_type/floor_area/year_built`; пропуск отдельный токен, не wildcard. Любой повтор ключа в одном training-году отвергает весь профиль. ELEVATION не часть ключа.
2. Обучающие пары — только соседние по времени уникальные training-записи профиля с donorYear<currentYear; первая запись профиля не имеет pair target. Порядок обхода — sorted key/year. На inference только latest strictpast training donor; query не добавляется в память.
3. Признаки: текущие исходные X без `id/target/site_eui/label`; donor_y, donor_rating, rating_delta, два missing-rating flags, age. Для каждого numeric current X вне LINK_FIELDS/Year_Factor/rating добавляются donor value и current−donor delta; это включает погодные измерения и ELEVATION. Статистики validation не вычисляются. Категории — строки с `__MISSING__`, числовые NaN обрабатывает CatBoost; inf→NaN. Query категории допускаются неизвестными.
4. **Один конфиг** CatBoostRegressor: iterations800, depth6, learning_rate.05, l2_leaf_reg10, loss_functionRMSE, random_seed42, thread_count4, verboseFalse, allow_writing_filesFalse. Все пары имеют единичные веса, target=current y в исходных единицах; log/ratio/нормализации target нет. Минимум32 пары и непостоянный target; иначе только baseline.
5. Для совпавшей query и обученной transition-модели: `prediction=.9*max(0,transition)+.1*baseline`. Фиксированная alpha=.9, без выбора по inner/outer; при отсутствии подходящего донора только baseline. Без переданного baseline используется fit mean. Невалидный baseline/transition отклоняется.

Выбор умеренной глубины и l2=10 сделан до realfit: пары являются меньшей и коррелированной выборкой, донорский target уже сильный признак. 800 небольших шагов дают возможность нелинейных взаимодействий без глубокой таблицы запоминания. Это инженерное допущение, не подтверждённый результат.

## Критерий успеха до запуска

Нужен strict RMSE gain над совместимым сохранённым H69. Для цели SPEC011 outer RMSE≤31.51375506213417; меньший выигрыш хранится как меньший. Inner сравнение с H70 возможно только по уже сохранённым predictions; старые контроли не fit. Outer уже открыт и остаётся разведочным. Max900сек на гипотезу; техническая ошибка не опровержение. Конфиг после score не меняется.

## Проверка реализации

`tests/test_sprint_panel.py`: strict-past train pairs/query donors, sameyear/future exclusion, full-profile collision rejection, missing не wildcard, donor target и weatherdelta, missing rating, ID exclusion, реальный CatBoost на синтетике, unknown category, pickle и невалидный baseline/target. Тестовые параметры iterations12/thread1 используются только для синтетики и не являются другим реальным кандидатом.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_sprint_panel.py -v`.

Фактическая проверка до передачи: **6/6** synthetic H76 green, включая row-order invariance и отсутствие query→donor mutation. Полный `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q`: **310 tests, OK**, 7.470сек; три предупреждения PyparsingDeprecationWarning из существующего matplotlib. Первый параллельный full-suite run имел единственный import error `test_sprint_posterior`: другой агент ещё не создал sprint_posterior.py; после появления модуля полный run green. Это unit-проверка, не проверка научной гипотезы и не real-data score.

## Реальные запуски и вывод

Реальных fit/новых validation scores: **0** при передаче координатору. Результат гипотезы: **Не оценивалась**; registration, source freeze, отдельный results/report и исход заполняет координатор до/после запуска.

SPEC011 result 2026-10-24T02:04:32.326547+03:00 HYP-076/inner/default: completed rmse=32.91081846583218 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-076-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
