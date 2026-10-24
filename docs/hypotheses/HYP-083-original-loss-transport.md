# HYP-083: временной ratio transport с исходным квадратичным loss

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: реализация и синтетические проверки готовы; real-data model fit и scientific score **не выполнены** при передаче.
- Основание: прямое поручение координатора реализовать один новый original-loss temporal transport после отрицательного incremental результата H81; source freeze и реальные запуски выполняет координатор.

## Утверждение и новизна

Условная монотонная функция отношения текущего и прошлого энергопотребления по изменению рейтинга может улучшить исходный RMSE по сравнению с H70, где коэффициент оценивается по одинаково взвешенным clipped log changes. Новый метод учитывает большой исходный потребительский масштаб пары через donor_y² и не ограничивается линейным log-ratio beta. H71 нормировала y по уровню рейтинга между разными зданиями и обучала ExtraTrees; здесь только within-profile временные пары и isotonic ratio curve. H76 direct-y CatBoost и H81 fuzzy matching не повторяются; старые H70 learned fits не переисполняются для получения state.

## Один зафиксированный метод до realfit

`OriginalLossTransport(alpha=.95, support_shrinkage=.25, age_decay=.25, group_minimum_pairs=100, group_shrinkage=200., log_change_cap=1.5)`.

1. Ключ State_Factor/building_class/facility_type/floor_area/year_built, canonical numeric representation как H70; ELEVATION не участвует. Все профили с хотя бы одним sameyear collision отвергаются целиком. Query донор — latest strictly-past training record; query в память не добавляется. ID не используется.
2. Только соседние training-пары одного retained профиля. Eligible pair: donor_y>0, current_y≥0, оба рейтинга finite. Общее y обязано быть finite и≥0. Известность current rating в паре — часть доступных training features, validation y не используется.
3. `q(rating)=ndtri(clip((100.5−rating)/100,.005,.995))`, x=Δq. Target ratio=`current_y/donor_y`, **без clipping observed ratio**; weight=`donor_y²`. Точная тождественная связь для любого candidate r: `donor_y²*(current_y/donor_y−r)²=(current_y−donor_y*r)²`. Поэтому каждая curve минимизирует исходную weighted-by-count сумму squared consumption errors, а не log-loss. Невыразимые finite ratio/weights отклоняются технической ошибкой, не скрытым удалением пары.
4. Global `IsotonicRegression(increasing=True,y_min=exp(−1.5),y_max=exp(1.5),out_of_bounds='clip')`. Bounds накладываются на fitted monotone curve; observed target ratios не обрезаются перед fit. Anchor r(0)=1 отсутствует: неизменный rating не доказывает неизменное годовое потребление, и такой anchor поменял бы заранее выбранный исходный loss.
5. Facility curve с теми же loss/bounds обучается только при≥100 eligible pairs. Результат shrink к global с постоянным weight=`count/(count+200)`. Эта регуляризующая смесь не объявляется отдельным точным минимумом общего объединённого objective; identity относится к loss каждого curve fit. Редкая/новая facility →global. Convex blend двух increasing curves сохраняет monotonicity/bounds.
6. На query `donor_hat=donor_y*r(Δq)`. Missing любой rating или отсутствие global eligible pairs →r1; донор сохраняется без rating adjustment. Weight=`.95*past_support/(past_support+.25)*exp(−.25*max(0,age−1))`. Output=`weight*donor_hat+(1−weight)*compatible_baseline`. Нет донора/ambiguous →baseline. Базовый predictor передаётся уже обученным; его fit не вызывается. Без него используется fit mean.

## Данные, критерий и воспроизводимость до запуска

Исходный regression train SHA256 `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775` из зарегистрированного provenance. Inner Year<4→4: сохранённый H69 RMSE42.46005247026025 и H70high30.663255379972725. Предпочтительный compatible fallback — тот же сохранённый H69, чтобы изменение относительно H70 касалось только transport. Не обучать H69/H70 повторно ради контроля. Для переноса нужен strict gain над H70 на inner, затем отдельно разрешённый outer Year<5→5 с H69RMSE39.39219382766771 / H70high34.318489927675174. Цель SPEC011 outerRMSE≤31.51375506213417; меньший выигрыш остаётся меньшим. Outer уже открыт и разведочный.

Max900сек; один config, без post-score настроек. Координатор до fit регистрирует ID, source/config/data/split SHA, создаёт отдельный results run и после fit сохраняет predictions/outcome/report. Ошибка запуска не научное опровержение. H83 source SHA256 при передаче `333e4e744688c2fe54cf5c22894ec083db6fa3ac29482253801f9c3b881fb26f`.

## Проверка реализации

`.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_sprint_ratio_transport.py -v`: **7/7**, 0.536сек. Проверены exact weighted-loss identity, ручной equal-score weighted optimum52/101, ловушка observed-ratio clipping100/0, monotonicity/bounds и ручной facility shrink, strict-past/sameyear/future, collisions, zero donors/current targets, missing ratings/no curve, permutation/pickle, invalid y и baseline.

При добавлении ручного shrink assertion тест сначала неверно предполагал out-of-bounds для query rating10; это была ошибка fixture, исправлена query rating0. Модель не менялась. Первый полный run в этот момент получил единственный такой test failure; финальный `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q`: **397 tests, OK**,8.743сек, три известных предупреждения PyparsingDeprecationWarning от matplotlib.

## Исход

Не оценивалась; scientific real-data fits0. Frozen implementation передана координатору.

SPEC011 result 2026-10-24T02:17:33.460579+03:00 HYP-083/inner/default: completed rmse=30.651578052812432 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-083-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:24:31.968307+03:00 HYP-083/outer/default: completed rmse=34.32053895104047 vscontrol=39.39219382766771; [report](../../results/2026-10-24/HYP-083-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.
