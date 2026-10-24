# HYP-081: консервативный fuzzy strict-past overlay поверх H70

- Дата: 2026-10-24, Москва.
- Связь: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: реализация и unit-проверки готовы; **реальный model fit и scientific score отсутствуют**.
- Основание: координатор выделил новый H81 после feature-only аудита H63; H63 не переименовывается в новую модель.

## Утверждение и границы новизны

Уникальный близкий физический профиль может восстановить небольшую часть временных связей, потерянных точным H70 из-за малых изменений площади или года постройки. Близость не доказывает идентичность здания. Метод добавляет только новые fuzzy-связи, никогда не меняет запрос с известным exact H70 profile key, включая неоднозначный training-key. Это новая схема соответствия с отказом при ближайшей конкурирующей связи, не повтор параметра alpha H70 или условного регрессора H76.

## H63: только feature-only диагностика, без метрики

Загружены только `Year_Factor,State_Factor,building_class,facility_type,floor_area,year_built` из `data/regression/train.csv`; `site_eui` не читалась. Для дизайна использованы исключительно строки Year_Factor<4. Ни queryYear4/5, ни validation y не использованы. Общий исходный SHA256 `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775` берётся из известного provenance проекта, сам этот аудит проверку SHA не повторял.

Проверялось ровно зафиксированное расстояние/ограничения ниже; donor profile с любым sameyear collision отвергался. По каждому retained профилю использована latest strictly-past запись, конкуренты сравнивались как distinct profiles.

| Query год | Past rows | Query rows | H70 exact support | Дополнительные fuzzy с margin | Из них changed area | Из них missing/0 year | Отклонены margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2117 | 9058 | 1633 | 53 | 31 | 19 | 3 |
| 3 | 11175 | 10879 | 6981 | 423 | 373 | 14 | 93 |

Дополнительное feature support на годе3:423/10879 =3.89 процентного пункта. Это **не научная метрика качества и не подтверждение правильной связи**. Без margin было56/516 дополнительных запросов соответственно. Начальный медленный диагностический скрипт дошёл до печати, но завершился `TypeError: int32 is not JSON serializable`; vectorized feature-only продолжение корректно вывело приведённые counts, научных fit/предсказаний не было. Это одна диагностическая подготовка, не повтор оценённой модели.

## Единственная конфигурация до realfit

`FuzzyTemporalOverlay(alpha=.5, area_tolerance=.001, year_tolerance=1., ambiguity_margin=.5, support_shrinkage=.25, age_decay=.25, beta_cap=2., log_change_cap=1.5)`.

1. Exact block: State_Factor/building_class/facility_type. Training profile key: block + exact floor_area/year_built; numeric key canonicalization совпадает с H70. Все sameyear collision profiles отвергаются целиком. Отсутствующая/неположительная площадь не участвует. ID не читается.
2. Candidate donor — latest strictly-past запись retained training profile. Relative area gap=`abs(a−b)/max(a,b)` ≤.001. Если оба года постройки finite и>0, их absolute gap≤1. Если хотя бы один year_built missing/0, площадь должна быть строго одинаковой.
3. Distance при известных годах=`relative_gap/.001+absolute_year_gap/1`; при missing/0 year distance=1. Candidate — profile, а не отдельная годовая строка. Второй distinct candidate должен быть дальше первого минимум на.5; иначе fallback. Несколько профилей одинаковой площади с неизвестным query-year дают одинаковый distance и отвергаются. При известных годах construction-year difference участвует в margin.
4. Если query exact key когда-либо присутствовал в fit, baseline остаётся неизменным, даже при отсутствии strict-past записи/коллизии. Для нового key применяются fuzzy-условия; matching никогда не использует y/rating. Поиск по sorted area index, без полной N×N матрицы.
5. `fit(X,y,base_predictor=None,transport_state=None)` строит только память и индекс. Base predictor — уже готовый совместимый **H70**. `transport_state` — независимая копия `rating_beta`/`group_rating_betas` из замороженного H70 audit; H70 для их получения не переобучается. Отсутствующий state означает beta0; невалидный state отклоняется. Caps фиксированы2/1.5.
6. Rating transfer использует H70 inverse-normal score ndtri(clip((100.5−rating)/100,.005,.995)); finite оба рейтинга → `transported=donor_y*exp(clip(beta_facility*delta_score,−1.5,1.5))`, иначе transported=donor_y. Это только reuse готового beta, не его новая оценка.
7. Weight=`.5*past_support/(past_support+.25)*exp(−.25*max(0,queryYear−donorYear−1))`; output=`weight*transported+(1−weight)*H70_prediction`. Unmatched/ambiguous/exact query output=H70_prediction. Без base predictor используется fit mean; в реальном опыте требуется совместимый H70. Query не добавляется в донорскую память.

## Протокол и критерий до запуска

Inner: Year<4→4, сравнение с сохранёнными H69 RMSE42.46005247026025 и H70high30.663255379972725. Outer при отдельном допуске: Year<5→5, H69RMSE39.39219382766771 / H70high34.318489927675174. Нужен strict gain над H70 для переноса именно overlay; цель SPEC011 остаётся outer RMSE≤31.51375506213417. Все исходы сохраняет координатор в отдельном results с source/data/split/config freeze. Старые H70/H69 не получают повторных scientific fit; production wrapper вправе обучить H70 один раз и передать его state/предиктор. Outer уже открыт, значит оценка разведочная. Max900сек.

## Unit-проверки

`tests/test_sprint_linkage.py`: **7/7** green; exact H70 unchanged, strict-past latest donor и support, margins/ties, sameyear collision, missing/0 year exactarea, frozen state copy/no estimation, block/area/year gates, pickle, row-order invariance, query-memory isolation, invalid targets/state/baseline. Полный `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q`: **367 tests, OK**,7.522сек; три известных PyparsingDeprecationWarning от matplotlib. Только синтетические fit индекса, научных fit0.

## Исход

Не оценивалась; реализованный метод передан координатору для регистрации/sourcefreeze. H63 диагностическое сходство не объявляется доказанной утечкой,20% улучшением или результатом H81.

SPEC011 result 2026-10-24T02:11:52.188392+03:00 HYP-081/inner/default: completed rmse=30.689409165993794 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-081-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
