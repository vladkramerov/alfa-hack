# HYP-104: DART dropout деревьев для точного CE + soft Macro-F1

Статус: proposed; параметры фиксируются до реального fit, результат пока неизвестен.
Перед реализацией прочитана история HYPOTHESES.md. Новый механизм — случайное
отключение уже построенных деревьев при обучении с перенормировкой весов DART;
это не новый seed и не повтор scalar gbtree H92 или vector-leaf gbtree H101.
Общие векторные листья H101 здесь **не используются**: по одному скалярному
дереву на класс в каждом раунде, поддерживаемый установленной XGBoost 2.1.4 DART.

## Наблюдение, гипотеза и границы вывода

H92 inner при 300 раундах дал .6075155330908212, при 900 — .599340154183666:
больше раундов в этой процедуре ухудшили независимый inner. Это совместимо с
переобучением, но не устанавливает причину; из результата LightGBM нельзя
автоматически вывести поведение XGBoost/DART. Предположение H104: dropout
уменьшит зависимость нового дерева от фиксированного ансамбля старых деревьев
и улучшит обобщение при прямой оптимизации Macro-F1. Гарантии улучшения нет;
шумные конфликтующие labels и сложность objective также могут ограничивать качество.

## Фиксированная конфигурация

`DropoutTeacherBoost(VectorLeafTeacherBoost)` наследует preprocessing и точный
objective H72/H101 без изменения научного исходника H101. XGBoost 2.1.4,
`booster=dart`, `multi_strategy=one_output_per_tree`, hist/CPU, 300 раундов,
depth6, eta.05, lambda5, alpha0, min_child_weight1, gamma0, max_bin256,
subsample1 и все colsample1, seed42, nthread2. DART: rate_drop.1, skip_drop.5,
sample_type uniform, normalize_type tree, one_drop0. Один fixed-кандидат без grid,
validation early stopping и выбора числа раундов по query. Требуемая библиотека
уже установлена; upgrades и внешние training data отсутствуют.

L=.5*mean(-log p_true)+.5*(1-soft macro-F1), p=softmax(logits).
Используется тот же точный полный градиент по логитам, умноженный на N.
Положительная диагональ `max(.5*p*(1-p),.001)` — только численный суррогат,
не точная кривизна и не доказанная верхняя граница. При dropout objective
дифференцируется по текущим логитам ансамбля с отключёнными деревьями;
гарантия уменьшения не заявляется. Class weights отсутствуют, строки не удаляются.

TrainOnlyEncoder H72 обучает медианы, признаки для log1p, средние/масштабы и OHE
исключительно на fit; clipping стандартизованных значений8. Teacher — TRUE
grouped RAW OOF H88 для training и существующий независимый RAW query; cache
проверяет провайдер H92. Новых teacher fits нет, H48 meta-insample teacher не
подменяет этот вход. Колонки строго соответствуют sorted training classes.

`base_margin=logclip1e-12(normalized teacher)` передаётся train и query DMatrix.
XGBoost уже включает query margin в output_margin; он не добавляется повторно.
Для inference заданы `training=False` и явный диапазон `(0,num_boosted_rounds)`:
ни случайного отключения деревьев, ни адаптации к составу query. Ноль раундов —
технический teacher-only baseline без learner-fit и без float32-квантования.

## Протокол, стоимость и критерии

Сначала один inner по исходным RAW fold0 индексам; outer допускается только
при строгом превышении лучшего нового inner .6334156298867437 (H97), иначе
outer не запускать. Только выбранный по inner кандидат может получить один
original outer. Сравнение original outer с H48
.6495311620074445 отдельно от RAW teacher. Цель SPEC011: Macro-F1 не ниже
.7794373944089334; inner улучшение само по себе её не подтверждает. Просмотренный
ранее outer остаётся exploratory. До 900 секунд **суммарно** на все реальные
fit гипотезы, включая outer; runner распределяет этот бюджет. DART дороже
обычного boosting из-за повторного вычисления прогнозов старых деревьев;
унаследованный callback прерывает превышение времени без оценки partial model.

## Проверка совместимости до реального запуска

Синтетические 60x2, K3, два раунда на установленной 2.1.4 успешно приняли NxK
custom gradient/Hessian и teacher margins. TDD проверяет конечные разности
точного objective, реальные scalar DART trees и изменённые weight_drop после
training dropout, deterministic repeated/batched/permuted query, margin once,
JSON/pickle, unknown category, class alignment и zero-round baseline.
Метрика на реальных данных пока не получена; synthetic совместимость её не заменяет.

Официальная документация тега v2.1.4:
[DART parameters](https://github.com/dmlc/xgboost/blob/v2.1.4/doc/parameter.rst),
[Booster.predict training=False и iteration_range](https://github.com/dmlc/xgboost/blob/v2.1.4/python-package/xgboost/core.py).
Параметрическая справка рекомендует явный диапазон деревьев для DART query;
здесь дополнительно явно отключён training mode согласно API исходнику этой версии.

SPEC011 outcome 2026-10-24T03:33:58.022556+03:00 HYP-104/inner/default: timeout; None=None; reference=None; [report](../../results/2026-10-24/HYP-104-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
