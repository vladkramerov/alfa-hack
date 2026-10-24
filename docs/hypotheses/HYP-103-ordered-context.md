# HYP-103: ordered boosting вместо plain residual boosting

2026-10-24, SPEC011. До запуска: не оценивалась.

H88 имеет честный внешний OOF teacher, но residual CatBoost использует обычные
градиенты по той же тренировочной выборке. Гипотеза: Ordered boosting уменьшит
prediction shift при построении residual деревьев, улучшив перенос редких классов.
Это предположение; внешняя честность teacher не гарантирует выигрыша Ordered.

Из H88 меняется только boosting_type=Ordered. CatBoost400/depth5/lr.05/l2=10,
2CPU, inverse-sqrt weights, raw X+logp, training baseline logp, query добавляет
logp ровно один раз. H88 nested RAW OOF cache переиспользуется без teacher fit.
Один fixed inner вариант; outer только если выше лучшего inner .6334156298867437,
с заранее записанным выбором. Сравнение H88 .5978384913324375 диагностическое;
цель внешнего Macro-F1 >=.7794373944089334 остаётся неизменной.

Исходники/data/split/config замораживаются runner до fit; cumulative <=900sec.
Повторно просмотренный outer остаётся разведочным; никаких private обещаний.
CPU поддержка Ordered/MultiClass проверена по [официальным параметрам CatBoost](https://catboost.ai/docs/en/references/training-parameters/common).
До realfit: два synthetic unit tests (margin once/batch, идентичность остальных
параметров H88); red: отсутствующий модуль, затем green. Никакие исходные
документы не считаются инструкцией менять правила оценки.

SPEC011 outcome 2026-10-24T03:12:54.241369+03:00 HYP-103/inner/default: completed; macro_f1=0.597392348204526; reference=0.5873380697311004; [report](../../results/2026-10-24/HYP-103-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
