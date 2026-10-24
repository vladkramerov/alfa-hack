# HYP-095: generalized cross entropy с честным teacher и train-class балансом

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: Запланирована; realdatasetfit при реализации0.

## Утверждение и новизна

Известны противоречивые y при одинаковых raw X; это не позволяет установить
истинный класс каждой строки или конкретный механизм label noise. Гипотеза:
Generalized Cross Entropy уменьшит влияние неправдоподобных наблюдаемых меток,
сохраняя train-class баланс, и улучшит Macro-F1. Гарантии пользы для редких
классов нет: уверенно ошибочный teacher также может подавить полезную метку.

Основа — [Zhang and Sabuncu, NeurIPS2018](https://arxiv.org/abs/1805.07836).
Перенос с рассмотренных в статье DNN/noise условий на табличный residual
boosting является новой гипотезой, а не доказанным свойством этих данных.

H49 меняла student targets, H39 назначала OOF difficulty weights, H50 focal
усиливала трудные метки, H51 меняла prior внутри softmax. Здесь наблюдаемый y
сохранён, а динамический множитель `p_true**.7` **подавляет** маловероятные метки.
Строки не удаляются, точные конфликты не превращаются в lookup labels.

## Frozen метод до fit

`sprint_robust.RobustTeacherBoost()` наследует teacher-offset fit/inference из
LovaszTeacherBoost, заменяя objective и train-only class weights:

- `q=.7` fixed; loss `L=mean[a_y*(1-p_y**q)/q]`.
- `a_k=1/sqrt(n_k)` по соответствующему fit y, затем все a делятся на
  `mean_i(a_yi)`; средний sample weight ровно1. Никаких query/outer class counts.
- Exact derivative по logits:
  `dL/dz_i=a_yi*p_yi**q*(p_i-onehot_yi)/N`.
  Возвращаемый LightGBM gradient умножается на N. Это настоящая производная
  изменённого loss, а не detached approximation cross entropy.
- Positive surrogate curvature `max(a_y*p_k*(1-p_k),.001)`. Это **не точный
  GCE Hessian**, не доказанная его верхняя граница. Nonconvex optimization и
  guaranteed descent не заявляются.
- Stable logsoftmax, вычисление loss через expm1, никаких probability clipping
  в objective. q0 helper — точный continuous CE limit, не scientific вариант.
- LightGBM300rounds/leaves63/min_child_samples30/lr.05/reg_lambda5/random_state42,
  deterministicTrue/force_col_wiseTrue/boost_from_averageFalse/metricNone,
  threads2. Один fixed вариант, без earlystop/validation tuning.
- Прежний train-only median/log1p nonnegative/standardization/clip8/OHE.
- Teacher обязателен: **proper grouped OOF** из read-only H88 reuse provider,
  train nested3fold для inner и old raw3fold для outer, query held-out raw.
  H48 meta state не используется. Никаких новых teacherfit.
- Init logits `log(clip(normalized teacher,1e-12,1))`; inference прибавляет
  query teacher ровно один раз к tree raw logits и делает softmax.
- Audit сохраняет actual classes/counts/class_weights/sample_weight mean/min/max.
  До900сек суммарно/hypothesis включая возможный outer, до4CPU,
  frozen default2CPU. Состав после первого realfit score не меняется.

## Данные и критерии успеха до запуска

Classification trainSHA256
`171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`.
Inner исходный raw fold0 fit47075/valid23454; caches H88 проверяет
`experiments.sprint_honest_reuse.load_reused_honest_teacher` без каких-либоfit.
Источники, data/split/teacherSHA фиксирует coordinator до запуска H95.

Utility — strict inner Macro-F1 gain над сопоставимым raw teacher
`.5873380697311004`; при выборе финалиста учитываются H78inner
`.6288481661155734` и все заранее допущенные honest-objective результаты.
Пользовательская цель20% определяется на исходном grouped outer70529/17141:
MacroF1 **>=.7794373944089334** при готовом H48 reference.6495311620074445.
Текущий H78outer.6593084142037399 — отдельный reference меньшего выигрыша.
Outer уже просмотрен, оценка разведочная; допуск только после inner выбора.
Показать per-class precision/recall и не скрывать ухудшение rare classes.

## Техническая проверка

TDD:5tests сначала упали с отсутствующим модулем, затем5/5green за.971сек.
Independent finite differences weighted q.7/q0 совпадают с exact gradient;
проверены ручные CE/GCE значения, подавление unlikely-label gradient,
extreme±1000 logits, invalid weights, inverse-sqrt counts/mean1, положительная
curvature,0round teacher baseline, query teacher ровно один раз, batching,
real synthetic learner и pickle. Scientific competition data не обучались.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_robust -v`.

## Запуски

| Дата | Отчёт | Статус | Вывод |
| --- | --- | --- | --- |
| — | Координатор создаёт отдельный results/ до fit | Не запускалась | Не оценивалась |

## Следующий шаг

Регистрация H95 в общем журнале и source/config/data/split/teacher freeze,
один innerfit, outcome сразу в отдельный results/. Никаких old controlfit.

SPEC011 result 2026-10-24T02:41:27.858100+03:00 HYP-095/inner/default: completed macro_f1=0.6227569652507932 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-095-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
