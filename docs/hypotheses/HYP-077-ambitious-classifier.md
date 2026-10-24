# HYP-077: ранговая Lovasz-Softmax коррекция teacher деревьями

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: Запланирована; реальные данные не обучались при реализации.
- Исход до experiment: **Не оценивалась**.

## Утверждение и новизна

Полные ранги ошибок вероятностей внутри каждого класса дают деревьям сигнал,
отличающийся от аддитивного cross entropy и сглаженных F1-сумм, и способны
улучшить Macro-F1 supplied teacher. Используется Lovasz extension Jaccard loss;
это surrogate macro-IoU, не тождественный Macro-F1, гарантии достижения20% нет.
От H72 отличается ранговым piecewise objective и tree learner; от H74 — loss
и LightGBM, от H75 — изучением нелинейных X-разбиений вместо affine decisions.
Не повторяет H50 focal, H51 balanced softmax, H37 multipliers, H40 pair-expert,
H49 distillation, H52 KNN, H73 ECOC. Полный HYPOTHESES.md прочитан до реализации.

Источник метода: [Berman et al., CVPR2018](https://arxiv.org/abs/1705.08790)
и [авторская реализация](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py).
Используется отдельная NumPy-реализация производной и fullbatch LightGBM learner;
torch, pretrained и external training data отсутствуют.

## Замороженный метод до realfit

`sprint_ambitious.LovaszTeacherBoost()`:

- Loss `0.5*mean CE + 0.5*mean_present LovaszSoftmax`.
- Для каждого присутствующего в train класса: descending stable argsort
  `abs(onehot_y-p)`, последовательные приращения Jaccard error задают ранговые
  веса; точный derivative по p и полный softmax chain по всем logits.
- Fullbatch gradient умножается на N для масштаба gradient boosting.
- Положительный диагональный surrogate curvature `max(.5*p*(1-p), .001)`.
  Это **не точный Hessian**, не его доказанная верхняя граница; convexity in
  logits и guaranteed descent не заявляются.
- LightGBM200rounds, learning_rate.05, leaves63, min_child_samples30,
  reg_lambda5, threads2, random_state42, deterministic/force_col_wise,
  boost_from_averageFalse, metricNone; никаких class weights.
- Train-only `TrainOnlyEncoder`: median numeric, log1p nonnegative columns,
  standardization/clip8 и unknown-safe OHE; строки не удаляются, label smoothing
  и синтетика отсутствуют. Точно те сырые X, которые разрешены harness.
- Teacher init_score — `log(clip(normalized supplied p,1e-12,1))`, columns
  соответствуют `np.unique(training_y)`. При inference query teacher logits
  явно прибавляются один раз к raw tree score; train init_score в модель не
  встроен. Учитель здесь никогда не обучается, cache не открывается.
- Все200rounds без validation early stopping; callback прерывает превышение900сек,
  partial model не считается завершённым scientific score. Родительский runner
  дополнительно ограничивает суммарный бюджет всех realfit этой гипотезы900сек.
- iterations0 предназначено для контракта teacher-only без learner fit, не
  отдельный кандидат на public data.

## Данные и честная оценка

Classification trainSHA256
`171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`.
Inner: существующий raw3fold OOF fold0, fit47075/valid23454; caller обязан
проверить provenance/indices и сохранить hashes. Готовый raw teacher MacroF1
`.5873380697311004`, H74 `.6143385510752261`; это proxy, не H48 fullpipeline.
Meta-in-sample H48 teacher на outertrain не является независимой innerоценкой.

Outer: исходный grouped fold seed42, fit70529/valid17141, только при выборе
H77 финалистом до score; готовый H48 `.6495311620074445`, новый реальный
порог20% `.7794373944089334`. Старый outer уже просмотрен, оценка разведочная.
Никаких ID-features, testlabels, query-cohort adaptation или старых controlfit.

## Критерий успеха — до запуска

Inner utility: MacroF1 строго выше готового сопоставимого raw teacher; при
выборе среди зарегистрированных кандидатов учитывается готовый H74 `.61433855`.
Настоящий успех20% определяется только original comparable outer MacroF1
`>=.7794373944089334`; меньший положительный выигрыш хранится отдельно.
Никакой параметр/состав не меняется после первого realfit score.

## Реализация и техническая проверка

TDD: до появления модуля7 synthetic tests упали с `H77 module missing`.
После реализации7/7 green; независимые finite differences по logits при
CEweight0/.5/1, hand-derived loss/gradient, extreme logits/absent class,
teacher-only0round, learning heldout context, teacher added once, query batch
invariance, pickle, teacher/schema validation, positive curvature contract.
Synthetic learner fits не являются проверкой гипотезы на competition data.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_ambitious -v`.
Известны внешние PyparsingDeprecationWarning при импорте LightGBM/matplotlib.

## Запуски

| Дата | Отчёт | Статус выполнения | Вывод |
| --- | --- | --- | --- |
| — | Координатор добавляет отдельный results/ до realfit | Не запускалась | Не оценивалась |

## Следующий шаг

Координатор регистрирует H77 в общем журнале, фиксирует source/data/split/config
SHA и запускает один inner в разрешённом harness; этот модуль общий журнал
не меняет. Технические тесты подтверждают вычисления, не улучшение качества.

SPEC011 result 2026-10-24T02:05:02.508902+03:00 HYP-077/inner/default: completed macro_f1=0.6110510854211656 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-077-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
