# HYP-098: exact empirical Macro-F1 по аддитивным probability offsets

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: Запланирована; реальные данные при реализации не fit.

## Утверждение и новизна

Коррекция решения по фактическим training labels и честным OOF probabilities
может лучше учесть редкие классы, чем expected-confusion approximation H75.
Новый score **p_k+b_k**, а не log(p_k)+b_k: последнее эквивалентно H37
multipliers и здесь не используется. H98 напрямую оптимизирует empirical
Macro-F1 на fit labels, без assumptions о calibration ожидаемой confusion.
OOF-метрика самого offset-training остаётся оптимистической тренировочной
метрикой; самостоятельной независимой оценкой она не объявляется.

## Замороженный exact coordinate algorithm до fit

`sprint_empirical.EmpiricalMacroF1Offsets()`:

- Начальные offsets0. Не более **3passes**, классы посещаются по возрастанию
  training count, при одинаковом count — sorted training class index.
- Для координаты c фиксируются остальные offsets, вычисляются other-winner
  `h_i=argmax_{k!=c}(p_ik+b_k)` и margins `r_i=max_{k!=c}(p_ik+b_k)-p_ic`.
- Margins сортируются; при увеличении offset все строки с `r_i<threshold`
  переходят из своего other-winner в c. Кумулятивные integer predicted counts
  и TP для c/других классов дают exact Macro-F1 каждого полного prefix.
  Одна сортировка плюс O(NK) count curve на координату, без O(N²) перебора
  всех полных предсказаний и без grid/hyperparameter search.
- Кандидаты — midpoints **между различными** margin values и representatives
  снаружи крайних endpoints. Одинаковые margins переходят только целиком;
  абсолютно одинаковые probability rows не разбиваются по произвольному order.
- При отсутствии представимого float midpoint такой интервал пропускается.
  Перед принятием кандидат проверяется реальным argmax(p+b): если rounding
  породил boundary tie/разбиение, не совпадающее с полным prefix, кандидат
  отклоняется. Рассматривается следующий по exact curve score.
- Принимается только actual empirical gain **>1e-12**. Если текущий score
  равен лучшему, текущий offset сохраняется. Среди равных улучшений действует
  ранний threshold; после полного pass без gain поиск прекращается.
- После fit actual final F1 повторно проверяется: он не может быть ниже initial
  и должен совпадать с последней принятой count-curve метрикой.
- Frozen inference — `softmax(p+offset)`, argmax задаёт метку. Эти probabilities
  **не calibrated posterior**; никаких query-cohort статистик или адаптации.
- Explicit teacher обязателен, columns соответствуют sorted training classes;
  вход нормализуется построчно. X используется только для проверки числа строк.
  Никаких learners, feature statistics, ID, synthetic rows или удаления labels.

## Однозначный честный teacher protocol

До запуска root и implementer согласовали **RAW и для inner, и для outer**.
`experiments.sprint_honest_reuse.load_reused_honest_teacher` возвращает H88
proper nested3fold training OOF в inner и исходный raw3fold OOF в outer;
query всегда held-out raw teacher/H11. Все caches/SHA/classes/groups/splits
проверяются provider, новых teacherfits нет.

H48 meta-in-sample training probabilities здесь **не применяются**: это
особенно существенно для алгоритма, непосредственно подгоняющего true-label
F1. Возможные комбинации H78/H98 — отдельные новые гипотезы и не варианты H98.

## Данные и критерий успеха до fit

Classification trainSHA256
`171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`.
Inner исходный raw fold0 fit47075/valid23454. Utility — strict Macro-F1 gain
на held-out query против raw teacher `.5873380697311004`; при выборе finalist
сравниваются все заранее допущенные кандидаты на том же inner protocol.
Training fit F1 в audit для выбора hyperparameters/outer не применяется.

Outer исходный grouped70529/17141, единственный запуск только после выбора
финалиста до score. Контроль H48 `.6495311620074445` сохраняется отдельно от
raw teacher; текущий H78outer `.6593084142037399`, цель20%
**>=.7794373944089334**. Это честный raw pipeline, но сам outer уже просмотрен,
поэтому результаты разведочные. Никаких дополнительных H48 proxy вариантов.
Все fits/вычисления≤900сек суммарно для гипотезы; source/config/data/split/
teacherSHA фиксируются до запуска, отрицательные outcomes сохраняются.

## Техническая проверка

TDD:8tests сначала упали при отсутствии модуля; затем8/8green. Проверены
independent exhaustive threshold search через sklearn F1, ручной perfect-F1
пример, whole-tie-group behavior, current-state ties retained, rare-first
class order, strict monotonic accepted history, row permutation, query batching,
pickle, invalid teacher/schema/classes и отличие p+offset от logp+multipliers.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_empirical -v`.
Ни одного scientific/public data fit или teacherfit при реализации.

## Запуски

| Дата | Отчёт | Статус | Вывод |
| --- | --- | --- | --- |
| — | Координатор создаёт отдельный results/ до fit | Не запускалась | Не оценивалась |

## Следующий шаг

Root регистрирует H98, замораживает source/config/data/split/teacherSHA и
запускает один innerfit; wrapper общий ledger и harness не меняет.

SPEC011 result 2026-10-24T02:48:47.437692+03:00 HYP-098/inner/default: completed macro_f1=0.6208270606519142 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-098-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
