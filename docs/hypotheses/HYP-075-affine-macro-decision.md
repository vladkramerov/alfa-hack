# HYP-075: аффинное правило решения для expected Macro-F1

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1102–1104, REQ1107–1109.
- Статус: Запланирована; реальные fit и оценки исполняет координатор после регистрации/source freeze.

## Утверждение и новизна

Train-only аффинные score `a_k*p_k+b_k` могут улучшить Macro-F1 существующего teacher. В отличие от H37 с множителями, этот механизм использует аддитивные class offsets из производных expected confusion objective. Это новая процедура решения, без переобучения старых learner.

## Метод и исходные условия — фиксировано до realfit

Только явно переданные training teacher probabilities и training class counts. Классы `np.unique(y)`; DataFrame columns выравниваются по именам, массив обязан уже иметь этот порядок. Вероятности нормируются по строке; дополнительной calibration или inverse-prior коррекции нет. Применение предполагает достаточную posterior calibration teacher; для weighted H48 это непроверенное ограничение, а не установленный факт.

`pi_k=count(y=k)/N`, `q_k=mean(decision=k)`, `t_k=mean(p_ik * 1[decision=k])`; `J=mean_k 2*t_k/(pi_k+q_k+1e-8)`. Производные дают `a_k=2/(pi_k+q_k+eps)` и `b_k=-2*t_k/(pi_k+q_k+eps)^2`, общий множитель `1/K` опущен. Инициализация `a=1,b=0`; **25 итераций, damping=.5, epsilon=1e-8**. На каждом шаге обновляются коэффициенты с damping, вычисляется новый hard argmax и ожидаемый training objective; сохраняется лучший шаг, включая исходный argmax. При равенстве остаётся более ранний шаг. Фактический training-label F1 и validation labels в выборе не используются. Все коэффициенты заморожены при query, batch statistics не вычисляются. `predict_proba` возвращает softmax аффинных score с тем же argmax: это не откалиброванный posterior.

Основание: [Ramaswamy, Narasimhan, Agarwal, 2015](https://arxiv.org/abs/1501.00287), cost-sensitive optimization over confusion matrices. Данная damped affine iteration — адаптация, а не дословный алгоритм статьи. Глобальная сходимость и concavity Macro-F1 не утверждаются.

## Критерий успеха — до запуска

На frozen inner protocol Macro-F1 должен превысить соответствующий cached teacher reference; перенос единственного заранее выбранного финалиста на exploratory outer сравнивается с сохранённым H48 `.6495311620074445`. Цель пользователя `>=.7794373944089334` проверяется отдельно; меньший положительный результат не означает достижения20%. Старые control fits запрещены. SHA/data/split/teacher provenance фиксирует координатор до запуска; без них fit не разрешён.

Синтетическая проверка реализации: finite-difference partials J; fixture с улучшением expected objective и ненулевыми offsets; валидация missing/invalid teacher; именованный порядок классов; одинаковые предсказания по одной строке и batch; train-only priors; pickle и JSON; конечные нормированные output. Эти проверки не оценивают модельную гипотезу на competition data.

## Запуски

Пока нет реальных запусков. Исходы и отдельные results artifacts добавляются координатором сразу после проверки.

## Текущий вывод

Не оценивалась; отсутствие гарантии качества и posterior calibration явно сохраняется.

SPEC011 result 2026-10-24T02:01:05.164393+03:00 HYP-075/inner/default: completed macro_f1=0.6212181165985486 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-075-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
