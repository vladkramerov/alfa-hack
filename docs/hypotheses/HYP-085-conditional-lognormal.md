# HYP-085: условное lognormal mean через совместные mu и variance

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: Запланирована; реальные competition data при реализации не обучались.
- Полный HYPOTHESES.md прочитан перед реализацией; H33/H71 не повторяются.

## Утверждение и отличие

Если `log1p(y)|X` приближённо Gaussian с неоднородной дисперсией, совместное
изучение `mu(X)` и `sigma²(X)` позволяет восстанавливать условное среднее
`exp(mu(X)+.5*sigma²(X))-1`. Это проверяемое модельное допущение, не свойство,
установленное для данного набора. Objective — Gaussian NLL на log1p(y),
не точный RMSE в исходных единицах; качество оценивается именно original RMSE.

H33 использовала глобальный Duan smearing factor после OOF log-target model;
здесь отдельная conditional variance head совместно обучается с mean head.
H71 нормализовала y по rating с g² weights; здесь нет rating target-normalizer,
глобального/классового smearing, sample weights или дополнительного baselinefit.

## Источник и установленный CatBoost

Проверена [официальная документация loss](https://catboost.ai/docs/en/concepts/loss-functions-regression#RMSEWithUncertainty).
Raw second output интерпретируется как `log(sigma)`, поэтому variance=`exp(2*a1)`.
При попытке открыть [uncertainty reference](https://catboost.ai/docs/en/references/uncertainty)
web returned timeout; это не подтверждение содержимого той страницы.

В установленном CatBoost отдельная tiny synthetic fixture проверила
`prediction_type='RMSEWithUncertainty'[:,1] == exp(2*RawFormulaVal[:,1])`.
Получено `.020600829755654797` против `.0206008297556548`; это variance,
несмотря на неоднозначный installed Python docstring о standard deviation.
Тот же контракт проверяется unit-тестом на настоящем tiny estimator.

## Frozen конфигурация до realfit

`sprint_uncertainty.ConditionalLognormalRegressor(threads=2)`:

- CatBoostRegressor800iterations/depth6/learning_rate.05/l2_leaf_reg10,
  loss RMSEWithUncertainty, random_seed42, thread_count2, verboseFalse,
  allow_writing_filesFalse; all rounds, no validation early stopping.
- Target `log1p(y)`, только конечные y>=0, никаких удалённых строк/весов.
- Raw allowed X → существующий train-only FeatureComposition29/30/31
  (wind cyclic/monthly harmonics/group-relative). Источник ID/target удаляет
  caller до fit, как в других разрешённых regression candidates.
- Numeric inf→NaN; native categories tagged `V:<value>`/`M:` без смешения
  missing и буквального sentinel. Никаких Gaussian/quantile transforms.
- Prediction raw columns=`mu, log_sigma`. Conditional variance capped at2
  **до** экспоненцирования чрезмерных log_sigma. Это заранее заданный cap,
  не оценённая граница по данным.
- Итог `expm1(clip(mu+.5*variance,0,700))`. Fixed log_sigma floor−350 и
  exponent upper700 — численная защита, lower0 задаёт nonnegative output;
  они не используют training/validation maxima или outer labels.
- `.fit(X,y,base_predictor=None)` принимает общий harness API, но supplied
  baseline не вызывает, не обучает и не хранит. `.predict(X)`, configuration(),
  audit_ переносимы, cache paths/внешние данные отсутствуют.
- До900сек на гипотезу суммарно inner+возможный outer, лимит контролирует
  координатор. Один learnerfit на split, до4CPU, frozen default2CPU.

## Данные и критерий успеха — до запуска

Regression trainSHA256
`096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`.
Inner — trainYear<4→Year4 внутри исходного outer-fit, готовый сопоставимый
H69 RMSE42.46005247026025; H70inner30.663255379972725 — текущий сильнейший
reference для выбора финалиста. Utility подтверждается только strict
уменьшением original RMSE относительно сопоставимого контроля, не log-NLL.

Outer — исходный trainYear<5→Year5, fit35000/valid18308, только после выбора
финалиста до score. Готовые H69 RMSE39.39219382766771 и H7034.318489927675174;
цель20% **<=31.51375506213417**. Старый outer уже открыт, результат
разведочный. Не подбирать variancecap/featurecomposition после score.
Возможная будущая комбинация с H70 требует отдельной новой регистрации и
положительной inner complementarity до outer, в H85 не включается.

## Техническая проверка

TDD:4tests сначала упали из-за отсутствующего модуля, затем4/4green за2.426сек.
Проверены ручное mean reconstruction, влияние conditional variance, cap и
extreme finite parameters, invalid targets/schema, fitted heteroscedastic toy,
installed CatBoost raw↔uncertainty convention, unknown/missing categorical,
query batching, pickle, source frame неизменен, baseline не используется.
Synthetic learnerfits не дают scientific метрику competition data.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_uncertainty -v`.

## Запуски

| Дата | Отчёт | Статус выполнения | Вывод |
| --- | --- | --- | --- |
| — | Координатор создаёт results/ и frozen manifest перед fit | Не запускалась | Не оценивалась |

## Следующий шаг

Координатор регистрирует H85 в общем журнале, фиксирует source/config/data/split
SHA и запускает один innerfit. Никакого скрытого refit старых контролей.

SPEC011 result 2026-10-24T02:20:38.569531+03:00 HYP-085/inner/default: completed rmse=46.02178005161315 vscontrol=42.46005247026025; [report](../../results/2026-10-24/HYP-085-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:27:12.563499+03:00 HYP-085/outer/default: completed rmse=44.12424608516781 vscontrol=39.39219382766771; [report](../../results/2026-10-24/HYP-085-02-outer-default-sprint/report.md). Повтор этой конфигурации запрещён.
