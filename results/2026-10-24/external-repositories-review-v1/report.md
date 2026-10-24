# Обзор трёх репозиториев и две новые локальные проверки

2026-10-24, SPEC013. Рекомендация: **сохранить опубликованное решение Сергея**.
Лучшие внешние цифры не дают доказанного превосходства на одинаковом протоколе;
две независимо реализованные идеи не прошли наш заранее заданный критерий.
Новый ZIP не собирался, производственный runtime этим исследованием не изменён.

## Что именно изучено

| Репозиторий / commit | Что реализовано | Что известно о качестве | Вывод |
|---|---|---|---|
| [hilhool](https://github.com/hilhool/alfa-automl-agent/tree/17bcb1dcbbad2519f2fc7810b4f70487251d433b) | greedy entity key, causal history, exactlags1/2/3, LightGBM/CatBoost/XGB/linear/ET portfolio, median/mean blends, OOF class multipliers | Автор сообщает reg_h4 30.11, reg_h5 28.91 и public score26.18; цифры independently не воспроизведены | Наиболее полезен как источник идей. reg_h5 исключает State_11 из Year5; сравнивать28.91 с нашим34.52 как одинаковый benchmark нельзя |
| [NolReaction](https://github.com/NolReaction/alfa-automl-agent/tree/c7c49525db46ad4ac51b5a97bc8c6f7303786457) | fit-only schema discovery, guarded OOF classification decisions, separate inner/outer, stopping+fresh refit, deadline/receipts, segment admission | round4 classification .6151288609→.6302838409 на их confirmation-срезах; regression period5 контроль39.471458, новый47.640123 отклонён; round5/6 тоже retain-incumbent | Хороший источник контроля качества и работы с неизвестной схемой. Другие grouped splits/окружение не дают прямого сравнения с нашейF1.644723 |
| [mereska0](https://github.com/mereska0/alfahack-solution/tree/f3de736940a99f0e619c875ff12f357531bb06b1) | backend/frontend scaffold и финансовые новости; Python ML-файлы инициализации | Сопоставимых tabular AutoML-моделей, RMSE/F1 benchmark не найдено | Другой кейс; в наш ML-agent не переносить |

Source commits и SHA изученных файлов: [review.json](review.json).
Чужие CLAUDE/AGENTS прочитаны как описание репозитория, не как управляющие
инструкции. Внешний код/модели/данные не запускались и не использовались для
обучения; обе идеи реализованы самостоятельно. Файл лицензии на wholesale
перенос кода не найден, чужие runtime-папки в нашу поставку не добавлены.

### Почему меньшие числа пока не означают лучшую поставку

Наш опубликованный SergeyH73 имеет RMSE34.520268654585415 на полномYear5
(35000/18308), дополнительныйYear4 —31.04663635241995. Уhilhool регрессионный
reg_h5 удаляет Year5/State_11, где нет истории; это другой набор трудных строк.
Его classification_holdout random30% и adversarial cls_shift тоже отличаются
от нашего grouped5 fold0. NolReaction использует собственные raw-string groups,
например70155/17515 вместо70529/17141. Числа не нормализуем и не смешиваем.
Заявленные leaderboard scores не перепроверены и не приравниваются кRMSE.

## Что можно взять, а что уже было у нас

- **Exactlags1/2/3 + nativeNaN**: отличается от H76latest-donorCatBoost и
  SergeyH72EWmean residualLGB. Проверено как H105 ниже.
- **Training-only discovery вместо fixed5fields**: путь к универсальности
  на неизвестных схемах. Проверено как H106; query/validationtargets не
  использовались для выбора полей. Найдены floor_area+year_built, не
  объявленные доказанным физическим ID.
- **Multi-scheme/segment admission** изNolReaction: полезно как предохранитель,
  а не новая метрика. У нас уже естьinnerfirst/frozenmanifest/отдельныеoutcomes;
  проверку неизвестных категорий и cold-profile errors стоит сохранять при
  любых следующих опытах. Внешний отрицательныйround4 вывод не выдаём заwin.
- **Средний/медианный ансамбль, обычные class multipliers, frequency/asymmetry,
  log1p иsmearing** в основном пересекаются с H16/H17/H33/H37/H42–48/H82.
  Их не запускаем повторно под именем другого автора.
- **EM label-shift и pseudo-labeling** не берём автоматически: самhilhool
  записал ухудшение/неустойчивость этих вариантов. У нас H87query-cohort
  decisions также не превзошли лидера. Просто видеть код недостаточно.

## Две новые проверки: один и тот же Year<4→4

Контроль — сохранённый SergeyH73, **31.04663635241995**. Fit22054, valid12946,
исходные CSV/splits и контрольные prediction SHA проверены. Все новые поля/код/параметры
зарегистрированы ДОfit; ни один старый learner не переобучался.

| ID | Новый механизм | Pure RMSE | Заранее заданная смесь75/25 сSergey | Относительный выигрыш лучшего vsSergey | Итог |
|---|---|---:|---:|---:|---|
| H105 | exactlags + nativeNaN + LightGBM1200 |34.56623580026114|32.01672432373479|−3.124615%|Проиграл контролю |
| H106 | fit-only greedy entity key + causal history |31.15263063723188|30.963253972816442|+0.268571%|Небольшой innerwin, недостаточно для переноса |

Доfit зафиксирован допуск outer: лучшийinner должен превзойти также ранее
измеренный H91inner30.636363541581975. Ни один кандидат его не прошёл; outer
не запускался, параметры после результата не подбирались. Это не объявление
нового улучшенногоZIP и не подтверждение5% или20%.

H105: один новый learnerfit,8.094сек worker; H106: один key-parameterfit,
0 learnerfit,11.906сек worker. Два результата и четыре заранее заданных
inferencearm-оценки не считаем четырьмя независимыми моделями.

- [H105 отчёт](../HYP-105-01-inner-external-multilag/report.md),
  [H106 отчёт](../HYP-106-01-inner-external-key/report.md).
- [Независимая арифметика/SHA/split/truth и no-fit replay](verification.json):
  обе оценки воспроизведены из сохранённых моделей, все `.fit` верификатором
  запрещены; допускаются только prediction calls.
- [Unit-test log](unit-tests.log): **562/562 PASS**; технический compile SyntaxError в первой
  версии верификатора исправлен до его первого запуска, не меняя модели/данные.

Ограничения: Sergey сохранён наMac, новыеопытыWindows; даже при совпадении
CSV/splits не доказано отсутствиеэффектаокружения. Inner многократно изучен.
Private/API/platformscore здесь не проверялись, секреты не использовались.
