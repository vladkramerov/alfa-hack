# SPEC015 / HYP-108: проверка TabM-mini

Дата: 24.10.2026, Москва. Статус: 4 научных запуска завершены; отдельная исследовательская реализация, основная поставка не менялась.

## Результаты

| Задача | Часть | Контроль | Только TabM-mini | Смесь | Доля TabM-mini | Относительный выигрыш |
| --- | --- | --- | --- | --- | --- | --- |
| regression | inner | 30.591603920 | 51.908456385 | 30.285884162 | 0.092833065 | 0.9994% |
| classification | inner | 0.628848166 | 0.516402108 | 0.628945310 | 0.100000000 | 0.0154% |
| regression | outer | 34.142888130 | 51.851896269 | 33.979840364 | 0.092833065 | 0.4775% |
| classification | outer | 0.659308414 | 0.509994445 | 0.660279593 | 0.100000000 | 0.1473% |

Для regression меньше RMSE — лучше; для classification больше Macro-F1 — лучше. На outer использованы веса, замороженные до чтения его прогнозов и таргетов: 0.0928330654623246 и 0.1. При оценке standalone TabM-mini значительно уступает контролю; измеренный небольшой выигрыш получен только в смеси. Это свидетельство комплементарности на этих разбиениях, а не замена текущей модели.

## Что именно проверено

Самостоятельная NumPy-реализация архитектуры TabM-mini: trainable random-sign входные адаптеры, общий двухслойный ReLU MLP, 8 отдельных голов, усреднение loss при обучении и прогнозов/вероятностей при инференсе. Это не официальный пакет TabM и не репликация чисел статьи. Параметры: width64,60epochs,batch256,AdamW lr.003/wd.0001; все 4 запуска завершили60epochs. Preprocessing и статистики target обучаются только на fit; удаления строк и ручной подгонки классов не было.

Код: [experimental_tabm.py](../../../experimental_tabm.py); [runner](../../../experiments/tabm_trial.py); [spec](../../../docs/specs/SPEC-015-tabm-trial.md); [карточка](../../../docs/hypotheses/HYP-108-tabm-mini.md).

Окружение: .venv-ml/Python3.12.14/Windows, NumPy1.26.4, pandas2.2.3, sklearn1.5.2; только разрешённые установленные зависимости. Новый torch/tabm не установлен агентом. [Проверка установки и точные сетевые ошибки](../tabm-environment-check/report.md). Пользователь установил torch2.14.1+cpu в другую .venv; требования платформы фиксируют2.5.1.

## Проверки и артефакты

- [6 новых тестов](tabm-tests.log): PASS, в том числе независимые конечные разности всех параметров сети, обучение синтетических задач, сохранение классов, неизвестные категории/NaN и сериализация.
- [Независимый no-fit replay](verification.json): SHA входных данных и snapshots, splits и отсутствие пересечения групп/будущих лет, targets/classes/positions, восстановление predictions из model.joblib, пересчёт всех RMSE/F1, перенос замороженных весов. Модели при сверке не обучались.
- [Первое техническое расхождение verifier](verification-roundoff-note.md): float32/float64 промежуточная арифметика; исправлен только проверяющий код, научные прогнозы/метрики не менялись, допуск не ослаблялся.
- [Полная серия unittest](tests.log): 574 tests,569passed/5errors. Все5 ошибок в существующем test_report_sprint.ReportPublicationTests, WinError5 при os.link. Они [воспроизведены отдельно](existing-report-tests-isolated.log),6tests/5errors; этот модуль не изменялся. Общая серия НЕ зелёная.

Имена пяти существующих ошибок: test_existing_pair_never_changes; test_second_create_failure_cleans_outputs_and_allows_retry; test_second_publish_collision_preserves_existing_file_and_rolls_back_own; test_second_write_failure_closes_fds_cleans_outputs_and_allows_retry; test_success_only_publishes_complete_files.

Каждая папка запуска содержит report.md, request.json, evidence.json, configuration-before-fit.json, configuration.json, audit.json, splits.npz, validation_predictions.npz, model.joblib, worker.log и frozen source/. Крупные fitted artifacts исключены из Git существующими правилами и сохраняются локально в results.

Команды:

```powershell
.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_tabm_trial -v
.venv-ml/Scripts/python.exe -X utf8 -m experiments.tabm_trial --kind regression --stage inner
.venv-ml/Scripts/python.exe -X utf8 -m experiments.tabm_trial --kind classification --stage inner
.venv-ml/Scripts/python.exe -X utf8 -m experiments.tabm_trial --kind regression --stage outer
.venv-ml/Scripts/python.exe -X utf8 -m experiments.tabm_trial --kind classification --stage outer
.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q
.venv-ml/Scripts/python.exe -X utf8 -m experiments.verify_tabm_trial
```

Четыре научные команды уже выполнены; повтор H108 запрещён. Runner защищает от повторной попытки, а Memory фиксирует started до fit и terminal после него. [Реестр запусков](attempts.json), [реестр уникальных fit](fit-history.json), [таблица в JSON](comparison.json). Новых learner fits4, старых learner refits0, verification fits0.

## Ограничения и решение

Контроли cached: H107 regression и H78 classification. Регрессионный контроль содержит исследовательскую cross-OS композицию; все outer-разбиения уже многократно использовались ранее и являются exploratory. Это не независимая проверка hidden score или установленная статистическая значимость, не доказательство качества runtime-графа.

Внедрение в LLM schema/portfolio/solution, изменение baseline и пересборка ZIP не выполнялись. Исследовательский компонент пригоден для дальнейшей интеграции, но перенос blend весов на новые задачи должен обучаться из их train, а не копировать опубликованные коэффициенты. Действующий агент пока не выбирает TabM автоматически.

Следующий технический шаг при продолжении: разрешённое целевое PyTorch-окружение и отдельно спроектированная интеграция train-only blending в агент, с сохранением fallback и памяти гипотез; не переобучать H108 просто из-за смены backend.

## Датированное техническое дополнение, 24.10.2026

Финальный no-fit replay 4/4 PASS: verification.json. Исправлена переносимость classification-y в NPZ: отдельные Unicode-копии без object pickle, все значения совпадают с сохранёнными оригиналами (archive-codec-fix.json). Оригинальные NPZ не перезаписывались; модели не переобучались. Добавлен седьмой тест — prediction archive без pickle. Последующая общая серия сохранена отдельно в tests-after-codec.log; итог исходного запуска tests.log не переписывался.

Финальная общая серия: 575 tests,570passed/5existing errors test_report_sprint; все7 новых tests прошли. Защищённые solution.py/baseline.json/requirements.txt/automl_models.py/automl_llm.py сохранили SHA. Сводная проверка: final-check.json.
