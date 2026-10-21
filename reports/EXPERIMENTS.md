# Все эксперименты команды

Дата: 21 октября 2026. Источник — сохранённые метрики, конфигурации и истории запусков.
В журнале 33 оценок экспериментов и 16 внутренних попыток Optuna. Отдельное финальное переобучение включено во время родительского запуска.

RMSE: ниже лучше. Macro F1: выше лучше, каждый класс имеет одинаковый вес. Accuracy — доля верных ответов; Weighted F1 учитывает размеры классов.

## Лучшие результаты основной валидации

| Задача | Запуск | Модель | Метрика |
|---|---|---|---:|
| regression | H4-zero-year | CatBoostRegressor | rmse=43.004639 |
| multiclass | H5-flow-ratios | RandomForestClassifier | macro_f1=0.628621 |

Ориентиры для дальнейшей работы с учётом дополнительной проверки: regression — H4-zero-year, multiclass — multiclass-forest-20261021-stage2. Малый основной прирост не считается устойчивым, если на другом split знак изменения обратный.

## Основная валидация

| ID | Задача / модель | Что проверили | Параметры | Разделение | RMSE | Macro F1 | Accuracy | Weighted F1 | Итог | Секунды |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|
| [constant-regression-reference](regression-20261021-195542.json) | regression / DummyRegressor | Постоянное среднее train | strategy=mean | Year_Factor < 5 train; == 5 validation | 56.799432 | — | — | — | Простой ориентир, вычислен внутри первого запуска | — |
| [constant-multiclass-reference](multiclass-20261021-195652.json) | multiclass / DummyClassifier | Всегда самый частый класс train | strategy=most_frequent | stratified 80/20; random_state=42 | — | 0.048414 | — | — | Простой ориентир, вычислен внутри первого запуска | — |
| [regression-20261021-195542](regression-20261021-195542.json) | regression / Ridge | Первый линейный ориентир; Ridge | alpha=1; solver=lsqr; tol=1e-6 (восстановлено по коду первого этапа) | Year_Factor < 5 train; == 5 validation | 50.079635 | — | — | — | Исходный ориентир / без парного сравнения | 2.000000 |
| [multiclass-20261021-195652](multiclass-20261021-195652.json) | multiclass / LogisticRegression | Первый линейный классификатор | solver=lbfgs; max_iter=3000; random_state=42 (по коду первого этапа) | stratified 80/20; random_state=42 | — | 0.428852 | 0.773184 | 0.756098 | Исходный ориентир / без парного сравнения | 23.260000 |
| [regression-forest-20261021-stage2](regression-forest-20261021-stage2.json) | regression / RandomForestRegressor | Замена линейной модели на Random Forest | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8 | Year_Factor < 5 train; == 5 validation | 47.470728 | — | — | — | Исходный ориентир / без парного сравнения | 30.290000 |
| [multiclass-forest-20261021-stage2](multiclass-forest-20261021-stage2.json) | multiclass / RandomForestClassifier | Замена линейной модели на Random Forest | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | — | 0.628538 | 0.829132 | 0.817671 | Исходный ориентир / без парного сравнения | 95.600000 |
| [regression-boosting-20261021-stage3](regression-boosting-20261021-stage3.json) | regression / CatBoostRegressor | Замена модели на CatBoost; категории OneHotEncoder | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 45.935836 | — | — | — | Исходный ориентир / без парного сравнения | 8.490000 |
| [multiclass-boosting-20261021-stage3](multiclass-boosting-20261021-stage3.json) | multiclass / CatBoostClassifier | Замена модели на CatBoost; категории OneHotEncoder | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | stratified 80/20; random_state=42 | — | 0.534100 | 0.819151 | 0.795942 | Исходный ориентир / без парного сравнения | 87.810000 |
| [regression-optuna-20261021-stage4-candidate](regression-optuna-20261021-stage4.json) | regression / CatBoostRegressor | Лучший по внутренней проверке кандидат Optuna, до решения о сохранении | {"depth": 8, "iterations": 700, "l2_leaf_reg": 3.7888252014533643, "learning_rate": 0.10390991237859405} | Year_Factor < 5 train; == 5 validation | 46.924138 | — | — | — | Кандидат отклонён | — |
| [regression-optuna-20261021-stage4](regression-optuna-20261021-stage4.json) | regression / CatBoostRegressor | Итог подбора: сохранена прежняя модель | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 45.935836 | — | — | — | Подбор не улучшил основную метрику | 44.530000 |
| [multiclass-optuna-20261021-stage4-candidate](multiclass-optuna-20261021-stage4.json) | multiclass / RandomForestClassifier | Лучший по внутренней проверке кандидат Optuna, до решения о сохранении | {"max_depth": 28, "max_features": 0.8, "min_samples_leaf": 4, "n_estimators": 100} | stratified 80/20; random_state=42 | — | 0.620352 | 0.825254 | 0.815989 | Кандидат отклонён | — |
| [multiclass-optuna-20261021-stage4](multiclass-optuna-20261021-stage4.json) | multiclass / RandomForestClassifier | Итог подбора: сохранена прежняя модель | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | — | 0.628538 | 0.829132 | 0.817671 | Подбор не улучшил основную метрику | 315.800000 |
| [C00-forest-control](hypotheses-20261021-stage5/C00-forest-control.json) | multiclass / RandomForestClassifier | Повтор прежнего Random Forest; контроль воспроизводимости | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | — | 0.628538 | 0.829132 | 0.817671 | Исходный ориентир / без парного сравнения | 79.360000 |
| [H1-balanced-forest](hypotheses-20261021-stage5/H1-balanced-forest.json) | multiclass / RandomForestClassifier | Только class_weight=balanced | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=balanced | stratified 80/20; random_state=42 | — | 0.613316 | 0.778430 | 0.803626 | Хуже контроля | 87.240000 |
| [R00-catboost-control](hypotheses-20261021-stage5/R00-catboost-control.json) | regression / CatBoostRegressor | Повтор CatBoost с OneHotEncoder; контроль | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 45.935836 | — | — | — | Исходный ориентир / без парного сравнения | 6.910000 |
| [H2-native-regression](hypotheses-20261021-stage5/H2-native-regression.json) | regression / CatBoostRegressor | Нативные категории CatBoost вместо OneHotEncoder | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 51.318227 | — | — | — | Хуже контроля | 6.330000 |
| [C00-catboost-control](hypotheses-20261021-stage5/C00-catboost-control.json) | multiclass / CatBoostClassifier | Повтор CatBoost с OneHotEncoder; контроль | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | stratified 80/20; random_state=42 | — | 0.534100 | 0.819151 | 0.795942 | Исходный ориентир / без парного сравнения | 86.870000 |
| [H2-native-multiclass](hypotheses-20261021-stage5/H2-native-multiclass.json) | multiclass / CatBoostClassifier | Нативные категории CatBoost вместо OneHotEncoder | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | stratified 80/20; random_state=42 | — | 0.541732 | 0.821490 | 0.799546 | Лучше контроля | 552.260000 |
| [H3-missing-flags](hypotheses-20261021-stage5/H3-missing-flags.json) | regression / CatBoostRegressor | Шесть индикаторов пропусков; прочее без изменений | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 46.802464 | — | — | — | Хуже контроля | 6.960000 |
| [H4-zero-year](hypotheses-20261021-stage5/H4-zero-year.json) | regression / CatBoostRegressor | Только year_built=0 заменён на пропуск | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 5 train; == 5 validation | 43.004639 | — | — | — | Лучше контроля | 6.680000 |
| [H5-flow-ratios](hypotheses-20261021-stage5/H5-flow-ratios.json) | multiclass / RandomForestClassifier | Четыре отношения объёмов/пакетов, знаменатель +1 | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=42 | — | 0.628621 | 0.827592 | 0.815667 | Лучше контроля | 90.510000 |
| [R00-forest-control](hypotheses-20261021-stage5/R00-forest-control.json) | regression / RandomForestRegressor | Повтор прежнего Random Forest для ансамбля | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8 | Year_Factor < 5 train; == 5 validation | 47.470728 | — | — | — | Исходный ориентир / без парного сравнения | 18.400000 |
| [H6-ensemble-regression](hypotheses-20261021-stage5/H6-ensemble-regression.json) | regression / EqualWeightEnsemble | Среднее прогнозов исходных леса и CatBoost, веса 50/50 | equal_weights=[0.5, 0.5] | Year_Factor < 5 train; == 5 validation | 44.616808 | — | — | — | Лучше контроля | 25.360000 |
| [H6-ensemble-multiclass](hypotheses-20261021-stage5/H6-ensemble-multiclass.json) | multiclass / EqualWeightEnsemble | Среднее вероятностей исходных леса и CatBoost, веса 50/50 | equal_weights=[0.5, 0.5] | stratified 80/20; random_state=42 | — | 0.578837 | 0.827136 | 0.810789 | Хуже контроля | 166.700000 |

## Дополнительные разделения

| ID | Задача / модель | Что проверили | Параметры | Разделение | RMSE | Macro F1 | Accuracy | Weighted F1 | Итог | Секунды |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|
| [S-regression-control](hypotheses-20261021-stage5/S-regression-control.json) | regression / CatBoostRegressor | Контроль на дополнительном разделении | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 4 train; == 4 validation | 45.518769 | — | — | — | Исходный ориентир / без парного сравнения | 1.800000 |
| [S-regression-candidate](hypotheses-20261021-stage5/S-regression-candidate.json) | regression / CatBoostRegressor | Кандидат H4-zero-year на дополнительном разделении | iterations=500; depth=6; learning_rate=0.05; l2_leaf_reg=3.0 | Year_Factor < 4 train; == 4 validation | 45.316735 | — | — | — | Лучше контроля | 1.800000 |
| [S-multiclass-control](hypotheses-20261021-stage5/S-multiclass-control.json) | multiclass / RandomForestClassifier | Контроль на дополнительном разделении | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=2026 | — | 0.608205 | 0.829531 | 0.816827 | Исходный ориентир / без парного сравнения | 35.200000 |
| [S-multiclass-candidate](hypotheses-20261021-stage5/S-multiclass-candidate.json) | multiclass / RandomForestClassifier | Кандидат H5-flow-ratios на дополнительном разделении | n_estimators=150; max_depth=20; min_samples_leaf=2; max_features=0.8; class_weight=None | stratified 80/20; random_state=2026 | — | 0.601301 | 0.829132 | 0.817276 | Хуже контроля | 38.370000 |

## Неудачная диагностика

| ID | Задача / модель | Что проверили | Параметры | Разделение | RMSE | Macro F1 | Accuracy | Weighted F1 | Итог | Секунды |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|
| [diagnostic-ols-regression](diagnostics/diagnostic-ols-regression.json) | regression / LinearRegression | LinearRegression | LinearRegression(); настройки по умолчанию, диагностический запуск | Year_Factor < 5 train; == 5 validation | 10615239987723.564453 | — | — | — | Численно нестабильная OLS; не использовать | 0.990000 |

## Технические повторы

| ID | Задача / модель | Что проверили | Параметры | Разделение | RMSE | Macro F1 | Accuracy | Weighted F1 | Итог | Секунды |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|
| [technical-repository-check-multiclass](diagnostics/technical-repository-check-multiclass.json) | multiclass / LogisticRegression | Первый линейный классификатор | solver=lbfgs; max_iter=3000; random_state=42 (по коду первого этапа) | stratified 80/20; random_state=42 | — | 0.428852 | 0.773184 | 0.756098 | Технический повтор, не новая гипотеза | 23.970000 |
| [technical-repository-check-regression](diagnostics/technical-repository-check-regression.json) | regression / Ridge | Первый линейный ориентир; Ridge | alpha=1; solver=lsqr; tol=1e-6 (восстановлено по коду первого этапа) | Year_Factor < 5 train; == 5 validation | 50.079635 | — | — | — | Технический повтор, не новая гипотеза | 1.930000 |
| [technical-verified-multiclass](diagnostics/technical-verified-multiclass.json) | multiclass / LogisticRegression | Первый линейный классификатор | solver=lbfgs; max_iter=3000; random_state=42 (по коду первого этапа) | stratified 80/20; random_state=42 | — | 0.428852 | 0.773184 | 0.756098 | Технический повтор, не новая гипотеза | 24.040000 |
| [technical-verified-regression](diagnostics/technical-verified-regression.json) | regression / Ridge | Первый линейный ориентир; Ridge | alpha=1; solver=lsqr; tol=1e-6 (восстановлено по коду первого этапа) | Year_Factor < 5 train; == 5 validation | 50.079635 | — | — | — | Технический повтор, не новая гипотеза | 1.530000 |

## Все внутренние попытки Optuna

Их метрики рассчитаны на внутренних split и не сравниваются напрямую с основной таблицей.

| ID | Задача | Параметры | Метрика | Секунды | Повтор trial |
|---|---|---|---:|---:|---|
| multiclass-optuna-20261021-stage4-trial-0 | multiclass | {"max_depth": 20, "max_features": 0.8, "min_samples_leaf": 2, "n_estimators": 150} | 0.596354 | 27.220000 | — |
| multiclass-optuna-20261021-stage4-trial-1 | multiclass | {"max_depth": 12, "max_features": 0.5, "min_samples_leaf": 1, "n_estimators": 150} | 0.570908 | 11.220000 | — |
| multiclass-optuna-20261021-stage4-trial-2 | multiclass | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 8, "n_estimators": 250} | 0.571901 | 23.070000 | — |
| multiclass-optuna-20261021-stage4-trial-3 | multiclass | {"max_depth": null, "max_features": 1.0, "min_samples_leaf": 1, "n_estimators": 250} | 0.591140 | 69.360000 | — |
| multiclass-optuna-20261021-stage4-trial-4 | multiclass | {"max_depth": 20, "max_features": 0.8, "min_samples_leaf": 2, "n_estimators": 150} | 0.596354 | 23.490000 | 0 |
| multiclass-optuna-20261021-stage4-trial-5 | multiclass | {"max_depth": 28, "max_features": 0.8, "min_samples_leaf": 4, "n_estimators": 100} | 0.597690 | 15.750000 | — |
| multiclass-optuna-20261021-stage4-trial-6 | multiclass | {"max_depth": 28, "max_features": 0.8, "min_samples_leaf": 4, "n_estimators": 100} | 0.597690 | 15.950000 | 5 |
| multiclass-optuna-20261021-stage4-trial-7 | multiclass | {"max_depth": 28, "max_features": 1.0, "min_samples_leaf": 4, "n_estimators": 100} | 0.593874 | 19.440000 | — |
| regression-optuna-20261021-stage4-trial-0 | regression | {"depth": 6, "iterations": 500, "l2_leaf_reg": 3.0, "learning_rate": 0.05} | 45.518769 | 2.140000 | — |
| regression-optuna-20261021-stage4-trial-1 | regression | {"depth": 8, "iterations": 500, "l2_leaf_reg": 4.426505507739147, "learning_rate": 0.0874116944886467} | 43.992092 | 2.970000 | — |
| regression-optuna-20261021-stage4-trial-2 | regression | {"depth": 4, "iterations": 400, "l2_leaf_reg": 8.605201659336437, "learning_rate": 0.0224831270535195} | 47.057078 | 1.440000 | — |
| regression-optuna-20261021-stage4-trial-3 | regression | {"depth": 7, "iterations": 700, "l2_leaf_reg": 11.135469245723131, "learning_rate": 0.020846957895078} | 45.947746 | 3.060000 | — |
| regression-optuna-20261021-stage4-trial-4 | regression | {"depth": 8, "iterations": 1000, "l2_leaf_reg": 1.0456446489411726, "learning_rate": 0.1410699020300044} | 44.751592 | 5.690000 | — |
| regression-optuna-20261021-stage4-trial-5 | regression | {"depth": 8, "iterations": 700, "l2_leaf_reg": 3.788825201453364, "learning_rate": 0.103909912378594} | 43.421465 | 3.940000 | — |
| regression-optuna-20261021-stage4-trial-6 | regression | {"depth": 6, "iterations": 800, "l2_leaf_reg": 2.4226293262246146, "learning_rate": 0.0657251405566256} | 44.371820 | 2.970000 | — |
| regression-optuna-20261021-stage4-trial-7 | regression | {"depth": 4, "iterations": 900, "l2_leaf_reg": 5.804889321664829, "learning_rate": 0.1479628341547509} | 44.590037 | 2.570000 | — |

Полные источники, commit, SHA-256 и конфигурации — в experiment_registry.json; плоский экспорт — EXPERIMENTS.csv. Технические повторы не являются независимыми подтверждениями.
