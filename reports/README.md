# Журнал экспериментов

Метрики получены на локальной валидации, не на скрытом test. LLM не используется.

| Запуск | Задача | Модель | RMSE ↓ / Macro F1 ↑ | Время, сек |
|---|---|---|---:|---:|
| [multiclass-20261021-195652](multiclass-20261021-195652.json) | multiclass | LogisticRegression | 0.428852 | 23.26 |
| [multiclass-boosting-20261021-stage3](multiclass-boosting-20261021-stage3.json) | multiclass | CatBoostClassifier | 0.534100 | 87.81 |
| [multiclass-forest-20261021-stage2](multiclass-forest-20261021-stage2.json) | multiclass | RandomForestClassifier | 0.628538 | 95.60 |
| [multiclass-optuna-20261021-stage4](multiclass-optuna-20261021-stage4.json) | multiclass | RandomForestClassifier | 0.628538 | 315.80 |
| [regression-20261021-195542](regression-20261021-195542.json) | regression | Ridge | 50.079635 | 2.00 |
| [regression-boosting-20261021-stage3](regression-boosting-20261021-stage3.json) | regression | CatBoostRegressor | 45.935836 | 8.49 |
| [regression-forest-20261021-stage2](regression-forest-20261021-stage2.json) | regression | RandomForestRegressor | 47.470728 | 30.29 |
| [regression-optuna-20261021-stage4](regression-optuna-20261021-stage4.json) | regression | CatBoostRegressor | 45.935836 | 44.53 |

Сравнивайте одинаковые задачи, разделения и контрольные суммы данных.
Время включает обучение на части train, валидацию, переобучение на всём train и прогноз test.
Полные CSV остаются в runs/ и не включаются в Git. Для публикации reports/ нужен commit и push.
