# Реестр гипотез и результатов

Обновлено: 2026-10-23 (Москва). Перед новым fit проверь данные и их SHA256, метрику, split, preprocessing и модель. Результат с другого split, иной версией данных или иной policy не считать прямым контролем. Дубликат seed/имени не является новой гипотезой.

Таблицы ниже — сохранённые исторические оценки с публичных задач. Они не являются результатами нового nested engine и не доказывают качество на hidden test. Исторический срез; актуальные public-v1…v5 и новая серия: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Сопоставимые с текущей SPEC-004 outer-схемой оценки

Данные: regression train `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`, `Year_Factor <5 → 5`, 35 000/18 308; multiclass train `171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`, first `StratifiedGroupKFold(5, shuffle=True, seed=42)` by exact raw feature groups, 70 529/17 141. В этих пределах оценки сопоставимы: одинаковая целевая метрика и сохранённые позиции validation. Внешняя проверка сверяла артефакты и значения, но не обучала модели заново.

| Гипотеза / модель | Конфигурация или отличие | Split / задача | Метрика | Статус evidence |
| --- | --- | --- | ---: | --- |
| HYP-012 ExtraTrees, log target | `n_estimators=200`, `max_depth=24`, `min_samples_leaf=2`, `max_features=0.8`, `random_state=42`; `log1p(y)` и обратное `expm1` | Regression temporal outer | RMSE **42.196700** | Лучший сохранённый результат SPEC-004 на этой схеме; подтверждён сохранёнными прогнозами. Исторический, outer уже использован для отбора. |
| HYP-011 XGBoost | `n_estimators=600`, `max_depth=6`, `learning_rate=.05`, `reg_lambda=1`, `tree_method=hist`, `device=cpu`, seed 42 | Regression temporal outer | RMSE 43.510161 | Проверена; уступила HYP-012. |
| HYP-021 ExtraTrees, log target + mandatory full-v1 | full-v1, scaling, до 5% train-only outlier filter, нормальные quantile transforms | Regression temporal outer | RMSE **43.837202** | Новее SPEC-004, но хуже на 1.640502. Составное изменение; отдельный эффект компонентов неизвестен. |
| HYP-009 LightGBM | `500` деревьев, `lr=.05`, `num_leaves=31`, `min_child_samples=20`, `reg_lambda=1`, native categories | Multiclass group outer | Macro F1 0.583855959 | Сохранённое сравнение SPEC-003; уступает SPEC-004 HYP-011. |
| HYP-011 XGBoost | `n_estimators=600`, `max_depth=6`, `learning_rate=.05`, `reg_lambda=1`, `tree_method=hist`, `device=cpu`, `multi:softprob`, seed 42 | Multiclass group outer | Macro F1 **0.591451371** | Лучший сохранённый SPEC-004 на этой схеме. |
| HYP-017 LightGBM + train-only category frequencies | `500` деревьев, `lr=.05`, `num_leaves=31`, `min_child_samples=20`, `reg_lambda=1` | Multiclass group outer | Macro F1 0.590968450 | Подтверждена в проверенных условиях; близко к HYP-011, ниже на 0.000483. |
| HYP-019 hierarchical LightGBM | Стадия `normal/attack`, затем тип атаки | Multiclass group outer | Macro F1 0.590111042 | Подтверждена в проверенных условиях; ниже HYP-011. |
| HYP-021 XGBoost + mandatory full-v1 | Тот же обязательный состав preprocessing policy, что выше | Multiclass group outer | Macro F1 **0.563787167** | Новее SPEC-004, но хуже на 0.027664. Составное изменение, outer использован. |

### Проверенные слабые направления на этих же split

| Гипотеза | Что проверяли | Результат | Вывод |
| --- | --- | ---: | --- |
| HYP-010 ExtraTrees | Без log target; те же основные параметры ExtraTrees | Regression RMSE 43.567568; multiclass Macro F1 0.580495 | На регрессии хуже HYP-012; на классификации ниже HYP-011. |
| HYP-013 LightGBM Tweedie | `objective=tweedie`, `variance_power=1.5` | Regression RMSE 47.188319 | Не подтверждена по критерию. |
| HYP-014 LightGBM linear leaves | `linear_tree=True` | Regression RMSE 178.084988 | Сильное ухудшение на проверенном split. |
| HYP-015 ExtraTrees + OOF target encoding | Train-only OOF encoding | Regression RMSE 44.402897 | Ниже HYP-012. |
| HYP-016 LightGBM categorical cross | Комбинация categorical features | Macro F1 0.580763596 | Ниже HYP-011. |
| HYP-018 LightGBM one-vs-rest | `multiclassova` | Macro F1 0.578639859 | Ниже HYP-011. |
| HYP-020 LightGBM inverse duplicate-group weights | Обратный размер группы одинаковых X | Macro F1 0.582934907 | Ниже HYP-011. |

## Историческая ветка Sergey: отдельный протокол

Эти результаты сохраняются как исторические, включая модели, которые были победителями на момент своих запусков. Для регрессии данные и temporal split совпадают с указанным выше протоколом; например, CatBoost H4 (`year_built=0 → missing`) дал RMSE **43.004638648** на `Year_Factor<5 → 5`. Оценка импортирована в позднюю историю, повторного fit для неё не делали.

Классификационные значения из Sergey использовали случайный stratified 80/20 split, seed 42 (RF `0.628538470`; CatBoost контроль `0.534099948`; flow-ratio RF `0.628620730`). Они **несопоставимы** с later group outer оценками `0.58–0.59`: дубликаты raw feature rows могли перейти между train и validation в random split. Даже внутри random split маленький прирост H5 (+0.000082 Macro F1) не перенёсся на sensitivity split seed 2026 (0.601301 против контроля 0.608205).

См. [таблицу Sergey](sources/sergey_branch--EXPERIMENTS.csv) и [registry](sources/sergey_branch--experiment_registry.json) для параметров и происхождения; [реестр Ivan](sources/ivan_branch--HYPOTHESES.md), [frozen plan](sources/ivan_branch--frozen-plan.json), [records](sources/ivan_branch--records.json) и [full-v1 report](sources/ivan_branch--FULL_PREPROCESSING_REPORT.md) содержат более новое evidence. Их SHA находятся в [manifest](sources/source-manifest.json).

## Правило для новых гипотез

Новые идеи должны улучшать выбранную метрику на search split и сохраняться отдельно. Search scores адаптивны: многие кандидаты сравниваются на одном search validation, поэтому они служат только выбору для следующего фиксированного audit gate. Baseline и search-selected кандидат фиксируются до audit; после него нельзя возвращаться к гипотезам на audit split. Все оценки новой архитектуры пока pending. Не объявлять статус `done` без фактической проверки; отличать `pending`, `не выполнена`, `не подтверждена по критерию` и `прошла в проверенных условиях`.
