# Журнал экспериментов команды

[Итоги гипотез](HYPOTHESES.md) — что помогло и что не помогло.
[Все эксперименты](EXPERIMENTS.md) — модели, изменения, настройки, метрики и выводы.
[Excel с фильтрами](EXPERIMENTS.xlsx) · [CSV](EXPERIMENTS.csv) · [Полные данные JSON](experiment_registry.json)

Учтено 33 оценок экспериментов и 16 внутренних попыток Optuna.

| Задача | Лучший основной результат | Запуск |
|---|---:|---|
| regression | rmse=43.004639 | H4-zero-year |
| multiclass | macro_f1=0.628621 | H5-flow-ratios |

RMSE меньше — лучше. Macro F1 больше — лучше. Метрики скрытого test неизвестны.
Основная валидация, внутренний подбор, дополнительная проверка и технические повторы разделены.
Для обновления CSV/Markdown/JSON: python build_experiment_registry.py. Excel — снимок на дату сборки. Для публикации на GitHub нужны commit и push в sergey_branch.
