# Alfa Hack: AutoResearch ML-агент

Новая серия [SPEC-004: десять гипотез](docs/specs/SPEC-004-ten-new-hypotheses.md) завершена: **ExtraTrees с log1p таргета — RMSE42.196700027194**, **XGBoost — grouped Macro F1 0.591451370919**. [Все11 оценок, выводы и ограничения](results/2026-10-23/TEN_HYPOTHESES_REPORT.md), [522 независимые проверки](results/2026-10-23/series-ten-hypotheses-v1/independent-verification.json). Новые методы реализованы в experiments/; платформенный solution.py/ZIP автоматически не подменялись экспериментальными победителями.

Рабочая ветка: `ivan_branch`. Агент, результаты предобработки, CatBoost/Random Forest, новые LightGBM/ExtraTrees и сохранённая история проверок.
Готовый [ZIP](dist/solution_ivan_branch.zip): 15 230 байт, только корневые Python-модули и неизменённый requirements.txt организаторов. Реальный API пока не проверен: env отсутствовали.

Перед КАЖДОЙ гипотезой прочитать [HYPOTHESES.md](HYPOTHESES.md). Проверенное не повторять; новое записать до запуска, исход сразу после. На платформе агент каждый раз читает постоянный hypotheses.json рядом с task.md и записывает started до fit, результат после; подробности в agent_runs/.

Выбраны CatBoost H4-zero-year (сохранённый RMSE43.004638647931) для регрессии и новый LightGBM (grouped Macro F1 0.583855959368) для классификации. Это лучшие наблюдавшиеся совместимые validation-оценки, без скрытого test и независимого подтверждения. [Новые результаты](results/2026-10-23/NEW_EXPERIMENT_REPORT.md), [SPEC-003](docs/specs/SPEC-003-platform-agent.md), [приёмка](docs/platform-preflight.md).

Запуск в Python3.12 с предоставленными зависимостями: `python solution.py /path/to/task.md`. Платформа задаёт OPENAI_BASE_URL, OPENAI_API_KEY и OPENAI_MODEL. Агент определяет данные/target/id/метрику, сразу сохраняет fallback; LLM предлагает инструмент эксперимента и ограниченную row-wise feature-функцию, worker обучает модель. Проверенный submission.csv сохраняется рядом с task.md; при отсутствии API работает ограниченный портфель моделей. Во время работы пакеты не устанавливаются.

- [Правила SDD](Codex.md).
- [Требования серии](docs/specs/SPEC-002-preprocessing-model-comparison.md) и [план](docs/plans/PLAN-002-preprocessing-model-comparison.md).
- [Исходные материалы и несоответствия](docs/sources/README.md).
- [Профиль данных](docs/sources/experiment-data-profile.json).
- [Журнал запусков](results/README.md).
- [Итоговый отчёт](results/2026-10-23/EXPERIMENT_REPORT.md), формируется после завершения обоих датасетов.

## Локальные инструменты и аудит

Используйте отдельное окружение Python 3.12 и установите `requirements-experiments.txt`.
В `data/regression/` и `data/multiclass/` должны лежать исходные `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv` и `task.md`.
Исходные файлы не изменяются; локальный runner использует эти явные пути независимо от несовпадающих путей внутри task.md.

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s tests -v
```

Скрипты run_suite, run_engineering_control и run_new_families сохранены для аудита; их уже проверенные конфигурации НЕ ПЕРЕЗАПУСКАТЬ. Сверять готовые метрики/предсказания без нового fit.
Данные, виртуальное окружение и большие бинарные артефакты остаются локальными и исключены из Git.

## Методика

Выбор preprocessing и силы штрафа выполняется только на inner. Полная outer-панель фиксируется до её первой оценки.
Для регрессии используются годы 1–3 → 4 и 1–4 → 5; для классификации одинаковые строки исходных признаков группируются, чтобы не попадать в train и validation одновременно.
Прежние случайные split из командного журнала напрямую с этими групповыми метриками не сравниваются.

Шаги: imputation/one-hot → корреляционный отбор → VIF → StandardScaler → Yeo–Johnson → предметные признаки.
Нормальность не обязательна для деревьев, а удаление зависимости может ухудшить качество: сохраняются и положительные, и отрицательные результаты.
L2 CatBoost проверяется штатным `l2_leaf_reg`; L1/L2 отдельно сравниваются на Lasso/Ridge и LogisticRegression, поскольку Random Forest таких штрафов не имеет.
