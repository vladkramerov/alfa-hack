# Alfa Hack: AutoResearch ML agent

Универсальный агент для табличной регрессии и классификации. Читает `task.md`, строит валидацию, исследует модели и признаки, сохраняет `submission.csv` рядом с описанием задачи.

```bash
python3 solution.py /path/to/task.md
python3 experiments/package_solution.py
```

Архив для платформы: `dist/alfa_autoresearch_submission.zip`. Он содержит только корневые Python-модули и оригинальный `requirements.txt`; данные и обученные модели не включены.

Контекст и продолжение работы: [AGENTS.md](AGENTS.md), [состояние исследования](docs/RESEARCH_STATE.md), [реестр гипотез](docs/HYPOTHESES.md), [контракт платформы](docs/PLATFORM_CONTRACT.md). Локальные метрики не являются score Boosters. Отправка архива на платформу выполняется пользователем.

Для разработки нужен Python 3.12 и разрешённые библиотеки из `requirements.txt`. Проверки используют также `pytest`; платформа в нём не нуждается. На macOS для LightGBM/XGBoost нужен локальный OpenMP runtime. Параметры тестовых запусков: `AUTOML_TIME_LIMIT` (не более 10200 секунд), `AUTOML_MAX_EXPERIMENTS`, `AUTOML_THREADS` (не более 8). Без переменных `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` действует исследовательский fallback; с ними LLM получает измеренные результаты и управляет следующими экспериментами.
