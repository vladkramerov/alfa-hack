# Alfa Hack AutoResearch

Универсальный табличный AutoResearch для Boosters: ранний `submission.csv`, ограниченные worker-процессы, выбор по внутренней валидации, отдельный outer audit, безопасный DSL признаков и offline fallback при недоступном LLM.

- Текущее состояние и команды: [AGENTS.md](AGENTS.md).
- Текущая ограниченная серия и сравнение команды: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).
- Предыдущие public-v1…v5: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- Официальный контракт: [docs/PLATFORM_CONTRACT.md](docs/PLATFORM_CONTRACT.md).

Запуск на Python 3.12: `python3 solution.py task.md`. Разрешённые зависимости — неизменённый официальный `requirements.txt`.

Сборка: `python experiments/package_solution.py`.
Проверка чистой распаковки: `python experiments/verify_package.py`.
Тесты: `python -m pytest -p no:cacheprovider -q tests`.

LLM получает только `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` из окружения платформы. Локальный `.env.local` автоматически не загружается агентом, не коммитится и не включается в ZIP.
