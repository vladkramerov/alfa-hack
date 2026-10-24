# Alfa Hack AutoResearch

Универсальный табличный AutoResearch для Boosters: ранний `submission.csv`, ограниченные worker-процессы, регрессионный выбор по последним известным train-периодам и causal history, проверенный классификационный маршрут, безопасный DSL и offline fallback. Независимая оценка выполняется внешним masked benchmark; исправленный nested audit доступен опционально.

- Текущее состояние и команды: [AGENTS.md](AGENTS.md).
- Текущая ограниченная серия и сравнение команды: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).
- Предыдущие public-v1…v5: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- Официальный контракт: [docs/PLATFORM_CONTRACT.md](docs/PLATFORM_CONTRACT.md).

Запуск на Python 3.12: `python3 solution.py task.md`. Разрешённые зависимости — неизменённый официальный `requirements.txt`.

Сборка: `python experiments/package_solution.py`.
Проверка чистой распаковки: `python experiments/verify_package.py`.
Тесты: `python -m pytest -p no:cacheprovider -q tests`.

LLM получает только `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` из окружения платформы. Локальный `.env.local` автоматически не загружается агентом, не коммитится и не включается в ZIP.
