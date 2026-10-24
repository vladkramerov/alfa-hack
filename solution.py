"""Точка входа на платформе: python solution.py [--data DIR] [--output PATH].

Интерфейс (переменные окружения, пути) описан в ag_io_contract.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ag_budget  # noqa: E402,F401  — фиксирует момент старта процесса для отсчёта 3 часов
from ag_orchestrator import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
