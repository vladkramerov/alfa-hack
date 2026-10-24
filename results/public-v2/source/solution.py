"""Boosters entry point: python3 solution.py task.md."""
import os
import sys
import time

STARTED = time.monotonic()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 solution.py task.md')
    threads = min(8, max(1, int(os.getenv('AUTOML_THREADS', '4'))))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)
    from automl_engine import main
    main(sys.argv[1], started=STARTED, threads=threads)
