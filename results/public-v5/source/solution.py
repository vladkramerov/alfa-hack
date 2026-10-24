"""Boosters entry point: python3 solution.py task.md."""
import os
import sys
import time
import traceback

STARTED = time.monotonic()


def _threads():
    try:
        return min(8, max(1, int(os.getenv('AUTOML_THREADS', '8'))))
    except ValueError:
        return 8


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 solution.py task.md')
    threads = _threads()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)
    try:
        from automl_engine import main
        main(sys.argv[1], started=STARTED, threads=threads)
    except Exception:
        # Never fail the whole platform run after a valid early fallback exists.
        traceback.print_exc()
        try:
            from automl_data import emergency_submission
            path = emergency_submission(sys.argv[1])
            print('emergency submission:', path)
        except Exception:
            traceback.print_exc()
            raise
