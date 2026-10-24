"""Wall-clock and process limits shared by the research controller and tests."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class Budget:
    def __init__(self, seconds, started=None):
        self.started = time.monotonic() if started is None else started
        self.seconds = min(10200., max(1., float(seconds)))
        self.deadline = self.started + self.seconds

    def remaining(self):
        return max(0., self.deadline - time.monotonic())

    def reserve(self, observed_fit=0.):
        return min(self.seconds * .35, max(self.seconds * .18, observed_fit * 4 + 5))


def child_environment(threads, include_api=False):
    # Workers receive no development secrets or user configuration.
    keep = ('PATH', 'SYSTEMROOT', 'WINDIR', 'TMPDIR', 'TEMP', 'TMP',
            'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE',
            'DYLD_FALLBACK_LIBRARY_PATH')
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.setdefault('PATH', os.defpath)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = str(threads)
    env.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    if include_api:
        for key in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL'):
            if os.getenv(key):
                env[key] = os.environ[key]
    return env


def _stop(process):
    if process.poll() is not None:
        return
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_child(script, arguments, directory, timeout, threads=4, include_api=False,
              memory_mb=28000):
    """One bounded process tree. HTTP and model hangs share the same hard timeout."""
    import psutil
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logfile = directory / 'process.log'
    started = time.monotonic()
    status = 'completed'
    peak_mb = 0.
    if timeout <= 0:
        return {'status': 'timeout', 'seconds': 0., 'peak_rss_mb': 0.}
    with logfile.open('wb') as log:
        env = child_environment(threads, include_api)
        env['MPLCONFIGDIR'] = str(directory / 'matplotlib-cache')
        process = subprocess.Popen([sys.executable, '-u', str(script), *map(str, arguments)],
            stdout=log, stderr=subprocess.STDOUT, cwd=directory,
            env=env, start_new_session=os.name == 'posix')
        try:
            observed = psutil.Process(process.pid)
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    status = 'timeout'
                    break
                try:
                    processes = [observed]
                    try:
                        processes.extend(observed.children(recursive=True))
                    except (psutil.AccessDenied, PermissionError):
                        pass
                    rss = sum(p.memory_info().rss for p in processes if p.is_running()) / 1024**2
                    peak_mb = max(peak_mb, rss)
                    if rss > memory_mb:
                        status = 'memory_limit'
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                    pass
                if logfile.stat().st_size > 8 * 1024**2:
                    status = 'log_limit'
                    break
                time.sleep(min(.15, max(.01, timeout - (time.monotonic() - started))))
        finally:
            if process.poll() is None:
                _stop(process)
    code = process.returncode
    if status == 'completed' and code:
        status = 'failed'
    return {'status': status, 'returncode': code, 'seconds': time.monotonic() - started,
            'peak_rss_mb': peak_mb}
