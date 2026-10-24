"""Песочница для экспериментов: дочерний процесс через fork с жёстким таймаутом.

fork наследует уже загруженные данные без копирования (copy-on-write), а зависший
или упавший эксперимент убивается, не роняя основной процесс агента.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
import tempfile
import time
import traceback
from typing import Any, Callable


class ExperimentTimeout(Exception):
    pass


class ExperimentError(Exception):
    def __init__(self, message: str, tb: str):
        super().__init__(message)
        self.tb = tb


def _child(fn: Callable, args: tuple, out_path: str) -> None:
    try:
        res = ("ok", fn(*args))
    except BaseException as e:  # noqa: BLE001 — передаём любую ошибку родителю
        res = ("err", (f"{type(e).__name__}: {e}", traceback.format_exc()))
    with open(out_path, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_with_timeout(fn: Callable, args: tuple, timeout: float, use_process: bool = True) -> Any:
    if not use_process or "fork" not in mp.get_all_start_methods():
        return fn(*args)
    ctx = mp.get_context("fork")
    fd, out_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    p = ctx.Process(target=_child, args=(fn, args, out_path), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
            p.join(2)
        _rm(out_path)
        raise ExperimentTimeout(f"превышен таймаут {timeout:.0f} с")
    try:
        with open(out_path, "rb") as f:
            status, payload = pickle.load(f)
    except Exception:  # noqa: BLE001
        raise ExperimentError(f"процесс завершился с кодом {p.exitcode} (вероятно, нехватка памяти)", "")
    finally:
        _rm(out_path)
    if status == "err":
        raise ExperimentError(payload[0], payload[1])
    return payload


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def timed(fn: Callable, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0
