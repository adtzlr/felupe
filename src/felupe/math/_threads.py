# SPDX-License-Identifier: GPL-3.0-or-later

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, local

_pool = None
_pool_lock = Lock()

# a flag which marks the threads of the pool during the evaluation of a task
_worker = local()


def cpu_count():
    """Return the number of CPUs which may be used by the current process. Unlike
    :func:`os.cpu_count`, this respects the CPU affinity of the process, e.g. the CPUs
    of a job on a shared HPC-node. For Python 3.13+, the number of CPUs may be
    overridden by the environment variable ``PYTHON_CPU_COUNT``."""

    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1

    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0))

    return os.cpu_count() or 1


def thread_pool():
    """Return the pool of threads, one per usable CPU. The pool is created on its
    first use and re-used afterwards, because the start of new threads takes longer
    than the evaluation of small arrays."""

    global _pool

    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=cpu_count())

    return _pool


def _reset_thread_pool():
    "A forked child process has no threads, i.e. the pool of the parent is unusable."

    global _pool, _pool_lock

    _pool = None
    _pool_lock = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_thread_pool)


def map_threaded(fun, items):
    """Return the list of results of ``fun(item)`` for all items, evaluated by the pool
    of threads.

    A task of the pool which waits for further tasks of the same pool may block all
    threads of the pool. Hence, calls from a thread of the pool, e.g. a math-function
    with ``parallel=True`` within a chunk of a constitutive material, are evaluated
    in serial.
    """

    if getattr(_worker, "active", False):
        return [fun(item) for item in items]

    def task(item):
        _worker.active = True
        try:
            return fun(item)
        finally:
            _worker.active = False

    return list(thread_pool().map(task, items))
