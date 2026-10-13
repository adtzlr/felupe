# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from string import ascii_letters
from threading import Lock

import numpy as np

_pool = None
_pool_lock = Lock()


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
    than the contractions of small arrays."""

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


def _parse_subscripts(subscripts, operands):
    """Return the subscripts of the operands and of the output of
    :func:`numpy.einsum`, with the ellipsis replaced by unused letters. Like in NumPy,
    the axes of the ellipsis are broadcasted right-aligned, i.e. an operand with less
    axes in its ellipsis gets the last letters of the ellipsis."""

    subscripts = subscripts.replace(" ", "")

    if "->" not in subscripts:
        raise ValueError("The subscripts of the output must be given explicitly.")

    inputs, output = subscripts.split("->")
    inputs = inputs.split(",")

    if len(inputs) != len(operands):
        raise ValueError(
            f"The number of subscripts ({len(inputs)}) must be equal to the "
            f"number of operands ({len(operands)})."
        )

    # number of axes of the ellipsis of each operand
    ndims = [
        x.ndim - len(s.replace("...", "")) if "..." in s else 0
        for s, x in zip(inputs, operands)
    ]
    unused = [letter for letter in ascii_letters if letter not in subscripts]
    ellipsis = "".join(unused[: max(ndims, default=0)])

    inputs = [
        s.replace("...", ellipsis[len(ellipsis) - n :]) for s, n in zip(inputs, ndims)
    ]
    output = output.replace("...", ellipsis)

    return inputs, output


def einsum_chunks(
    subscripts,
    *operands,
    out=None,
    idx=None,
    chunksize=None,
    optimize=False,
    parallel=False,
    **kwargs,
):
    """Evaluate :func:`numpy.einsum` in chunks along a free index of the output and
    write the results into the output array.

    Parameters
    ----------
    subscripts : str
        The subscripts for the summation with an explicit output, e.g.
        ``"ij...,jk...->ik..."``.
    *operands : list of ndarray
        The arrays for the operation.
    out : ndarray or None, optional
        The output array. If None, a new array in C-order is created. A single chunk
        is evaluated by :func:`numpy.einsum` as a whole, i.e. its result has the memory
        layout chosen by NumPy (default is None).
    idx : str or None, optional
        The subscript (a single letter) of the output along which the operands are
        divided into chunks. If None, the last axis of the output is used, i.e. the
        cells by FElupe's convention (default is None). Axes of the ellipsis are not
        available by a letter. If None and the output has no axes or its last index
        repeats within an operand, the operands are evaluated as a single chunk.
    chunksize : int or None, optional
        The number of items along ``idx`` per chunk. If None, the items are divided
        into one chunk per thread. If None and ``parallel`` is False, the operands are
        passed to :func:`numpy.einsum` without further checks (default is None).
    optimize : bool or str or list, optional
        The contraction path, passed to :func:`numpy.einsum` for each chunk (default
        is False).
    parallel : bool, optional
        Flag to evaluate the chunks by a (re-used) pool of threads, one per usable CPU
        (default is False).
    **kwargs : dict, optional
        Optional keyword-arguments for :func:`numpy.einsum`, passed for each chunk,
        e.g. ``dtype``. The output array of more chunks is created with ``dtype``.

    Returns
    -------
    ndarray
        The output array.

    Notes
    -----
    Only the operands with the index ``idx`` in their subscripts are sliced. Operands
    with a broadcasted axis (of length one) along ``idx`` are not sliced. The index
    must not repeat within the subscripts of an operand. Chunks of cells, which fit
    into the CPU caches, may speed up contractions with large intermediate arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from felupe.math._einsum import einsum_chunks
    >>>
    >>> rotation = np.eye(3)  # no axis of the cells
    >>> F = np.ones((3, 3, 4, 5))  # 4 quadrature points and 5 cells
    >>>
    >>> res = einsum_chunks("ik,kj...->ij...", rotation, F, chunksize=2)
    >>> res.shape
    (3, 3, 4, 5)
    """

    if not parallel and chunksize is None:  # a single chunk
        return np.einsum(subscripts, *operands, out=out, optimize=optimize, **kwargs)

    def einsum(*operands, out=None):
        return np.einsum(subscripts, *operands, out=out, optimize=optimize, **kwargs)

    operands = [np.asarray(x) for x in operands]
    inputs, output = _parse_subscripts(subscripts, operands)

    if idx is None:
        # the last index of the output, if it may be divided into chunks
        if len(output) == 0 or any(s.count(output[-1]) > 1 for s in inputs):
            return einsum(*operands, out=out)

        idx = output[-1]

    if len(idx) != 1 or idx not in output:
        raise ValueError(f"The index '{idx}' must be a free index of the output.")

    if any(s.count(idx) > 1 for s in inputs):
        raise ValueError(f"The index '{idx}' must not repeat within an operand.")

    # the lengths of the axes, broadcasted along axes of length one
    lengths = {}
    for s, x in zip(inputs, operands):
        for letter, length in zip(s, x.shape):
            if length != 1 or letter not in lengths:
                lengths[letter] = length

    missing = [letter for letter in output if letter not in lengths]
    if missing:
        raise ValueError(f"The output indices {missing} are not given in the inputs.")

    # axes of the chunked index in the operands (-1 if not sliced) and in the output
    axes = []
    for s, x in zip(inputs, operands):
        a = s.find(idx)
        axes.append(a if a >= 0 and x.shape[a] > 1 else -1)

    axis = output.find(idx)

    length = lengths[idx]
    workers = cpu_count() if parallel else 1

    if chunksize is None:
        chunksize = -(-length // workers)  # one chunk per thread

    chunksize = max(1, chunksize)
    starts = range(0, length, chunksize)
    workers = min(workers, len(starts))

    if out is None:
        if len(starts) <= 1:
            return einsum(*operands)

        shape = tuple(lengths[letter] for letter in output)
        dtype = kwargs.get("dtype")

        if dtype is None:
            dtype = np.result_type(*operands)

        out = np.empty(shape, dtype=dtype)

    def evaluate(start):
        chunk = slice(start, start + chunksize)
        sliced = [
            x[(slice(None),) * a + (chunk,)] if a >= 0 else x
            for x, a in zip(operands, axes)
        ]
        einsum(*sliced, out=out[(slice(None),) * axis + (chunk,)])

    if workers > 1:
        list(thread_pool().map(evaluate, starts))
    else:
        for start in starts:
            evaluate(start)

    return out
