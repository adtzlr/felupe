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

import numpy as np


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
        The output array. If None, a new array is created (default is None).
    idx : str or None, optional
        The subscript (a single letter) of the output along which the operands are
        divided into chunks. If None, the last axis of the output is used, i.e. the
        cells by FElupe's convention (default is None). Axes of the ellipsis are not
        available by a letter.
    chunksize : int or None, optional
        The number of items along ``idx`` per chunk. If None, the items are divided
        into one chunk per thread or into a single chunk, if ``parallel`` is False
        (default is None).
    optimize : bool or str or list, optional
        The contraction path, passed to :func:`numpy.einsum` for each chunk (default
        is False).
    parallel : bool, optional
        Flag to evaluate the chunks by a pool of threads, one per usable CPU and at
        most one per chunk (default is False).

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

    operands = [np.asarray(x) for x in operands]
    inputs, output = _parse_subscripts(subscripts, operands)

    if idx is None:
        if len(output) == 0:
            raise ValueError("The output of a full contraction can't be divided.")
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

    if out is None:
        shape = tuple(lengths[letter] for letter in output)
        out = np.empty(shape, dtype=np.result_type(*operands))

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

    def evaluate(start):
        chunk = slice(start, start + chunksize)
        sliced = [
            x[(slice(None),) * a + (chunk,)] if a >= 0 else x
            for x, a in zip(operands, axes)
        ]
        res = out[(slice(None),) * axis + (chunk,)]
        np.einsum(subscripts, *sliced, optimize=optimize, out=res)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(evaluate, starts))
    else:
        for start in starts:
            evaluate(start)

    return out
