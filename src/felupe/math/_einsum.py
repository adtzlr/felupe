# SPDX-License-Identifier: GPL-3.0-or-later

from string import ascii_letters

import numpy as np

from . import _threads
from ._chunks import chunk_slices, map_chunks


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

    if chunksize is None:
        workers = _threads.cpu_count() if parallel else 1
        chunksize = -(-length // workers)  # one chunk per thread

    chunks = chunk_slices(length, chunksize)

    if out is None:
        if len(chunks) <= 1:
            return einsum(*operands)

        shape = tuple(lengths[letter] for letter in output)
        dtype = kwargs.get("dtype")

        if dtype is None:
            dtype = np.result_type(*operands)

        out = np.empty(shape, dtype=dtype)

    def evaluate(chunk):
        sliced = [
            x[(slice(None),) * a + (chunk,)] if a >= 0 else x
            for x, a in zip(operands, axes)
        ]
        einsum(*sliced, out=out[(slice(None),) * axis + (chunk,)])

    map_chunks(evaluate, chunks, parallel=parallel)

    return out
