# SPDX-License-Identifier: GPL-3.0-or-later

"""
Evaluation of arrays in chunks of cells.

Arrays with the cells as last axis are evaluated in chunks of cells, e.g. the
contractions of :func:`numpy.einsum` in the assembly (``math._einsum.einsum_chunks()``)
and the gradients and hessians of the constitutive materials
(``constitution._chunks.evaluate_chunks()``). The intermediate arrays of a chunk fit
into the CPU caches and the chunks may be evaluated by the pool of threads of
``math._threads``.

The size of a chunk is given by one of two rules, depending on what is known about the
memory of the evaluation:

* The memory of all (intermediate) arrays per cell is known, e.g. for a contraction of
  :func:`numpy.einsum` with known shapes of the operands. Then, the arrays of a chunk
  have a size of about ``CHUNKSIZE_BYTES`` (2 MiB, about the L2 cache of one core),
  see the bilinear form of the gradients in ``assembly._cartesian``.
* The memory of the intermediate arrays is unknown, e.g. for a constitutive material
  with a user-defined function or automatic differentiation. Then, a chunk has about
  ``CHUNKSIZE_POINTS`` (8192) evaluation points, i.e. quadrature points times cells,
  which is the default value of the attribute ``ConstitutiveMaterial.chunksize``.
"""

from . import _threads

# Approximate number of bytes of all arrays which are processed per chunk of cells, if
# the memory per cell is known. The arrays of a chunk should fit into the CPU caches,
# about the size of the L2 cache of one core. Smaller chunks increase the overhead of
# the calls per chunk.
CHUNKSIZE_BYTES = 2**21

# Approximate number of evaluation points (quadrature points times cells) per chunk of
# cells, if the memory of the intermediate arrays is unknown.
CHUNKSIZE_POINTS = 8192


def chunk_slices(length, chunksize):
    """Return the slices of the chunks of an axis with ``length`` items, each with at
    most ``chunksize`` (at least one) items.

    Examples
    --------
    >>> from felupe.math._chunks import chunk_slices
    >>>
    >>> chunk_slices(5, chunksize=2)
    [slice(0, 2, None), slice(2, 4, None), slice(4, 5, None)]
    """

    chunksize = max(1, int(chunksize))

    return [
        slice(start, min(start + chunksize, length))
        for start in range(0, length, chunksize)
    ]


def map_chunks(fun, chunks, parallel=False):
    """Return the list of results of ``fun(chunk)`` for all chunks.

    The chunks are evaluated by the pool of threads if ``parallel`` is True, if there
    is more than one chunk and if more than one CPU is usable. Otherwise, or if called
    from a thread of the pool, the chunks are evaluated in serial.
    """

    if parallel and len(chunks) > 1 and _threads.cpu_count() > 1:
        return _threads.map_threaded(fun, chunks)

    return [fun(chunk) for chunk in chunks]
