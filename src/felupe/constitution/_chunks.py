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

import numpy as np

from ..math._chunks import CHUNKSIZE_POINTS, chunk_slices, map_chunks


def evaluate_chunks(
    fun, x, *args, chunksize=CHUNKSIZE_POINTS, parallel=False, out=None, **kwargs
):
    r"""Evaluate a function of a constitutive material, e.g. the gradient or the hessian
    of the strain energy density function, in chunks of cells and return the list of
    results.

    Parameters
    ----------
    fun : callable
        The function ``fun(x, *args, **kwargs)`` which returns a list of arrays.
    x : list of ndarray
        The list of input arrays (e.g. the extracted fields and the state variables)
        with the cells as last axis.
    *args : tuple, optional
        Optional arguments for the function.
    chunksize : int or None, optional
        The approximate number of evaluation points (quadrature points times cells)
        per chunk. If None, all cells are evaluated at once (default is 8192).
    parallel : bool, optional
        A flag to evaluate the chunks by a pool of threads (default is False).
    out : ndarray or None, optional
        The output array of the first result (default is None).
    **kwargs : dict, optional
        Optional keyword-arguments for the function.

    Returns
    -------
    list of ndarray
        The list of results with the cells as last axis.

    Notes
    -----
    The arrays of the input list are sliced along their last axis, if it has the length
    of the cells. Other inputs, e.g. ``None``, arrays which are broadcasted along the
    cells (length one) or scalars, are passed unchanged. Results which are broadcasted
    along the cells, e.g. a constant elasticity tensor of shape ``(3, 3, 3, 3, 1, 1)``,
    are taken from the first chunk. Each chunk has at least two cells.

    Constitutive materials are evaluated point-wise, i.e. the result of a cell must not
    depend on other cells. With ``parallel=True``, the function must be thread-safe.
    """

    def per_cell(a, length):
        "Check if an array has a last axis (of the cells) with the given length."
        return isinstance(a, np.ndarray) and a.ndim > 0 and a.shape[-1] == length

    arrays = [a for a in x if isinstance(a, np.ndarray) and a.ndim > 0]
    ncells = max([a.shape[-1] for a in arrays], default=1)

    # number of quadrature points per cell (the trailing axes are points and cells)
    npoints = max(
        [a.shape[-2] for a in arrays if per_cell(a, ncells) and a.ndim > 1], default=1
    )

    cells = ncells if chunksize is None else max(2, chunksize // npoints)

    if ncells <= cells:  # a single chunk
        res = list(fun(x, *args, **kwargs))

        if out is not None and isinstance(res[0], np.ndarray):
            np.copyto(out, res[0])
            res[0] = out

        return res

    def chunk_of(x, chunk):
        return [a[..., chunk] if per_cell(a, ncells) else a for a in x]

    chunks = chunk_slices(ncells, cells)

    # evaluate the first chunk (serial) or a probe of two cells (parallel) to create the
    # output arrays of the results per cell. A first chunk in serial would limit the
    # speedup of the threads, e.g. by about 12 % for eight chunks and two threads.
    first = slice(0, 2 if parallel else cells)
    length = first.stop
    res = list(fun(chunk_of(x, first), *args, **kwargs))

    sliced = []
    for i, r in enumerate(res):
        if per_cell(r, length):
            if i == 0 and out is not None:
                result = out
            else:
                result = np.empty((*r.shape[:-1], ncells), dtype=r.dtype)

            result[..., first] = r
            res[i] = result
            sliced.append(i)

    def evaluate(chunk):
        results = fun(chunk_of(x, chunk), *args, **kwargs)

        for i in sliced:
            res[i][..., chunk] = results[i]

    if parallel:
        map_chunks(evaluate, chunks, parallel=True)
    else:
        map_chunks(evaluate, chunks[1:])  # the first chunk is already evaluated

    return res
