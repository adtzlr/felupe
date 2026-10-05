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

from weakref import WeakKeyDictionary

import numpy as np
from scipy.sparse import csr_matrix


class SparsityPattern:
    r"""The sparsity pattern of a bilinear form of a test and a trial field in CSR
    format, together with the position of each value of the integrated form in the
    data array of the sparse matrix.

    Parameters
    ----------
    v : Field
        The test field.
    u : Field
        The trial field.

    Notes
    -----
    The degrees of freedom of both fields must be numbered point-wise, i.e.
    ``dof = point * dim + i``. The unique pairs of points per cell are determined
    instead of the unique pairs of degrees of freedom. The pattern of the degrees of
    freedom follows by arithmetic: each pair of points is a ``(dim_v, dim_u)``-block
    and the columns of a row are sorted.

    The integrated values of shape ``(a, i, b, k, c)`` are summed up into the data
    array of the sparse matrix by :func:`numpy.bincount`. This replaces the conversion
    of all (duplicate) values from the coordinate (COO) to the CSR format on each
    assembly.
    """

    def __init__(self, v, u):
        cv, cu = v.indices.cai, u.indices.cai  # (c, a, i) and (c, b, k)
        ncells, na, dv = cv.shape
        nb, du = cu.shape[1:]

        self.ncells = ncells
        self.shape = (v.indices.shape[0], u.indices.shape[0])

        nrows, ncols = self.shape[0] // dv, self.shape[1] // du  # number of points
        pv = cv[:, :, 0].T // dv  # (a, c) points of the test field
        pu = cu[:, :, 0].T // du  # (b, c) points of the trial field

        # unique pairs of points (row-major, hence the columns of a row are sorted)
        keys = pv[:, None, :].astype(np.int64) * ncols + pu[None, :, :]  # (a, b, c)
        pairs, pair = np.unique(keys.ravel(), return_inverse=True)
        pair = pair.reshape(na, nb, ncells)
        row, col = np.divmod(pairs, ncols)

        # position of each pair of points within its row of points
        npairs = np.bincount(row, minlength=nrows)
        first = np.concatenate([[0], np.cumsum(npairs)[:-1]])
        rank = np.arange(len(pairs)) - first[row]

        # the row of the degree of freedom (point, i) starts at start + i * length
        length = npairs * du
        start = np.concatenate([[0], np.cumsum(length * dv)])
        self.nnz = int(start[-1])

        i, k = np.arange(dv), np.arange(du)
        self.indptr = np.append(
            (start[:-1, None] + i * length[:, None]).ravel(), self.nnz
        )

        # positions and column indices of all pairs of points, shape (pairs, i, k)
        position = (
            (start[row] + rank * du)[:, None, None]
            + i[:, None] * length[row][:, None, None]
            + k
        )
        self.indices = np.empty(self.nnz, dtype=np.int64)
        self.indices[position.ravel()] = np.broadcast_to(
            (col * du)[:, None, None] + k, position.shape
        ).ravel()

        # position of each value of the integrated form, shape (a, i, b, k, c)
        offset = (start[row] + rank * du)[pair]
        length_pair = length[row][pair]
        self.position = (
            offset[:, None, :, None, :]
            + i[None, :, None, None, None] * length_pair[:, None, :, None, :]
            + k[None, None, None, :, None]
        ).ravel()

        if self.nnz < np.iinfo(np.int32).max:
            self.indices = self.indices.astype(np.int32)
            self.indptr = self.indptr.astype(np.int32)

    def assemble(self, values):
        "Assemble the integrated values of shape (a, i, b, k, c) to a sparse matrix."

        # broadcast values of a uniform grid mesh
        if values.shape[-1] != self.ncells:
            values = np.broadcast_to(values, (*values.shape[:-1], self.ncells))

        data = np.bincount(self.position, weights=values.ravel(), minlength=self.nnz)

        # copies of the indices, the sparse matrix may be modified in-place
        matrix = csr_matrix(
            (data, self.indices.copy(), self.indptr.copy()), shape=self.shape
        )
        matrix.has_canonical_format = True

        return matrix


def is_pointwise(indices):
    "Return True if the degrees of freedom are numbered point-wise."
    cai = indices.cai
    dim = cai.shape[-1]
    first = cai[..., :1]

    return bool(np.all(first % dim == 0)) and np.array_equal(
        cai, first + np.arange(dim)
    )


# sparsity patterns, weakly referenced by the indices of the test and the trial field.
# The patterns are not copied with the fields and they are released together with the
# indices of the fields.
_patterns = WeakKeyDictionary()


def sparsity_pattern(v, u):
    """Return the (cached) sparsity pattern of a bilinear form of the test field ``v``
    and the trial field ``u`` or None if the degrees of freedom of the fields are not
    numbered point-wise."""

    patterns = _patterns.setdefault(v.indices, WeakKeyDictionary())

    if u.indices not in patterns:
        if is_pointwise(v.indices) and is_pointwise(u.indices):
            patterns[u.indices] = SparsityPattern(v, u)
        else:
            patterns[u.indices] = None

    return patterns[u.indices]
