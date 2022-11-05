# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from scipy.sparse import csr_matrix as sparsematrix

from .._field import Field
from ..region import Region


def topoints(values, region, sym=True, mode="tensor"):

    rows = region.mesh.cells.T.ravel()
    cols = np.zeros_like(rows)

    if mode == "tensor":
        dim = values.shape[0]
        if sym:
            if dim == 3:
                ij = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
            elif dim == 2:
                ij = [(0, 0), (1, 1), (0, 1)]
        else:
            if dim == 3:
                ij = [
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (1, 1),
                    (1, 2),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                ]
            elif dim == 2:
                ij = [(0, 0), (0, 1), (1, 0), (1, 1)]

        out = Field(region, dim=len(ij)).values

        for a, (i, j) in enumerate(ij):
            out[:, a] = (
                sparsematrix(
                    (values.reshape(dim, dim, -1)[i, j], (rows, cols)),
                    shape=(region.mesh.npoints, 1),
                ).toarray()[:, 0]
                / region.mesh.cells_per_point
            )

    elif mode == "scalar":
        out = sparsematrix(
            (values.ravel(), (rows, cols)), shape=(region.mesh.npoints, 1)
        ).toarray()[:, 0]
        out = out / region.mesh.cells_per_point

    return out


def project(values, region, average=True, mean=False):
    """Projection (and optionally averaging) of scalar or vectorial values
    at quadrature points to mesh-points.
    """

    # 1d-reshaped values
    dim = int(np.product(values.shape[:-2]))
    weights = region.quadrature.weights

    if mean:

        # evaluate how often the values must be repeated to match the number
        # of element-points
        reps = np.ones(len(values.shape), dtype=int)
        reps[-2] = len(region.element.points)

        # np.average(keepdims=True) requires numpy >= 1.23.0
        values = np.tile(
            np.average(values, axis=-2, weights=weights),
            reps=reps,
        )

        # workaround for np.average(keepdims=True)
        shape = values.shape
        shape = np.insert(shape, -1, 1)
        values = values.reshape(*shape)

    u = values.T.reshape(-1, dim)

    # disconnected mesh
    m = region.mesh.disconnect()

    if mean:
        # region on disconnected mesh with original quadrature scheme
        r = Region(m, region.element, region.quadrature, grad=False)
    else:
        # region on disconnected mesh with inverse quadrature scheme
        r = Region(m, region.element, region.quadrature.inv(), grad=False)

    # field for values on disconnected mesh; project values to mesh-points
    f = Field(r, dim=dim, values=u)
    v = f.interpolate()

    if mean:
        v = np.tile(
            np.average(v, axis=-2, weights=weights).reshape(dim, 1, -1), reps=reps
        )

    if average:

        # create dummy field for values on original mesh
        # (used for calculation of sparse-matrix indices)
        g = Field(region, dim=dim)

        # average values
        w = sparsematrix(
            (v.T.ravel(), g.indices.ai), shape=(dim * region.mesh.npoints, 1)
        ).toarray().reshape(-1, dim) / region.mesh.cells_per_point.reshape(-1, 1)

    else:

        w = v.reshape(-1, dim)

    return w
