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
from scipy.sparse import csr_matrix as sparsematrix

from ..field import Field
from ..region import Region


def topoints(values, region, sym=True, mode="tensor"):
    "Shift of scalar or tensorial values at quadrature points to mesh-points."

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

    dim = values.shape[:-2]
    size = int(np.prod(dim))
    weights = region.quadrature.weights

    # transpose values
    idx = np.arange(len(values.shape))
    idx[: len(dim)] = idx[: len(dim)][::-1]
    values = values.transpose(idx)

    if mean:
        # evaluate how often the values must be repeated to match the number
        # of element-points
        reps = np.ones(len(values.shape), dtype=int)
        reps[-2] = len(region.element.points)

        # np.average(keepdims=True) requires numpy >= 1.23.0
        values = np.average(values, axis=-2, weights=weights)

        # workaround for np.average(keepdims=True)
        shape = values.shape
        shape = np.insert(shape, -1, 1)
        values = values.reshape(*shape)

        # broadcast averaged values to match the number of element-points
        values = np.broadcast_to(values, shape=np.broadcast_shapes(shape, reps))

    # reshape from (shape, nint.points, nelements) to 1d-values
    u = values.T.reshape(-1, size)

    # disconnect the mesh
    m = region.mesh.disconnect()

    if mean:
        # region on disconnected mesh with original quadrature scheme
        # a single-point quadrature would be sufficient
        # but would require additional (not available) informations
        r = Region(m, region.element, region.quadrature, grad=False)
    else:
        # region on disconnected mesh with inverse quadrature scheme
        r = Region(m, region.element, region.quadrature.inv(), grad=False)

    # field for values on disconnected mesh; project values to mesh-points
    f = Field(r, dim=size, values=u)
    v = f.interpolate()

    if mean:
        # due to the usage of the original quadrature scheme the averaging must be
        # applied again
        # np.average(keepdims=True) requires numpy >= 1.23.0
        v = np.average(v, axis=-2, weights=weights).reshape(size, 1, -1)

        # broadcast averaged values to match the number of element-points
        shape = np.array([*v.shape[:-2], len(region.element.points), v.shape[-1]])
        v = np.broadcast_to(v, shape=shape)

    if average:
        # create dummy field for values on original mesh
        # (used for calculation of sparse-matrix indices)
        g = Field(region, dim=size)

        # average values
        w = sparsematrix(
            (v.T.ravel(), g.indices.ai), shape=(size * region.mesh.npoints, 1)
        ).toarray().reshape(-1, size) / region.mesh.cells_per_point.reshape(-1, 1)

    else:
        w = v.T.reshape(-1, size)

    return w
