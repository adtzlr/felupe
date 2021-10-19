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
