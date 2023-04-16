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

from ._helpers import mesh_or_data


@mesh_or_data
def dual(
    points,
    cells,
    cell_type,
    points_per_cell=None,
    disconnect=True,
    calc_points=False,
    offset=0,
    npoints=None,
):
    """Create a new dual mesh with given points per cell. The point coordinates are not
    used in a dual mesh and hence, by default they are all zero.
    """

    ncells = len(cells)
    dim = points.shape[1]
    cell_type_new = None

    if points_per_cell is None:
        points_per_cell = cells.shape[1]
        cell_type_new = cell_type

    if disconnect:
        cells_new = np.arange(ncells * points_per_cell).reshape(ncells, points_per_cell)
    else:
        cells_new = cells[:, :points_per_cell]

    if calc_points:
        points_new = points[cells[:, :points_per_cell]].reshape(-1, dim)
    else:
        points_new = np.broadcast_to(
            np.zeros((1, dim), dtype=int), (ncells * points_per_cell, dim)
        )

    if offset > 0:
        cells_new += offset
        points_new = np.pad(points_new, ((offset, 0), (0, 0)))

    if npoints is not None and npoints > len(points_new):
        points_new = np.pad(points_new, ((npoints - len(points_new), 0), (0, 0)))

    return points_new, cells_new, cell_type_new
