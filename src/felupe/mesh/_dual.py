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
    """Create a new dual mesh with given points per cell.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    points_per_cell : int or None, optional
        Number of points per cell, must be equal or lower than ``cells.shape[1]`` (
        default is None). If None, all points per cell are considered for the dual mesh.
    disconnect : bool, optional
        A flag to disconnect the mesh (each cell has its own points). Default is True.
    calc_points : bool, optional
        A flag to calculate the point coordinates for the dual mesh (default is False).
        If False, the points array is filled with zeros.
    offset : int, optional
        An offset to be added to the cells array (default is 0).
    npoints : int or None, optional
        Number of points for the dual mesh. If the given number of points is greater
        than ``npoints * points_per_cell``, then the missing points are added to the
        points array (filled with zeros). Default is None.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Notes
    -----
    ..  note::
        The points array of the dual mesh always has a shape of
        ``(npoints * points_per_cell, dim)``.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Rectangle(n=5).add_midpoints_edges()
    >>> region = fem.RegionQuadraticQuad(mesh=mesh)
    >>>
    >>> mesh_dual = fem.mesh.dual(mesh, points_per_cell=1, disconnect=False)
    >>> region_dual = fem.RegionConstantQuad(
    ...     mesh_dual, quadrature=region.quadrature, grad=False
    ... )
    >>>
    >>> displacement = fem.FieldPlaneStrain(region, dim=2)
    >>> pressure = fem.Field(region_dual)
    >>> field = fem.FieldContainer([displacement, pressure])

    See Also
    --------
    felupe.Mesh.dual : Create a new dual mesh with given points per cell.

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
    elif npoints is None and cells_new.max() > len(points_new):
        points_new = np.pad(
            points_new, ((1 + cells_new.max() - len(points_new), 0), (0, 0))
        )

    return points_new, cells_new, cell_type_new
