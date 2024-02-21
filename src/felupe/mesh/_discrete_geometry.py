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

from copy import deepcopy

import numpy as np


class DiscreteGeometry:
    """A discrete geometry with points, cells and optional a specified cell type.

    Parameters
    ----------
    points : ndarray
        Point coordinates.
    cells : ndarray
        Point-connectivity of cells.
    cell_type : str or None, optional
        An optional string in VTK-convention that specifies the cell type (default is
        None). Necessary when a discrete geometry is saved to a file.

    Attributes
    ----------
    points : ndarray
        Point coordinates.
    cells : ndarray
        Point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.
    npoints : int
        Amount of points.
    dim : int
        Dimension of mesh point coordinates.
    ndof : int
        Amount of degrees of freedom.
    ncells : int
        Amount of cells.
    points_with_cells : array
        Array with points connected to cells.
    points_without_cells : array
        Array with points not connected to cells.
    cells_per_point : array
        Array which counts connected cells per point. Used for averaging results.

    See Also
    --------
    felupe.Mesh : A mesh with points, cells and optional a specified cell type.

    """

    def __init__(self, points, cells, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells)
        self.cell_type = cell_type

        self.update()

    def copy(self, points=None, cells=None, cell_type=None):
        """Return a deepcopy."""

        out = deepcopy(self)

        if points is not None or cells is not None or cell_type is not None:
            out.update(points=points, cells=cells, cell_type=cell_type)

        return out

    def update(self, points=None, cells=None, cell_type=None, callback=None):
        """Update the mesh with given points and cells arrays inplace. Optionally, a
        callback is evaluated.

        Parameters
        ----------
        points : ndarray or None
            New point coordinates (default is None). If None, it is unchanged.
        cells : ndarray
            New point-connectivity of cells (default is None). If None, it is unchanged.
        cell_type : str or None, optional
            New string in VTK-convention that specifies the cell type (default is
            None). If None, it is unchanged.
        callback : callable or None, optional
            A callable which is called after the mesh is updated (default is None).

        Examples
        --------
        ..  warning::
            If the points of a mesh are modified and a region was already created with
            the mesh, it is important to re-evaluate (reload) the
            :class:`~felupe.Region`.

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> new_points = mesh.rotate(angle_deg=-90, axis=2).points
        >>> mesh.update(points=new_points, callback=region.reload)

        See Also
        --------
        felupe.Region.reload : Reload the numeric region.
        """

        if points is not None:
            self.points = points

        if cells is not None:
            self.cells = cells

        if cell_type is not None:
            self.cell_type = cell_type

        # obtain dimensions
        self.npoints, self.dim = self.points.shape
        self.ndof = self.points.size
        self.ncells = self.cells.shape[0]

        # get number of cells per point
        points_in_cell, self.cells_per_point = np.unique(self.cells, return_counts=True)

        # check if there are points without cells
        if self.npoints != len(self.cells_per_point):
            self.point_has_cell = np.isin(np.arange(self.npoints), points_in_cell)
            # update "cells_per_point" ... cells per point
            cells_per_point = -np.ones(self.npoints, dtype=int)
            cells_per_point[points_in_cell] = self.cells_per_point
            self.cells_per_point = cells_per_point

            self.points_without_cells = np.arange(self.npoints)[~self.point_has_cell]
            self.points_with_cells = np.arange(self.npoints)[self.point_has_cell]
        else:
            self.points_without_cells = np.array([], dtype=int)
            self.points_with_cells = np.arange(self.npoints)

        if callable(callback):
            callback(self)

    @property
    def x(self):
        "Return the first column (x-component) of the points array."
        return self.points[:, 0]

    @property
    def y(self):
        "Return the second column (y-component) of the points array."
        return self.points[:, 1]

    @property
    def z(self):
        "Return the third column (z-component) of the points array."
        return self.points[:, 2]
