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

    """

    def __init__(self, points, cells, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells)
        self.cell_type = cell_type

        self.update(self.cells)

    def update(self, cells, cell_type=None):
        "Update the cell and dimension attributes with a given cell array."
        self.cells = cells

        if cell_type is not None:
            self.cell_type = cell_type

        # obtain dimensions
        self.npoints, self.dim = self.points.shape
        self.ndof = self.points.size
        self.ncells = self.cells.shape[0]

        # get number of cells per point
        points_in_cell, self.cells_per_point = np.unique(cells, return_counts=True)

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
