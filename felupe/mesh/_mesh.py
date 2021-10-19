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


class Mesh:
    def __init__(self, points, cells, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells)
        self.cell_type = cell_type

        self.update(self.cells)

    def update(self, cells):
        self.cells = cells

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

    def save(self, filename="mesh.vtk"):
        "Export mesh as VTK file."

        if self.cell_type is None:
            raise TypeError("Cell type missing.")
        else:
            import meshio

        cells = {self.cell_type: self.cells}
        meshio.Mesh(self.points, cells).write(filename)
