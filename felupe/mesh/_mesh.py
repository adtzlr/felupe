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
from copy import deepcopy


class Mesh:
    """A mesh with points, cells and optional a specified cell type.

    Parameters
    ----------
    points : ndarray
        Point coordinates.
    cells : ndarray
        Point-connectivity of cells.
    cell_type : str or None, optional
        An optional string in VTK-convention that specifies the cell type (default is None). Necessary when a mesh is saved to a file.

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
        Array which counts connected cells per point. Used for averging results.

    """

    def __init__(self, points, cells, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells)
        self.cell_type = cell_type

        self.update(self.cells)

    def update(self, cells):
        "Update the cell and dimension attributes with a given cell array."
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

    def disconnect(self):
        "Return a new instance of a Mesh with disconnected cells."

        points = self.points[self.cells].reshape(-1, self.dim)
        cells = np.arange(self.cells.size).reshape(*self.cells.shape)

        return Mesh(points, cells, cell_type=self.cell_type)

    def as_meshio(self, **kwargs):
        "Export the mesh as ``meshio.Mesh``."

        import meshio

        cells = {self.cell_type: self.cells}
        return meshio.Mesh(self.points, cells, **kwargs)

    def save(self, filename="mesh.vtk", **kwargs):
        """Export the mesh as VTK file. For XDMF-export please ensure to have
        ``h5py`` (as an optional dependancy of ``meshio``) installed.

        Parameters
        ----------
        filename : str, optional
            The filename of the mesh (default is ``mesh.vtk``).

        """

        self.as_meshio(**kwargs).write(filename)

    def copy(self):
        """Return a deepcopy of the mesh.

        Returns
        -------
        Mesh
            A deepcopy of the mesh.

        """

        return deepcopy(self)

    def __repr__(self):
        header = "<felupe mesh object>"
        points = f"  Number of points: {len(self.points)}"
        cells_header = "  Number of cells:"
        cells = [f"    {self.cell_type}: {self.ncells}"]

        return "\n".join([header, points, cells_header, *cells])

    def __str__(self):
        return self.__repr__()
