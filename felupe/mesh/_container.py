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

from copy import deepcopy
import numpy as np

from ._mesh import Mesh
from ._tools import sweep


class MeshContainer:
    """A container which operates on a list of meshes with identical
    dimensions.

    Parameters
    ----------
    meshes : [felupe.Mesh, ...]
        A list with meshes.

    Attributes
    ----------
    dim : int
        The (identical) dimension of all underlying meshes.
    points : ndarray
        Point coordinates.
    meshes : [felupe.Mesh, ...]
        A list with meshes.
    """

    def __init__(self, meshes, merge=False, decimals=None):
        """A container which operates on a list of meshes with identical
        dimensions."""

        # obtain the dimension from the first mesh
        self.dim = meshes[0].dim

        # init points and list of meshes
        self.points = np.zeros((0, self.dim))
        self.meshes = []

        # append all meshes
        [self.append(mesh) for mesh in meshes]

        if merge:
            self.merge_duplicate_points(decimals=decimals)

    def append(self, mesh):
        "Append a Mesh to the list of meshes."

        # number of points
        points = np.vstack([self.points, mesh.points])
        self.meshes.append(Mesh(points, mesh.cells + len(self.points), mesh.cell_type))

        # ensure identical points-arrays
        for i, m in enumerate(self.meshes):
            self.meshes[i].points = self.points = points

    def pop(self, index):
        "Pop an item of the list of meshes."
        item = self.meshes.pop(index)
        return item

    def cells(self):
        "Return a list of tuples with cell-types and cell-connectivities."
        return [(mesh.cell_type, mesh.cells) for mesh in self.meshes]

    def merge_duplicate_points(self, decimals=None):
        "Merge duplicate points and update meshes."

        # sweep points
        for i, mesh in enumerate(self.meshes):
            self.meshes[i] = sweep(mesh, decimals=decimals)

        # ensure identical points-arrays
        points = self.meshes[0].points
        for i, m in enumerate(self.meshes):
            self.meshes[i].points = self.points = points

    def as_meshio(self, combined=True, **kwargs):
        "Export a (combined) mesh object as ``meshio.Mesh``."

        import meshio

        if not combined:
            cells = [
                meshio.CellBlock(cell_type, data) for cell_type, data in self.cells()
            ]

        else:
            cells = {}
            for mesh in self.meshes:
                if mesh.cell_type not in cells.keys():
                    cells[mesh.cell_type] = mesh.cells
                else:
                    cells[mesh.cell_type] = np.vstack(
                        [cells[mesh.cell_type], mesh.cells]
                    )

        return meshio.Mesh(self.points, cells, **kwargs)

    def copy(self):
        "Return a deepcopy of the mesh container."
        return deepcopy(self)

    def __iadd__(self, mesh):
        self.append(mesh)
        return self

    def __getitem__(self, index):
        return self.meshes[index]

    def __repr__(self):
        header = "<felupe mesh container object>"
        points = f"  Number of points: {len(self.points)}"
        cells_header = "  Number of cells:"
        cells = []

        for cells_type, cells_data in self.cells():
            cells.append(f"    {cells_type}: {len(cells_data)}")

        return "\n".join([header, points, cells_header, *cells])

    def __str__(self):
        return self.__repr__()
