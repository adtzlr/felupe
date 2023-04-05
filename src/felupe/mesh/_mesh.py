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

from functools import wraps

import numpy as np

from ._convert import (
    add_midpoints_edges,
    add_midpoints_faces,
    add_midpoints_volumes,
    collect_edges,
    collect_faces,
    collect_volumes,
    convert,
)
from ._discrete_geometry import DiscreteGeometry
from ._tools import expand, mirror, revolve, rotate, runouts, sweep, triangulate


def as_mesh(obj):
    "Convert a ``DiscreteGeometry`` object to a ``Mesh`` object."
    return Mesh(points=obj.points, cells=obj.cells, cell_type=obj.cell_type)


class Mesh(DiscreteGeometry):
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

    """

    def __init__(self, points, cells, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells)
        self.cell_type = cell_type

        super().__init__(points=points, cells=cells, cell_type=cell_type)

    def __repr__(self):
        header = "<felupe mesh object>"
        points = f"  Number of points: {len(self.points)}"
        cells_header = "  Number of cells:"
        cells = [f"    {self.cell_type}: {self.ncells}"]

        return "\n".join([header, points, cells_header, *cells])

    def __str__(self):
        return self.__repr__()

    @wraps(expand)
    def expand(self, n=11, z=1):
        return as_mesh(expand(self, n=n, z=z))

    @wraps(rotate)
    def rotate(self, angle_deg, axis, center=None):
        return as_mesh(rotate(angle_deg=angle_deg, axis=axis, center=center))

    @wraps(revolve)
    def revolve(self, n=11, phi=180, axis=0):
        return as_mesh(revolve(self, n=n, phi=phi, axis=axis))

    @wraps(sweep)
    def sweep(self, decimals=None):
        return as_mesh(sweep(self, decimals=decimals))

    @wraps(mirror)
    def mirror(self, normal=[1, 0, 0], centerpoint=[0, 0, 0], axis=None):
        return as_mesh(mirror(self, normal=normal, centerpoint=centerpoint, axis=axis))

    @wraps(triangulate)
    def triangulate(self, mode=3):
        return as_mesh(triangulate(self, mode=mode))

    @wraps(runouts)
    def add_runouts(
        self,
        values=[0.1, 0.1],
        centerpoint=[0, 0, 0],
        axis=0,
        exponent=5,
        mask=slice(None),
    ):
        return as_mesh(
            runouts(
                self,
                values=values,
                centerpoint=centerpoint,
                axis=axis,
                exponent=exponent,
                mask=mask,
            )
        )

    @wraps(convert)
    def convert(
        self,
        order=0,
        calc_points=False,
        calc_midfaces=False,
        calc_midvolumes=False,
    ):
        return as_mesh(
            convert(
                self,
                order=order,
                calc_points=calc_points,
                calc_midfaces=calc_midfaces,
                calc_midvolumes=calc_midvolumes,
            )
        )

    @wraps(collect_edges)
    def collect_edges(self):
        return collect_edges

    @wraps(collect_faces)
    def collect_faces(self):
        return collect_faces

    @wraps(collect_volumes)
    def collect_volumes(self):
        return collect_volumes

    @wraps(add_midpoints_edges)
    def add_midpoints_edges(self):
        return add_midpoints_edges

    @wraps(add_midpoints_faces)
    def add_midpoints_faces(self):
        return add_midpoints_faces

    @wraps(add_midpoints_volumes)
    def add_midpoints_volumes(self):
        return add_midpoints_volumes
