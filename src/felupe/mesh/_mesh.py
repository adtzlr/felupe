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
from ._dual import dual
from ._tools import (
    expand,
    flip,
    mirror,
    revolve,
    rotate,
    runouts,
    sweep,
    translate,
    triangulate,
)


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
        An optional string in VTK-convention that specifies the cell type (default is
        None). Necessary when a mesh is saved to a file.

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

        self.__mesh__ = Mesh

    def __repr__(self):
        header = "<felupe Mesh object>"
        points = f"  Number of points: {len(self.points)}"
        cells_header = "  Number of cells:"
        cells = [f"    {self.cell_type}: {self.ncells}"]

        return "\n".join([header, points, cells_header, *cells])

    def __str__(self):
        return self.__repr__()

    def disconnect(self, points_per_cell=None, calc_points=True):
        """Return a new instance of a Mesh with disconnected cells. Optionally, the
        points-per-cell may be specified (must be lower or equal the number of points-
        per-cell of the original Mesh). If the Mesh is to be used as a *dual* Mesh, then
        the point-coordinates do not have to be re-created because they are not used.
        """

        return self.dual(
            points_per_cell=points_per_cell,
            disconnect=True,
            calc_points=calc_points,
            offset=0,
            npoints=None,
        )

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

    @wraps(dual)
    def dual(
        self,
        points_per_cell=None,
        disconnect=True,
        calc_points=False,
        offset=0,
        npoints=None,
    ):
        return as_mesh(
            dual(
                self,
                points_per_cell=points_per_cell,
                disconnect=disconnect,
                calc_points=calc_points,
                offset=offset,
                npoints=npoints,
            )
        )

    @wraps(expand)
    def expand(self, n=11, z=1):
        return as_mesh(expand(self, n=n, z=z))

    @wraps(rotate)
    def rotate(self, angle_deg, axis, center=None):
        return as_mesh(rotate(self, angle_deg=angle_deg, axis=axis, center=center))

    @wraps(revolve)
    def revolve(self, n=11, phi=180, axis=0):
        return as_mesh(revolve(self, n=n, phi=phi, axis=axis))

    @wraps(sweep)
    def sweep(self, decimals=None):
        return as_mesh(sweep(self, decimals=decimals))

    @wraps(flip)
    def flip(self, mask=None):
        return as_mesh(flip(self, mask=mask))

    @wraps(mirror)
    def mirror(self, normal=[1, 0, 0], centerpoint=[0, 0, 0], axis=None):
        return as_mesh(mirror(self, normal=normal, centerpoint=centerpoint, axis=axis))

    @wraps(translate)
    def translate(self, move, axis):
        return as_mesh(translate(self, move=move, axis=axis))

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
        return collect_edges(self)

    @wraps(collect_faces)
    def collect_faces(self):
        return collect_faces(self)

    @wraps(collect_volumes)
    def collect_volumes(self):
        return collect_volumes(self)

    @wraps(add_midpoints_edges)
    def add_midpoints_edges(self):
        return add_midpoints_edges(self)

    @wraps(add_midpoints_faces)
    def add_midpoints_faces(self):
        return add_midpoints_faces(self)

    @wraps(add_midpoints_volumes)
    def add_midpoints_volumes(self):
        return add_midpoints_volumes(self)
