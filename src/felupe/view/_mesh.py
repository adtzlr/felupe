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

from ._scene import Scene


class ViewMesh(Scene):
    r"""Provide Visualization methods for :class:`felupe.Mesh` with optional given
    dicts of point- and cell-data items.

    Parameters
    ----------
    mesh : felupe.Mesh
        The mesh object.
    point_data : dict or None, optional
        Additional point-data dict (default is None).
    cell_data : dict or None, optional
        Additional cell-data dict (default is None).
    cell_type : pyvista.CellType or None, optional
        Cell-type of PyVista (default is None).

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> displacement = np.arange(81).reshape(27, 3) / 300
        >>> view = fem.ViewMesh(mesh, point_data={"Displacement": displacement})
        >>>
        >>> view.plot("Displacement", component=None).show()

    See Also
    --------
    felupe.view.Scene : Base class for plotting a static scene.
    felupe.ViewField : Provide Visualization methods for a field container.
    felupe.ViewSolid : Provide Visualization methods for a field container or a
        solid body.

    """

    def __init__(self, mesh, point_data=None, cell_data=None, cell_type=None):
        self.mesh = mesh.as_pyvista(cell_type=cell_type)

        if point_data is None:
            point_data = {}

        if cell_data is None:
            cell_data = {}

        for label, data in point_data.items():
            self.mesh.point_data[label] = data

        for label, data in cell_data.items():
            self.mesh.cell_data[label] = data

        self.mesh.set_active_scalars(None)
        self.mesh.set_active_vectors(None)
        self.mesh.set_active_tensors(None)
