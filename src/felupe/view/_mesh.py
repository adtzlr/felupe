# SPDX-License-Identifier: GPL-3.0-or-later

from ._backends import PyVistaBackend
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
    points : ndarray
        Point coordinates of the mesh.
    cells : ndarray
        Point-connectivity of the cells of the mesh.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type of the mesh.
    point_data : dict
        The point-data dict.
    cell_data : dict
        The cell-data dict.
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`. It is created on first access, i.e.
        PyVista is only required for this attribute and for plotting.

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
        self.points = mesh.points
        self.cells = mesh.cells
        self.cell_type = mesh.cell_type

        if point_data is None:
            point_data = {}

        if cell_data is None:
            cell_data = {}

        self.point_data = dict(point_data)
        self.cell_data = dict(cell_data)

        self._pyvista_cell_type = cell_type
        self._mesh = None

    @property
    def mesh(self):
        "The mesh with point- and cell-data as :class:`pyvista.UnstructuredGrid`."

        if self._mesh is None:
            self._mesh = PyVistaBackend().as_unstructured_grid(
                self, cell_type=self._pyvista_cell_type
            )

        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
