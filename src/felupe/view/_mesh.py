# SPDX-License-Identifier: GPL-3.0-or-later

from copy import copy

import numpy as np

from ._backends import PyVistaBackend
from ._scene import Scene

NORMALS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-x": (-1.0, 0.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "-z": (0.0, 0.0, -1.0),
}


def _pad(x):
    "Pad the last axis of an array of coordinates with zeros to three components."

    x = np.asarray(x, dtype=float)
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 3 - x.shape[-1])])


def _vector(x, name):
    "Return a vector with one to three components, padded with zeros to three."

    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or not 1 <= len(x) <= 3:
        raise ValueError(
            f"The {name} must be a vector with one to three components, "
            f"not of shape {x.shape}."
        )

    return _pad(x)


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

    def clip(self, normal="x", origin=None, invert=True, value=0.0, crinkle=True):
        r"""Clip the scene by a plane and return the clipped scene.

        The arguments are those of :meth:`pyvista.DataSetFilters.clip`. By default,
        the clip is crinkled, i.e. the cells are kept as a whole and not cut by the
        plane. All cells with at least one point on the kept side of the plane are
        kept, together with their point- and cell-data. Unused points are removed.
        The crinkled clip is evaluated by NumPy and it is supported by all plotting
        backends.

        Parameters
        ----------
        normal : str or array_like, optional
            The normal vector of the plane, e.g. ``"x"``, ``"-y"`` or ``(1, 1, 0)``
            (default is ``"x"``). A given vector is normalized.
        origin : array_like or None, optional
            A point on the plane (default is None). If None, the center of the bounds
            of the points is used.
        invert : bool, optional
            A flag to keep the part of the scene behind the plane, i.e. on the
            opposite side of the normal vector (default is True). If False, the part
            in front of the plane is kept.
        value : float, optional
            The plane is shifted by this value along the (normalized) normal vector
            (default is 0.0).
        crinkle : bool, optional
            A flag to keep the clipped cells as a whole (default is True). If False,
            the cells are cut by the plane by :meth:`pyvista.DataSetFilters.clip`.
            This requires PyVista and the clipped scene is only supported by the
            plotting backend ``"pyvista"``.

        Returns
        -------
        felupe.view.Scene
            The clipped scene. A crinkled clip returns a scene of the same class,
            e.g. a :class:`~felupe.ViewField`, otherwise a :class:`~felupe.view.Scene`
            with the clipped mesh as :class:`pyvista.UnstructuredGrid`.

        Notes
        -----
        The scene is clipped in the undeformed configuration, i.e. the points of the
        mesh are not displaced by the point-data ``"Displacement"``.

        Examples
        --------
        ..  pyvista-plot::
            :force_static:

            >>> import felupe as fem
            >>>
            >>> mesh = fem.Cube(n=6)
            >>> view = fem.ViewMesh(mesh, cell_data={"Cell ID": range(mesh.ncells)})
            >>>
            >>> view.clip("-x", origin=(0.3, 0, 0)).plot("Cell ID").show()
        """

        if not crinkle:
            scene = Scene()
            scene.mesh = self.mesh.clip(
                normal=normal, origin=origin, invert=invert, value=value
            )
            return scene

        if isinstance(normal, str):
            if normal.lower() not in NORMALS:
                raise ValueError(
                    f"The normal must be one of {list(NORMALS)} or a vector, "
                    f'not "{normal}".'
                )
            normal = NORMALS[normal.lower()]

        normal = _vector(normal, "normal")
        length = np.linalg.norm(normal)

        if not length > 0:
            raise ValueError("The normal must be a non-zero vector.")

        normal = normal / length
        points = _pad(self.points)

        if origin is None:
            origin = (points.min(axis=0) + points.max(axis=0)) / 2

        origin = _vector(origin, "origin")

        # the signed distances of the points to the (shifted) plane
        distance = (points - origin) @ normal - value

        if invert:
            kept_points = distance < 0
        else:
            kept_points = distance > 0

        cells = np.asarray(self.cells)
        kept_cells = kept_points[cells].any(axis=1)

        # remove unused points and renumber the point-connectivity of the cells
        used = np.unique(cells[kept_cells])
        index = np.zeros(len(points), dtype=cells.dtype)
        index[used] = np.arange(len(used), dtype=cells.dtype)

        scene = copy(self)
        scene.points = self.points[used]
        scene.cells = index[cells[kept_cells]]
        scene.point_data = {
            label: np.asarray(data)[used] for label, data in self.point_data.items()
        }
        scene.cell_data = {
            label: np.asarray(data)[kept_cells]
            for label, data in self.cell_data.items()
        }
        scene._mesh = None

        return scene
