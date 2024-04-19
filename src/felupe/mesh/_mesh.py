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

from functools import wraps

import numpy as np

from ..tools._plot import ViewMesh
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
    fill_between,
    flip,
    merge_duplicate_cells,
    merge_duplicate_points,
    mirror,
    revolve,
    rotate,
    runouts,
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
    npoints : int
        Amount of points.
    dim : int
        Dimension of mesh point coordinates.
    ndof : int
        Amount of degrees of freedom.
    ncells : int
        Amount of cells.
    points_with_cells : ndarray
        Array with points connected to cells.
    points_without_cells : ndarray
        Array with points not connected to cells.
    cells_per_point : ndarray
        Array which counts connected cells per point. Used for averaging results.

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import numpy as np
       >>> import felupe as fem
       >>>
       >>> points = np.array(
       ...     [[0.0, 0.0], [0.5, 0.1], [1.0, 0.2], [0.0, 1.0], [0.5, 0.9], [1.0, 0.8]]
       ... )
       >>> cells = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
       >>> mesh = fem.Mesh(points, cells, cell_type="quad")
       >>>
       >>> mesh.plot().show()


    See Also
    --------
    felupe.MeshContainer : A container which operates on a list of meshes with identical
        dimensions.
    felupe.Rectangle : A rectangular 2d-mesh with quads between ``a`` and ``b`` with
        ``n`` points per axis.
    felupe.Cube : A cube shaped 3d-mesh with hexahedrons between ``a`` and ``b`` with
        ``n`` points per axis.
    felupe.Grid : A grid shaped 3d-mesh with hexahedrons. Basically a wrapper for
        :func:`numpy.meshgrid` with  default ``indexing="ij"``.
    felupe.Circle : A circular shaped 2d-mesh with quads and ``n`` points on the
        circumferential edge of a 45-degree section. 90-degree ``sections`` are placed
        at given angles in degree.
    felupe.mesh.Triangle : A triangular shaped 2d-mesh with quads and ``n`` points at
        the edges of the three sub-quadrilaterals.

    """

    def __init__(self, points, cells, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells)
        self.cell_type = cell_type

        super().__init__(points=points, cells=cells, cell_type=cell_type)

        self.__mesh__ = Mesh

        # alias
        self.sweep = self.merge_duplicate_points
        self.save = self.write

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

        See Also
        --------
        felupe.Mesh.dual : Create a new dual mesh with given points per cell.
        """

        return self.dual(
            points_per_cell=points_per_cell,
            disconnect=True,
            calc_points=calc_points,
            offset=0,
            npoints=None,
        )

    def as_meshio(self, **kwargs):
        """Export the mesh as :class:`meshio.Mesh`.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword-arguments for ``meshio.Mesh(points, cells, **kwargs)``.

        Returns
        -------
        meshio.Mesh
            The mesh as :class:`meshio.Mesh`.
        """

        import meshio

        points = np.pad(self.points, ((0, 0), (0, 3 - self.points.shape[1])))

        return meshio.Mesh(points=points, cells={self.cell_type: self.cells}, **kwargs)

    def as_pyvista(self, cell_type=None, **kwargs):
        """Export the mesh as :class:`pyvista.UnstructuredGrid`.

        Parameters
        ----------
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).
        **kwargs : dict, optional
            Additional keyword-arguments for :class:`pyvista.UnstructuredGrid`.

        Returns
        -------
        pyvista.UnstructuredGrid
            The mesh as :class:`pyvista.UnstructuredGrid`.
        """

        import pyvista as pv

        if cell_type is None:
            cell_type = {
                "line": pv.CellType.LINE,
                "triangle": pv.CellType.TRIANGLE,
                "triangle6": pv.CellType.QUADRATIC_TRIANGLE,
                "tetra": pv.CellType.TETRA,
                "tetra10": pv.CellType.QUADRATIC_TETRA,
                "quad": pv.CellType.QUAD,
                "quad8": pv.CellType.QUADRATIC_QUAD,
                "quad9": pv.CellType.BIQUADRATIC_QUAD,
                "hexahedron": pv.CellType.HEXAHEDRON,
                "hexahedron20": pv.CellType.QUADRATIC_HEXAHEDRON,
                "hexahedron27": pv.CellType.TRIQUADRATIC_HEXAHEDRON,
                "VTK_LAGRANGE_HEXAHEDRON": pv.CellType.LAGRANGE_HEXAHEDRON,
                "VTK_LAGRANGE_QUADRILATERAL": pv.CellType.LAGRANGE_QUADRILATERAL,
                "VTK_LAGRANGE_LINE": pv.CellType.LAGRANGE_CURVE,
            }[self.cell_type]

        points = np.pad(self.points, ((0, 0), (0, 3 - self.points.shape[1])))
        cells = np.pad(
            self.cells, ((0, 0), (1, 0)), constant_values=self.cells.shape[1]
        )
        cell_types = cell_type * np.ones(self.ncells, dtype=int)

        return pv.UnstructuredGrid(cells, cell_types, points)

    def write(self, filename="mesh.vtk", **kwargs):
        """Write the mesh to a file.

        Parameters
        ----------
        filename : str, optional
            The filename of the mesh (default is ``mesh.vtk``).
        **kwargs : dict, optional
            Additional keyword arguments for :meth:`meshio.Mesh.write`.

        Notes
        -----
        ..  note::
            For XDMF-export please ensure to have ``h5py`` (as an optional dependency of
            ``meshio``) installed.

        Examples
        --------
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=3)
        >>> mesh.write(filename="mesh.vtk")

        See Also
        --------
        felupe.mesh.read : Read a mesh from a file using :func:`meshio.read`.
        felupe.Mesh.write : Write the mesh to a file.

        """

        self.as_meshio(**kwargs).write(filename)

    def view(self, point_data=None, cell_data=None, cell_type=None):
        """View the mesh with optional given dicts of point- and cell-data items.

        Parameters
        ----------
        point_data : dict or None, optional
            Additional point-data dict (default is None).
        cell_data : dict or None, optional
            Additional cell-data dict (default is None).
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).

        Returns
        -------
        felupe.ViewMesh
            A object which provides visualization methods for :class:`felupe.Mesh`.

        See Also
        --------
        felupe.ViewMesh : Visualization methods for :class:`~felupe.Mesh`.
        """

        return ViewMesh(
            self,
            point_data=point_data,
            cell_data=cell_data,
            cell_type=cell_type,
        )

    def plot(self, *args, **kwargs):
        """Plot the mesh.

        See Also
        --------
        felupe.Scene.plot : Plot method of a scene.
        """
        return self.view().plot(*args, show_undeformed=False, **kwargs)

    def screenshot(
        self,
        *args,
        filename="mesh.png",
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the mesh.

        See Also
        --------
        pyvista.Plotter.screenshot : Take a screenshot of a PyVista plotter.
        """

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

    def imshow(self, *args, ax=None, **kwargs):
        """Take a screenshot of the mesh, show the image data in a figure and return the
        ax.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

        ax.imshow(self.screenshot(*args, filename=None, **kwargs))
        ax.set_axis_off()

        return ax

    def get_point_ids(self, value, fun=np.isclose, mode=np.all, **kwargs):
        """Return point ids for points which are close to a given value.

        Parameters
        ----------
        value : float, list or ndarray
            Scalar value or point coordinates.
        fun : callable, optional
            The function used to compare the points to the given value, i.e. with a
            function signature ``fun(mesh.points, value, **kwargs)``. Default is
            :func:`numpy.isclose`.
        mode : callable, optional
            A callable used to combine the search results, either :func:`numpy.any` or
            :func:`numpy.all`.
        **kwargs : dict, optional
            Additional keyword arguments for ``fun(mesh.points, value, **kwargs)``.

        Returns
        -------
        ndarray
            Array with point ids.

        Examples
        --------
        Get point ids at given coordinates for a mesh with duplicate points.

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=11)
        >>> mesh.update(points=np.vstack([mesh.points, mesh.points]))
        >>> point_ids = mesh.get_point_ids([0, 1, 1])
        >>> point_ids
        array([1320, 2651])

        >>> mesh.points[point_ids]
        array([[0., 1., 1.],
               [0., 1., 1.]])

        """
        return np.argwhere(mode(fun(self.points, value, **kwargs), axis=1))[:, 0]

    def get_cell_ids(self, point_ids):
        """Return cell ids which have the given point ids in their connectivity.

        Parameters
        ----------
        point_ids : list or ndarray
            Array with point ids which are used to search for cells.

        Returns
        -------
        ndarray
            Array with cell ids which have the given point ids in their connectivity.

        Examples
        --------
        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=11)
        >>> point_ids = mesh.get_point_ids([0, 1, 1])
        >>> point_ids
        array([1320])

        >>> cell_ids = mesh.get_cell_ids(point_ids)
        >>> cell_ids
        array([990])

        """
        return np.argwhere(np.isin(self.cells, point_ids).any(axis=1))[:, 0]

    def get_cell_ids_neighbours(self, cell_ids):
        """Return cell ids which share points with given cell ids.

        Parameters
        ----------
        cell_ids : list or ndarray
            Array with cell ids which are used to search for neighbour cells.

        Returns
        -------
        ndarray
            Array with cell ids which are next to the given cells.

        Examples
        --------
        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=11)
        >>> point_ids = mesh.get_point_ids([0, 1, 1])
        >>> point_ids
        array([1320])

        >>> cell_ids = mesh.get_cell_ids(point_ids)
        >>> cell_ids
        array([990])

        Find the cell ids which share at least one point with the given cell id(s).

        >>> cell_ids_neighbours = mesh.get_cell_ids_neighbours(cell_ids)
        >>> cell_ids_neighbours
        array([880, 881, 890, 891, 980, 981, 990, 991])

        """
        return self.get_cell_ids(self.cells[cell_ids])

    def get_point_ids_shared(self, cell_ids_neighbours):
        """Return shared point ids for given cell ids.

        Parameters
        ----------
        cells_neighbours : list or ndarray
            Array with cell ids.

        Returns
        -------
        ndarray
            Array with point ids which are connected to all given cell neighbours.

        Examples
        --------
        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=11)
        >>> point_ids = mesh.get_point_ids([0, 1, 1])
        >>> point_ids
        array([1320])

        >>> cell_ids = mesh.get_cell_ids(point_ids)
        >>> cell_ids
        array([990])

        Find the cell ids which share at least one point with the given cell id(s).

        >>> cell_ids_neighbours = mesh.get_cell_ids_neighbours(cell_ids)
        >>> cell_ids_neighbours
        array([880, 881, 890, 891, 980, 981, 990, 991])

        Find the shared point ids for the list of cell ids.

        >>> point_ids_shared = mesh.get_point_ids_shared(cell_ids_neighbours)
        >>> point_ids_shared
        array([1189])

        """

        neighbours = self.cells[cell_ids_neighbours]
        cell = neighbours[0]
        return cell[[np.isin(neighbours, point).any(axis=1).all() for point in cell]]

    def get_point_ids_corners(self):
        """Return point ids which are located at (xmin, ymin), (xmax, ymin), etc.

        Returns
        -------
        ndarray
            Array with point ids which are located at the corners.

        Examples
        --------
        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=11)
        >>> point_ids_corners = mesh.get_point_ids_corners()
        >>> point_ids_corners
        array([   0, 1210,   10, 1220,  110, 1320,  120, 1330])

        """

        xmin = np.min(self.points, axis=0)
        xmax = np.max(self.points, axis=0)
        points = np.vstack(
            [x.ravel() for x in np.meshgrid(*np.vstack([xmin, xmax]).T)]
        ).T

        return np.concatenate([self.get_point_ids(point) for point in points])

    def modify_corners(self, point_ids=None):
        """Modify the corners of a regular rectangle (quad) or cube (hexahedron)
        inplace. Only the cells array is modified, the points array remains unchanged.

        Parameters
        ----------
        point_ids : ndarray or None, optional
            Array with point ids located at the corners which are modified (default is
            None). If None, all corners are modified.

        Notes
        -----
        Description of the algorithm:

        1. Get corner point ids.

        For each corner point:

        2. Get attached cell and find cell neighours.
        3. Get the shared point of the cell and its neighbours.
        4. Get pair-wise shared points which are located on an edge.
        5. Replace the shared points with the corner point.
        6. Delete the cell attached to the corner point.

        Examples
        --------
        ..  pyvista-plot::
            :context:

            >>> import numpy as np
            >>> import felupe as fem
            >>>
            >>> mesh = fem.Rectangle(b=(3, 1), n=(16, 6))
            >>> mesh.plot().show()

        ..  pyvista-plot::
            :context:

            >>> mesh = mesh.modify_corners()  # inplace
            >>> mesh.plot().show()

        """

        if self.cell_type not in ["quad", "hexahedron"]:
            message = [
                "Cell type not supported.",
                "Must be either 'quad' or 'hexahedron'",
                f"but given cell type is '{self.cell_type}'.",
            ]
            raise TypeError(" ".join(message))

        if point_ids is None:
            point_ids = self.get_point_ids_corners()

        for point_id in point_ids:
            cell_id = self.get_cell_ids(point_id)[0]
            cell_id_with_neighbours = self.get_cell_ids_neighbours(cell_id)

            cell_ids_neighbours = cell_id_with_neighbours[
                cell_id_with_neighbours != cell_id
            ]

            point_id_shared = self.get_point_ids_shared(cell_id_with_neighbours)[0]
            point_ids_shared_individual = [
                self.get_point_ids_shared([cell_id, neighbour])
                for neighbour in cell_ids_neighbours
            ]
            if self.cell_type == "hexahedron":
                edges = np.argwhere(
                    np.isclose(self.points, self.points[point_id]).sum(axis=1) >= 2
                )[:, 0]
                point_ids_shared_individual = [
                    p[np.isin(p, edges)] for p in point_ids_shared_individual
                ]
            point_ids_shared_individual = [
                shared[shared != point_id_shared]
                for shared in point_ids_shared_individual
            ]

            for shared, neighbour in zip(
                point_ids_shared_individual, cell_ids_neighbours
            ):
                point_id_replace = np.argwhere(np.isin(self.cells[neighbour], shared))
                if len(point_id_replace) > 0:
                    self.cells[neighbour, point_id_replace[0][0]] = point_id

            self.cells = np.delete(self.cells, cell_id, axis=0)
            self.update(cells=self.cells)

        return self

    def dual(
        self,
        points_per_cell=None,
        disconnect=True,
        calc_points=False,
        offset=0,
        npoints=None,
    ):
        """Create a new dual mesh with given points per cell.

        Parameters
        ----------
        points_per_cell : int or None, optional
            Number of points per cell, must be equal or lower than ``cells.shape[1]`` (
            default is None). If None, all points per cell are considered for the dual
            mesh.
        disconnect : bool, optional
            A flag to disconnect the mesh (each cell has its own points). Default is
            True.
        calc_points : bool, optional
            A flag to calculate the point coordinates for the dual mesh (default is
            False). If False, the points array is filled with zeros.
        offset : int, optional
            An offset to be added to the cells array (default is 0).
        npoints : int or None, optional
            Number of points for the dual mesh. If the given number of points is greater
            than ``npoints * points_per_cell``, then the missing points are added to the
            points array (filled with zeros). Default is None.

        Returns
        -------
        Mesh
            The dual mesh.

        Notes
        -----
        ..  note::
            The points array of the dual mesh always has a shape of
            ``(npoints * points_per_cell, dim)``.

        Examples
        --------
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=5).add_midpoints_edges()
        >>> region = fem.RegionQuadraticQuad(mesh=mesh)
        >>>
        >>> mesh_dual = mesh.dual(points_per_cell=1, disconnect=False)
        >>> region_dual = fem.RegionConstantQuad(
        ...     mesh_dual, quadrature=region.quadrature, grad=False
        ... )
        >>>
        >>> displacement = fem.FieldPlaneStrain(region, dim=2)
        >>> pressure = fem.Field(region_dual)
        >>> field = fem.FieldContainer([displacement, pressure])

        See Also
        --------
        felupe.mesh.dual : Create a new dual mesh with given points per cell.

        """
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

    def expand(self, n=11, z=1, axis=-1, expand_dim=True):
        """Expand a 0d-Point to a 1d-Line, a 1d-Line to a 2d-Quad or a 2d-Quad to a
        3d-Hexahedron Mesh.

        Parameters
        ----------
        n : int, optional
            Number of n-point repetitions or (n-1)-cell repetitions, default is 11. Must
            be greater than 0.
        z : float or ndarray, optional
            Total expand dimension as float (edge length in expand direction is z / n),
            default is 1. Optionally, if an array is passed these entries are taken as
            expansion and ``n`` is ignored.
        axis : int, optional
            Axis of expansion (default is -1).
        mask : ndarray or None, optional
            A boolean mask to select points which are rotated (default is None).
        expand_dim : bool, optional
            Expand the dimension of the point coordinates (default is True).

        Returns
        -------
        Mesh
            The expanded mesh.

        Examples
        --------
        Expand a rectangle to a cube.

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> rect = fem.Rectangle(n=4)
           >>> cube = rect.expand(n=7, z=2)
           >>>
           >>> cube.plot().show()

        >>> cube
        <felupe Mesh object>
          Number of points: 112
          Number of cells:
            hexahedron: 54

        See Also
        --------
        felupe.mesh.expand : Expand a 0d-Point to a 1d-Line, a 1d-Line to a 2d-Quad or a
            2d-Quad to a 3d-Hexahedron Mesh.
        """
        return as_mesh(expand(self, n=n, z=z, axis=axis, expand_dim=expand_dim))

    def rotate(self, angle_deg, axis, center=None, mask=None):
        """Rotate a Mesh.

        Parameters
        ----------
        angle_deg : int
            Rotation angle in degree.
        axis : int
            Rotation axis.
        center : list or ndarray or None, optional
            Center point coordinates (default is None).
        mask : ndarray or None, optional
            A boolean mask to select points which are rotated (default is None).

        Returns
        -------
        Mesh
            The rotated mesh.

        Examples
        --------
        Rotate a rectangle in the xy-plane by 35 degree.

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> rect = fem.Rectangle(b=(3, 1), n=(10, 4))
           >>> mesh = rect.rotate(angle_deg=35, axis=2, center=[1.5, 0.5])
           >>> mesh.plot().show()

        >>> mesh
        <felupe Mesh object>
          Number of points: 40
          Number of cells:
            quad: 27

        See Also
        --------
        felupe.mesh.rotate : Rotate a Mesh.
        """
        return as_mesh(
            rotate(self, angle_deg=angle_deg, axis=axis, center=center, mask=mask)
        )

    def revolve(self, n=11, phi=180, axis=0, expand_dim=True):
        """Revolve a 2d-Quad to a 3d-Hexahedron Mesh.

        Parameters
        ----------
        n : int, optional
            Number of n-point revolutions (or (n-1) cell revolutions),
            default is 11.
        phi : float or ndarray, optional
            Revolution angle in degree (default is 180).
        axis : int, optional
            Revolution axis (default is 0).
        expand_dim : bool, optional
            Expand the dimension of the point coordinates (default is True).

        Returns
        -------
        Mesh
            The revolved mesh.

        Examples
        --------
        Revolve a cylinder from a rectangle.

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> rect = fem.Rectangle(a=(0, 4), b=(3, 5), n=(10, 4))
           >>> mesh = rect.revolve(n=11, phi=180, axis=0)
           >>> mesh.plot().show()

        >>> mesh
        <felupe Mesh object>
          Number of points: 440
          Number of cells:
            hexahedron: 270

        See Also
        --------
        felupe.mesh.revolve : Revolve a 2d-Quad to a 3d-Hexahedron Mesh.
        """
        return as_mesh(revolve(self, n=n, phi=phi, axis=axis, expand_dim=expand_dim))

    def merge_duplicate_points(self, decimals=None):
        """Merge duplicate points and update cells of a Mesh.

        Parameters
        ----------
        decimals : int or None, optional
            Number of decimals for point coordinate comparison (default is None).

        Returns
        -------
        Mesh
            The mesh with merged duplicate points and updated cells.

        Notes
        -----
        ..  warning::
            This function re-sorts points.

        ..  note::
            This function does not merge duplicate cells.

        Examples
        --------
        Two quad meshes to be merged overlap some points. Merge these duplicated
        points and update the cells.

        >>> import felupe as fem

        >>> rect1 = fem.Rectangle(n=11)
        >>> rect2 = fem.Rectangle(a=(0.9, 0), b=(1.9, 1), n=11)
        >>> rect2
        <felupe Mesh object>
          Number of points: 121
          Number of cells:
            quad: 100

        Each mesh contains 121 points and 100 cells. These two meshes are now stored in a
        :class:`~felupe.MeshContainer`.

        >>> container = fem.MeshContainer([rect1, rect2])
        >>> container
        <felupe mesh container object>
          Number of points: 242
          Number of cells:
            quad: 100
            quad: 100

        ..  image:: images/mesh_container.png
            :width: 400px

        The meshes of the mesh container are :func:`stacked <felupe.mesh.stack>`.

        >>> stack = fem.mesh.stack(container.meshes)
        >>> stack
        <felupe Mesh object>
          Number of points: 242
          Number of cells:
            quad: 200

        After merging the duplicated points and cells, the number of points is reduced but
        the number of cells is unchanged.

        >>> mesh = stack.merge_duplicate_points()
        >>> mesh
        <felupe Mesh object>
          Number of points: 220
          Number of cells:
            quad: 200

        >>> ax = mesh.imshow(opacity=0.6)

        ..  image:: images/mesh_sweep.png
            :width: 400px

        ..  note::
            The :class:`~felupe.MeshContainer` may be directly created with
            ``merge=True``. This enforces :func:`~felupe.mesh.merge_duplicate_points`
            for the shared points array of the container.

        See Also
        --------
        felupe.mesh.merge_duplicate_points : Merge duplicated points and update cells of
            a Mesh.
        felupe.MeshContainer : A container which operates on a list of meshes with
            identical dimensions.
        """
        return as_mesh(merge_duplicate_points(self, decimals=decimals))

    @wraps(merge_duplicate_cells)
    def merge_duplicate_cells(self):
        """Merge duplicate cells of a Mesh.

        Returns
        -------
        Mesh
            The mesh with merged duplicate cells.

        Notes
        -----
        ..  warning::
            This function re-sorts cells.

        ..  note::
            This function does not merge duplicate points.

        Examples
        --------
        Two quad meshes to be merged overlap some cells. Merge these duplicated
        points and update the cells.

        >>> import felupe as fem

        >>> rect1 = fem.Rectangle(n=11)
        >>> rect2 = fem.Rectangle(a=(0.9, 0), b=(1.9, 1), n=11)
        >>> rect2
        <felupe Mesh object>
          Number of points: 121
          Number of cells:
            quad: 100

        Each mesh contains 121 points and 100 cells. These two meshes are now stored in
        a :class:`~felupe.MeshContainer`.

        >>> container = fem.MeshContainer([rect1, rect2])
        >>> container
        <felupe mesh container object>
          Number of points: 242
          Number of cells:
            quad: 100
            quad: 100

        The meshes of the mesh container are :func:`stacked <felupe.mesh.stack>`.

        >>> stack = fem.mesh.stack(container.meshes)
        >>> stack
        <felupe Mesh object>
          Number of points: 242
          Number of cells:
            quad: 200

        After merging the duplicated points and cells, the number of points is reduced
        but the number of cells is unchanged.

        >>> mesh = stack.merge_duplicate_points()
        >>> mesh
        <felupe Mesh object>
          Number of points: 220
          Number of cells:
            quad: 200

        >>> ax = mesh.imshow(opacity=0.6)

        ..  image:: images/mesh_sweep.png
            :width: 400px

        ..  note::
            The :class:`~felupe.MeshContainer` may be directly created with
            ``merge=True``. This enforces :func:`~felupe.mesh.merge_duplicate_points`
            for the shared points array of the container.

        The duplicate cells are merged in a second step.

        >>> merged = mesh.merge_duplicate_cells()
        >>> merged
        <felupe Mesh object>
          Number of points: 220
          Number of cells:
            quad: 190

        ..  image:: images/mesh_merged.png
            :width: 400px

        See Also
        --------
        felupe.mesh.merge_duplicate_points : Merge duplicate points of a Mesh.
        felupe.mesh.merge_duplicate_cells : Merge duplicate cells of a Mesh.
        felupe.MeshContainer : A container which operates on a list of meshes with
            identical dimensions.
        """
        return as_mesh(merge_duplicate_cells(self))

    def fill_between(self, other_mesh, n=11):
        """Fill a 2d-Quad Mesh between two 1d-Line Meshes, embedded in 2d-space, or a
        3d-Hexahedron Mesh between two 2d-Quad Meshes, embedded in 3d-space, by expansion.
        Both meshes must have equal number of points and cells. The cells-array is taken
        from the first mesh.

        Parameters
        ----------
        other_mesh : felupe.Mesh
            The other line- or quad-mesh.
        n : int or ndarray
            Number of n-point repetitions or (n-1)-cell repetitions,
            (default is 11). If an array is given, then its values are used for the
            relative positions in a reference configuration (-1, 1) between the two meshes.

        Returns
        -------
        felupe.Mesh
            The expanded mesh.

        Examples
        --------
        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> inner = fem.mesh.revolve(fem.Point(1)).expand(z=0.4).translate(0.2, axis=2)
           >>> outer = fem.mesh.revolve(fem.Point(2), phi=160).rotate(
           ...     axis=2, angle_deg=20
           ... ).expand(z=1.2)
           >>> mesh = inner.fill_between(outer, n=6)
           >>>
           >>> mesh.plot().show()

        See Also
        --------
        felupe.mesh.fill_between : Fill a 2d-Quad Mesh between two 1d-Line Meshes, embedded
            in 2d-space, or a 3d-Hexahedron Mesh between two 2d-Quad Meshes, embedded in
            3d-space, by expansion.

        """
        return as_mesh(fill_between(self, other_mesh=other_mesh, n=n))

    def flip(self, mask=None):
        """Ensure positive cell volumes for `tria`, `tetra`, `quad` and `hexahedron`
        cell types.

        Parameters
        ----------
        mask: list, ndarray or None, optional
            Boolean mask for selected cells to flip (default is None). If None, all
            cells are selected to be flipped.

        Returns
        -------
        Mesh
            The mesh with a rearranged cells array to ensure positive cell volumes.

        Examples
        --------
        A quad mesh with negative cell volumes occurs if one coordinate axis is multiplied
        by -1. The error pops up if a region is created with this mesh.

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=3)
        >>> mesh.update(points=mesh.points * np.array([[-1, 1]]))
        >>> region = fem.RegionQuad(mesh)

        The sum of the differential volumes :math:`V = \sum_c \sum_q dV_{qc}` is evaluated
        to -1.0.

        >>> region.dV.sum()
        -1.0

        Let's try to fix the mesh.

        >>> mesh.cells
        array([[0, 1, 4, 3],
               [1, 2, 5, 4],
               [3, 4, 7, 6],
               [4, 5, 8, 7]])

        The cells array is rearranged to ensure positive cell volumes.

        >>> mesh_fixed = mesh.flip()
        >>> mesh_fixed.cells
        array([[3, 4, 1, 0],
               [4, 5, 2, 1],
               [6, 7, 4, 3],
               [7, 8, 5, 4]])

        A region now correctly evaluates the total volume of the mesh to 1.0.

        >>> region_fixed = fem.RegionQuad(mesh_fixed)
        >>> region_fixed.dV.sum()
        1.0

        See Also
        --------
        felupe.mesh.flip : Ensure positive cell volumes for `tria`, `tetra`, `quad` and
            `hexahedron` cell types.

        """
        return as_mesh(flip(self, mask=mask))

    def mirror(self, normal=[1, 0, 0], centerpoint=[0, 0, 0], axis=None):
        """Mirror points by plane normal and ensure positive cell volumes for
        `tria`, `tetra`, `quad` and `hexahedron` cell types.

        Parameters
        ----------
        normal: list or ndarray, optional
            Mirror-plane normal vector (default is [1, 0, 0]).
        centerpoint: list or ndarray, optional
            Center-point coordinates on the mirror plane (default is [0, 0, 0]).
        axis: int or None, optional
            Mirror axis (default is None).

        Returns
        -------
        Mesh
            The mirrored mesh.

        Examples
        --------
        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Circle(sections=[0, 90, 180], n=5)
           >>> mesh.plot().show()

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Circle(sections=[0, 90, 180], n=5)
           >>> mesh.mirror(normal=[0, 1, 0]).plot().show()

        See Also
        --------
        felupe.Mesh.mirror : Mirror points by plane normal and ensure positive cell
            volumes for `tria`, `tetra`, `quad` and `hexahedron` cell types.
        """
        return as_mesh(mirror(self, normal=normal, centerpoint=centerpoint, axis=axis))

    def translate(self, move, axis):
        """Translate (move) a Mesh along a given axis.

        Parameters
        ----------
        move : float
            Translation along given axis.
        axis : int
            Translation axis.

        Returns
        -------
        Mesh
            The translated mesh.

        Examples
        --------
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Circle(n=6)
        >>> mesh.points.min(axis=0), mesh.points.max(axis=0)
        (array([-1., -1.]), array([1., 1.]))

        >>> translated = mesh.translate(0.3, axis=1)
        >>> translated.points.min(axis=0), translated.points.max(axis=0)
        (array([-1. , -0.7]), array([1. , 1.3]))

        See Also
        --------
        felupe.mesh.translate : Translate (move) a Mesh along a given axis.
        """

        return as_mesh(translate(self, move=move, axis=axis))

    def triangulate(self, mode=3):
        """Triangulate a quad or a hex mesh.

        Parameters
        ----------
        mode: int, optional
            Choose a mode how to convert hexahedrons to tets [1]_ (default is 3).

        Returns
        -------
        Mesh
            The triangulated mesh.

        Examples
        --------
        Use ``mode=0`` to convert a mesh of hexahedrons into tetrahedrons [1]_.

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Cube(n=6)
           >>> mesh.triangulate(mode=0).plot().show()

        Use ``mode=3`` to convert a mesh of hexahedrons into tetrahedrons [1]_.

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Cube(n=6)
           >>> mesh.triangulate(mode=3).plot().show()

        References
        ----------
        .. [1] Dompierre, J., Labbé, P., Vallet, M. G., & Camarero, R. (1999).
           How to Subdivide Pyramids, Prisms, and Hexahedra into Tetrahedra.
           IMR, 99, 195.

        See Also
        --------
        felupe.mesh.triangulate : Triangulate a quad or a hex mesh.
        """
        return as_mesh(triangulate(self, mode=mode))

    def add_runouts(
        self,
        values=[0.1, 0.1],
        centerpoint=[0, 0, 0],
        axis=0,
        exponent=5,
        mask=slice(None),
        normalize=False,
    ):
        """Add simple rubber-runouts for realistic rubber-metal structures.

        Parameters
        ----------
        values : list or ndarray, optional
            Relative amount of runouts (per coordinate) perpendicular to the axis
            (default is 10% per coordinate, i.e. [0.1, 0.1]).
        centerpoint : list or ndarray, optional
            Center-point coordinates (default is [0, 0, 0]).
        axis : int or None, optional
            Axis (default is 0).
        exponent : int, optional
            Positive exponent to control the shape of the runout. The higher
            the exponent, the steeper the transition (default is 5).
        mask : list or None, optional
            List of points to be considered (default is None).
        normalize : bool, optional
            Normalize the runouts to create indents, i.e. maintain the original shape at the
            ends (default is False).

        Returns
        -------
        Mesh
            The mesh with the modified point coordinates.

        Examples
        --------
        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> rect = fem.Rectangle(a=(-3, -1), b=(3, 1), n=(31, 11))
           >>> mesh = rect.add_runouts(axis=1, values=[0.2], normalize=True)
           >>>
           >>> mesh.plot().show()

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> cube = fem.Cube(a=(-3, -2, -1), b=(3, 2, 1), n=(31, 21, 11))
           >>> mesh = cube.add_runouts(axis=2, values=[0.1, 0.3], normalize=True)
           >>>
           >>> mesh.plot().show()

        See Also
        --------
        felupe.mesh.runouts : Add simple rubber-runouts for realistic rubber-metal
            structures.

        """
        return as_mesh(
            runouts(
                self,
                values=values,
                centerpoint=centerpoint,
                axis=axis,
                exponent=exponent,
                mask=mask,
                normalize=normalize,
            )
        )

    def convert(
        self,
        order=0,
        calc_points=False,
        calc_midfaces=False,
        calc_midvolumes=False,
    ):
        """Convert a mesh to a given order. Only conversions to ``order=0`` and
        ``order=2`` are supported. This function supports meshes with cell types
        ``"triangle"``, ``"tetra"``, ``"quad"`` and ``"hexahedron"``.

        Parameters
        ----------
        order : int, optional
            The order of the converted mesh (default is 0). If 0, the points-array will
            be of shape ``(ncells, dim)``. If 0 and ``calc_points`` is True, the mean of
            all points per cell is evaluated. If 0 and ``calc_points`` is False, the
            points array is filled with zeros. If 2, at least midpoints on cell edges
            are added to the mesh. If 2 and ``calc_midfaces`` is True, midpoints on cell
            faces are also added. If 2 and ``calc_midvolumes`` is True, midpoints on
            cell volumes are also added. Raises an error if not 0 or 2.
        calc_points : bool, optional
            Flag to return the mean of all points per cell if ``order=0`` (default is
            False). If False, the points-array is filled with zeros.
        calc_midfaces : bool, optional
            Flag to add midpoints on cell faces if ``order=2`` (default is False).
        calc_midvolumes : bool, optional
            Flag to add midpoints on cell volumes if ``order=2`` (default is False).

        Returns
        -------
        Mesh
            The converted mesh.

        Examples
        --------
        Convert a mesh of hexahedrons to quadratic hexahedrons by inserting midpoints on
        the cell edges.

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Rectangle(n=6)
           >>> mesh2 = mesh.convert(order=2)
           >>>
           >>> plotter = mesh2.plot(plotter=mesh.plot(), style="points", color="black")
           >>> plotter.show()

        >>> mesh2
        <felupe Mesh object>
          Number of points: 96
          Number of cells:
            quad8: 25

        See Also
        --------
        felupe.mesh.add_midpoints_edges : Add midpoints on edges for given points and cells
            and update cell_type accordingly.
        felupe.mesh.add_midpoints_faces : Add midpoints on faces for given points and cells
            and update cell_type accordingly.
        felupe.mesh.add_midpoints_volumes : Add midpoints on volumes for given points and
            cells and update cell_type accordingly.
        felupe.Mesh.add_midpoints_edges : Add midpoints on edges for given points and cells
            and update cell_type accordingly.
        felupe.Mesh.add_midpoints_faces : Add midpoints on faces for given points and cells
            and update cell_type accordingly.
        felupe.Mesh.add_midpoints_volumes : Add midpoints on volumes for given points and
            cells and update cell_type accordingly.

        """
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

    def add_midpoints_edges(self, cell_type=None):
        """Add midpoints on edges for given points and cells and update cell_type
        accordingly.

        Parameters
        ----------
        cell_type: str or None, optional
            A string in VTK-convention that specifies the new cell type (default is None).
            If None, the cell type is chosen automatically.

        Returns
        -------
        Mesh
            A new mesh with inserted midpoints on cell edges.

        Examples
        --------
        Convert a mesh of hexahedrons to quadratic hexahedrons by inserting midpoints on
        the cell edges.

        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Rectangle(n=6)
           >>> mesh_with_midpoints_edges = mesh.add_midpoints_edges()
           >>>
           >>> mesh_with_midpoints_edges.plot(
           ...     plotter=mesh.plot(), style="points", color="black"
           ... ).show()

        >>> mesh_with_midpoints_edges
        <felupe Mesh object>
          Number of points: 96
          Number of cells:
            quad8: 25

        See Also
        --------
        felupe.mesh.add_midpoints_edges : Add midpoints on edges for given points and
            cells and update cell_type accordingly.

        """
        return add_midpoints_edges(self, cell_type_new=cell_type)

    def add_midpoints_faces(self, cell_type=None):
        """Add midpoints on faces for given points and cells and update cell_type
        accordingly.

        Parameters
        ----------
        cell_type: str or None, optional
            A string in VTK-convention that specifies the new cell type (default is
            None). If None, the cell type is chosen automatically.

        Returns
        -------
        Mesh
            A new mesh with inserted midpoints on cell faces.

        Examples
        --------
        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Rectangle(n=6)
           >>> mesh_with_midpoints_faces = mesh.add_midpoints_faces(cell_type="quad")
           >>>
           >>> mesh_with_midpoints_faces.plot(
           ...     plotter=mesh.plot(), style="points", color="black"
           ... ).show()

        >>> mesh_with_midpoints_faces
        <felupe Mesh object>
          Number of points: 61
          Number of cells:
            quad: 25

        See Also
        --------
        felupe.mesh.add_midpoints_faces : Add midpoints on faces for given points and
            cells and update cell_type accordingly.
        """
        return add_midpoints_faces(self, cell_type_new=cell_type)

    def add_midpoints_volumes(self, cell_type=None):
        """Add midpoints on volumes for given points and cells and update cell_type
        accordingly.

        Parameters
        ----------
        cell_type: str or None, optional
            A string in VTK-convention that specifies the new cell type (default is None).
            If None, the cell type is chosen automatically.

        Returns
        -------
        Mesh
            A new mesh with inserted midpoints on cell volumes.

        Examples
        --------
        .. pyvista-plot::
           :include-source: True

           >>> import felupe as fem
           >>>
           >>> mesh = fem.Cube(n=6)
           >>> mesh_with_midpoints_volumes = fem.mesh.add_midpoints_volumes(
           ...     mesh, cell_type_new="hexahedron9"
           ... )
           >>>
           >>> plotter = mesh.plot(opacity=0.5)
           >>> actor = plotter.add_points(mesh_with_midpoints_volumes.points, color="black")
           >>> plotter.show()

        >>> mesh_with_midpoints_volumes
        <felupe Mesh object>
          Number of points: 341
          Number of cells:
            hexahedron9: 125

        See Also
        --------
        felupe.mesh.add_midpoints_volumes : Add midpoints on volumes for given points
            and cells and update cell_type accordingly.
        """
        return add_midpoints_volumes(self, cell_type_new=cell_type)
