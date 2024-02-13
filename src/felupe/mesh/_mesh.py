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
        felupe.ViewMesh : Visualization methods for :class:`felupe.Mesh`.
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
        felupe.Scene.plot: Plot method of a scene.
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
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

    def imshow(self, ax=None, *args, **kwargs):
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

    def expand(self, n=11, z=1):
        """Expand a 1d-Line to a 2d-Quad or a 2d-Quad to a 3d-Hexahedron Mesh.

        Parameters
        ----------
        n : int, optional
            Number of n-point repetitions or (n-1)-cell repetitions,
            default is 11.
        z : float or ndarray, optional
            Total expand dimension as float (edge length in expand direction is z / n),
            default is 1. Optionally, if an array is passed these entries are
            taken as expansion and `n` is ignored.

        Returns
        -------
        Mesh
            The expanded mesh.

        Examples
        --------
        Expand a rectangle to a cube.

        >>> import felupe as fem

        >>> rect = fem.Rectangle(n=4)
        >>> rect.expand(n=7, z=2)
        <felupe Mesh object>
          Number of points: 112
          Number of cells:
            hexahedron: 54

        ..  image:: images/mesh_expand.png
            :width: 400px

        See Also
        --------
        mesh.expand : Expand a 1d-Line to a 2d-Quad or a 2d-Quad to a 3d-Hexahedron
            Mesh.
        """
        return as_mesh(expand(self, n=n, z=z))

    def rotate(self, angle_deg, axis, center=None):
        """Rotate a Mesh.

        Parameters
        ----------
        angle_deg : int
            Rotation angle in degree.
        axis : int
            Rotation axis.
        center : list or ndarray or None, optional
            Center point coordinates (default is None).

        Returns
        -------
        Mesh
            The rotated mesh.

        Examples
        --------
        Rotate a rectangle in the xy-plane by 35 degree.

        >>> import felupe as fem

        >>> rect = fem.Rectangle(b=(3, 1), n=(10, 4))
        >>> rect.rotate(angle_deg=35, axis=2, center=[1.5, 0.5])
        <felupe Mesh object>
          Number of points: 40
          Number of cells:
            quad: 27

        ..  image:: images/mesh_rotate.png
            :width: 400px

        See Also
        --------
        mesh.rotate : Rotate a Mesh.
        """
        return as_mesh(rotate(self, angle_deg=angle_deg, axis=axis, center=center))

    def revolve(self, n=11, phi=180, axis=0):
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

        Returns
        -------
        Mesh
            The revolved mesh.

        Examples
        --------
        Revolve a cylinder from a rectangle.

        >>> import felupe as fem

        >>> rect = fem.Rectangle(a=(0, 4), b=(3, 5), n=(10, 4))
        >>> rect.revolve(n=11, phi=180, axis=0)
        <felupe Mesh object>
          Number of points: 440
          Number of cells:
            hexahedron: 270

        ..  image:: images/mesh_revolve.png
            :width: 400px

        See Also
        --------
        mesh.rotate : Revolve a 2d-Quad to a 3d-Hexahedron Mesh.
        """
        return as_mesh(revolve(self, n=n, phi=phi, axis=axis))

    def sweep(self, decimals=None):
        """Merge duplicated points and update cells of a Mesh.

        Parameters
        ----------
        decimals : int or None, optional
            Number of decimals for point coordinate comparison (default is None).

        Returns
        -------
        Mesh
            The mesh with merged duplicated points and updated cells.

        Notes
        -----
        ..  warning::
            This function re-sorts points.

        ..  note::
            This function does not merge duplicated cells.

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

        >>> mesh = fem.mesh.sweep(stack)
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
            ``merge=True``. This enforces :func:`~felupe.mesh.sweep` for the shared
            points array of the container.

        See Also
        --------
        mesh.sweep : Merge duplicated points and update cells of a Mesh.
        MeshContainer : A container which operates on a list of meshes with identical
            dimensions.
        """
        return as_mesh(sweep(self, decimals=decimals))

    @wraps(fill_between)
    def fill_between(self, other_mesh, n=11):
        return as_mesh(fill_between(self, other_mesh=other_mesh, n=n))

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
        normalize=False,
    ):
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
    def add_midpoints_edges(self, cell_type=None):
        return add_midpoints_edges(self, cell_type_new=cell_type)

    @wraps(add_midpoints_faces)
    def add_midpoints_faces(self, cell_type=None):
        return add_midpoints_faces(self, cell_type_new=cell_type)

    @wraps(add_midpoints_volumes)
    def add_midpoints_volumes(self, cell_type=None):
        return add_midpoints_volumes(self, cell_type_new=cell_type)
