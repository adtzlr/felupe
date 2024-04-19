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

import numpy as np

from ._helpers import mesh_or_data


@mesh_or_data
def convert(
    points,
    cells,
    cell_type,
    order=0,
    calc_points=False,
    calc_midfaces=False,
    calc_midvolumes=False,
):
    """Convert a mesh to a given order. Only conversions to ``order=0`` and ``order=2``
    are supported. This function supports meshes with cell types ``"triangle"``,
    ``"tetra"``, ``"quad"`` and ``"hexahedron"``.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type. Must be one of
        ``"triangle"``, ``"tetra"``, ``"quad"`` or ``"hexahedron"``.
    order : int, optional
        The order of the converted mesh (default is 0). If 0, the points-array will be
        of shape ``(ncells, dim)``. If 0 and ``calc_points`` is True, the mean of all
        points per cell is evaluated. If 0 and ``calc_points`` is False, the points
        array is filled with zeros. If 2, at least midpoints on cell edges are added to
        the mesh. If 2 and ``calc_midfaces`` is True, midpoints on cell faces are also
        added. If 2 and ``calc_midvolumes`` is True, midpoints on cell volumes are also
        added. Raises an error if not 0 or 2.
    calc_points : bool, optional
        Flag to return the mean of all points per cell if ``order=0`` (default is
        False). If False, the points-array is filled with zeros.
    calc_midfaces : bool, optional
        Flag to add midpoints on cell faces if ``order=2`` (default is False).
    calc_midvolumes : bool, optional
        Flag to add midpoints on cell volumes if ``order=2`` (default is False).

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : list or ndarray
        Converted cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    Convert a mesh of hexahedrons to quadratic hexahedrons by inserting midpoints on
    the cell edges.

    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle(n=6)
       >>> mesh2 = fem.mesh.convert(mesh, order=2)
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

    ncells = len(cells)
    dim = points.shape[1]

    if cell_type not in ["triangle", "tetra", "quad", "hexahedron"]:
        raise NotImplementedError("Cell type not supported for conversion.")

    if order == 0:
        if calc_points:
            points_new = np.stack([np.mean(points[cell], axis=0) for cell in cells])
        else:
            points_new = np.zeros((ncells, dim), dtype=int)

        cells_new = np.arange(ncells).reshape(-1, 1)
        cell_type_new = cell_type

    elif order == 2:
        points_new, cells_new, cell_type_new = add_midpoints_edges(
            points, cells, cell_type
        )

        if calc_midfaces:
            points_new, cells_new, cell_type_new = add_midpoints_faces(
                points_new, cells_new, cell_type_new
            )

        if calc_midvolumes:
            points_new, cells_new, cell_type_new = add_midpoints_volumes(
                points_new, cells_new, cell_type_new
            )

    else:
        raise NotImplementedError("Unsupported order conversion.")

    return points_new, cells_new, cell_type_new


@mesh_or_data
def collect_edges(points, cells, cell_type):
    """Collect all unique edges, calculate and return midpoints on edges as well as the
    additional cells array.

    See Also
    --------
    felupe.mesh.add_midpoints_edges : Add midpoints on cell edges for given points and
        cells and update cell_type accordingly.
    felupe.Mesh.add_midpoints_edges : Add midpoints on cell edges for given points and
        cells and update cell_type accordingly.
    """

    supported_cell_types = ["triangle", "tetra", "quad", "hexahedron"]

    if cell_type not in supported_cell_types:
        raise TypeError("Cell type not implemented.")

    number_of_edges = {"triangle": 3, "tetra": 6, "quad": 4, "hexahedron": 12}

    if cell_type in ["triangle", "tetra"]:
        # k-th edge is (i[k], j[k])
        i = [0, 1, 2, 3, 3, 3][: number_of_edges[cell_type]]
        j = [1, 2, 0, 0, 1, 2][: number_of_edges[cell_type]]

    elif cell_type in ["quad", "hexahedron"]:
        # k-th edge is (i[k], j[k])
        i = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3][: number_of_edges[cell_type]]
        j = [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7][: number_of_edges[cell_type]]

    edges_to_stack = cells[:, i], cells[:, j]

    # sort points of edges
    edges = np.sort(np.dstack(edges_to_stack).reshape(-1, 2), axis=1)

    # obtain unique edges and inverse mapping
    edges_unique, inverse = np.unique(edges, False, True, False, 0)

    # calculate midpoints on edges as mean
    points_edges = np.mean(points[edges_unique.T], axis=0)

    # create the additionals cells array
    cells_edges = inverse.reshape(len(cells), -1)

    return points_edges, cells_edges, cell_type


@mesh_or_data
def collect_faces(points, cells, cell_type):
    """Collect all unique faces, calculate and return midpoints on faces as well as the
    additional cells array.

    See Also
    --------
    felupe.mesh.add_midpoints_faces : Add midpoints on cell faces for given points and
        cells and update cell_type accordingly.
    felupe.Mesh.add_midpoints_faces : Add midpoints on cell faces for given points and
        cells and update cell_type accordingly.
    """

    supported_cell_types = [
        "triangle",
        "triangle6",
        "tetra",
        "tetra10",
        "quad",
        "quad8",
        "hexahedron",
        "hexahedron20",
    ]

    if cell_type not in supported_cell_types:
        raise TypeError("Cell type not implemented.")

    if "triangle" in cell_type:
        # k-th face is (i[k], j[k], k[k])
        i = [0]
        j = [1]
        k = [2]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k]

    if "tetra" in cell_type:
        # k-th face is (i[k], j[k], k[k])
        # ordering?
        i = [0, 0, 0, 1]
        j = [1, 1, 2, 2]
        k = [2, 3, 3, 3]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k]

    elif "quad" in cell_type:
        # k-th edge is (i[k], j[k], k[k], m[k])
        i = [0]
        j = [1]
        k = [2]
        m = [3]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k], cells[:, m]

    elif "hexahedron" in cell_type:
        # k-th edge is (i[k], j[k], k[k], l[k])
        i = [0, 1, 1, 2, 0, 4]
        j = [3, 2, 0, 3, 1, 5]
        k = [7, 6, 4, 7, 2, 6]
        m = [4, 5, 5, 6, 3, 7]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k], cells[:, m]

    # sort points of edges
    faces = np.sort(np.dstack(faces_to_stack).reshape(-1, len(faces_to_stack)), axis=1)

    # obtain unique edges and inverse mapping
    faces_unique, inverse = np.unique(faces, False, True, False, 0)

    # calculate midpoints on edges as mean
    points_faces = np.mean(points[faces_unique.T], axis=0)

    # create the additionals cells array
    cells_faces = inverse.reshape(len(cells), -1)

    return points_faces, cells_faces, cell_type


@mesh_or_data
def collect_volumes(points, cells, cell_type):
    """Collect all volumes, calculate and return midpoints on volumes as well as the
    additional cells array.

    See Also
    --------
    felupe.mesh.add_midpoints_volumes : Add midpoints on cell volumes for given points
        and cells and update cell_type accordingly.
    felupe.Mesh.add_midpoints_volumes : Add midpoints on cell volumes for given points
        and cells and update cell_type accordingly.

    """

    supported_cell_types = [
        "tetra",
        "tetra10",
        "tetra14",
        "hexahedron",
        "hexahedron20",
        "hexahedron26",
    ]

    if cell_type not in supported_cell_types:
        raise TypeError("Cell type not implemented.")

    if "tetra" in cell_type:
        number_of_vertices = 3

    elif "hexahedron" in cell_type:
        number_of_vertices = 8

    if cell_type in supported_cell_types:
        points_volumes = np.mean(points[cells][:, :number_of_vertices, :], axis=1)
        cells_volumes = np.arange(cells.shape[0]).reshape(-1, 1)

    return points_volumes, cells_volumes, cell_type


@mesh_or_data
def add_midpoints_edges(points, cells, cell_type, cell_type_new=None):
    """Add midpoints on edges for given points and cells and update cell_type
    accordingly.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    cell_type_new: str or None, optional
        A string in VTK-convention that specifies the new cell type (default is None).
        If None, the cell type is chosen automatically.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    Convert a mesh of hexahedrons to quadratic hexahedrons by inserting midpoints on
    the cell edges.

    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle(n=6)
       >>> mesh_with_midpoints_edges = fem.mesh.add_midpoints_edges(mesh)
       >>>
       >>> plotter = mesh_with_midpoints_edges.plot(
       ...     plotter=mesh.plot(), style="points", color="black"
       ... )
       >>> plotter.show()

    >>> mesh_with_midpoints_edges
    <felupe Mesh object>
      Number of points: 96
      Number of cells:
        quad8: 25

    See Also
    --------
    felupe.Mesh.add_midpoints_edges : Add midpoints on edges for given points and cells
        and update cell_type accordingly.

    """

    # collect edges
    points_edges, cells_edges, _ = collect_edges(
        points,
        cells,
        cell_type,
    )

    # add offset to point index for edge-midpoints
    # in additional cells array
    cells_edges += len(points)

    # vertical stack of points and horizontal stack of edges
    points_new = np.vstack((points, points_edges))
    cells_new = np.hstack((cells, cells_edges))

    if cell_type_new is None:
        cell_type_new = {
            "triangle": "triangle6",
            "tetra": "tetra10",
            "quad": "quad8",
            "hexahedron": "hexahedron20",
        }[cell_type]

    return points_new, cells_new, cell_type_new


@mesh_or_data
def add_midpoints_faces(points, cells, cell_type, cell_type_new=None):
    """Add midpoints on faces for given points and cells and update cell_type
    accordingly.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    cell_type_new: str or None, optional
        A string in VTK-convention that specifies the new cell type (default is None).
        If None, the cell type is chosen automatically.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle(n=6)
       >>> mesh_with_midpoints_faces = fem.mesh.add_midpoints_faces(
       ...     mesh, cell_type_new="quad"
       ... )
       >>>
       >>> plotter = mesh_with_midpoints_faces.plot(
       ...     plotter=mesh.plot(), style="points", color="black"
       ... )
       >>> plotter.show()

    >>> mesh_with_midpoints_faces
    <felupe Mesh object>
      Number of points: 61
      Number of cells:
        quad: 25

    See Also
    --------
    felupe.Mesh.add_midpoints_faces : Add midpoints on faces for given points and cells
        and update cell_type accordingly.
    """

    # collect faces
    points_faces, cells_faces, _ = collect_faces(
        points,
        cells,
        cell_type,
    )

    # add offset to point index for faces-midpoints
    # in additional cells array
    cells_faces += len(points)

    # vertical stack of points and horizontal stack of edges
    points_new = np.vstack((points, points_faces))
    cells_new = np.hstack((cells, cells_faces))

    if cell_type_new is None:
        cell_type_new = {
            None: None,
            "triangle": None,
            "triangle6": "triangle7",
            "tetra10": "tetra14",
            "quad": None,
            "quad8": "quad9",
            "hexahedron": None,
            "hexahedron20": "hexahedron26",
        }[cell_type]

    return points_new, cells_new, cell_type_new


@mesh_or_data
def add_midpoints_volumes(points, cells, cell_type, cell_type_new=None):
    """Add midpoints on volumes for given points and cells and update cell_type
    accordingly.

    Parameters
    ----------
    points : list or ndarray
        Original point coordinates.
    cells : list or ndarray
        Original point-connectivity of cells.
    cell_type : str
        A string in VTK-convention that specifies the cell type.
    cell_type_new: str or None, optional
        A string in VTK-convention that specifies the new cell type (default is None).
        If None, the cell type is chosen automatically.

    Returns
    -------
    points : ndarray
        Modified point coordinates.
    cells : ndarray
        Modified point-connectivity of cells.
    cell_type : str or None
        A string in VTK-convention that specifies the cell type.

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
    felupe.Mesh.add_midpoints_volumes : Add midpoints on volumes for given points and
        cells and update cell_type accordingly.
    """

    # collect volumes
    points_volumes, cells_volumes, _ = collect_volumes(
        points,
        cells,
        cell_type,
    )

    # add offset to point index for volumes-midpoints
    # in additional cells array
    cells_volumes += len(points)

    # vertical stack of points and horizontal stack of edges
    points_new = np.vstack((points, points_volumes))
    cells_new = np.hstack((cells, cells_volumes))

    if cell_type_new is None:
        cell_type_new = {
            None: None,
            "tetra": None,
            "tetra10": None,
            "tetra14": "tetra15",
            "hexahedron": None,
            "hexahedron20": None,
            "hexahedron26": "hexahedron27",
        }[cell_type]

    return points_new, cells_new, cell_type_new
