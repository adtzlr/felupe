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
from ._mesh import Mesh


def expand(mesh, n=11, z=1):
    "Expand 1d line to 2d quad or 2d quad to 3d hexahedron mesh."

    if isinstance(mesh, Mesh):
        points = mesh.points
        cells = mesh.cells
        return_mesh = True
    else:
        points, cells = mesh
        return_mesh = False

    dim = points.shape[1]
    if dim == 1:
        sl = slice(None, None, -1)
        cell_type = "quad"
    elif dim == 2:
        sl = slice(None, None, None)
        cell_type = "hexahedron"
    else:
        raise ValueError("Expansion of a 3d mesh is not supported.")

    p = np.pad(points, (0, 1))[:-1]
    zeros = np.zeros(dim)
    points_new = np.vstack([p + np.array([*zeros, h]) for h in np.linspace(0, z, n)])

    c = [cells + len(p) * a for a in np.arange(n)]
    cells_new = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    if return_mesh:
        return Mesh(points_new, cells_new, cell_type)
    else:
        return points_new, cells_new


def rotation_matrix(alpha_deg, dim=3, axis=0):
    "2d or 3d rotation matrix around specified axis."
    a = np.deg2rad(alpha_deg)
    rotation_matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if dim == 3:
        # rotation_matrix = np.pad(rotation_matrix, (1, 0))
        # rotation_matrix[0, 0] = 1
        rotation_matrix = np.insert(rotation_matrix, [axis], np.zeros((1, 2)), axis=0)
        rotation_matrix = np.insert(rotation_matrix, [axis], np.zeros((3, 1)), axis=1)
        rotation_matrix[axis, axis] = 1

    return rotation_matrix


def rotate(mesh, angle_deg, axis, center=None):
    "Rotate mesh."

    if isinstance(mesh, Mesh):
        points = mesh.points
        cells = mesh.cells
        cell_type = mesh.cell_type
        return_mesh = True
    else:
        points, cells = mesh
        return_mesh = False

    dim = points.shape[1]

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.array(center)
    center = center.reshape(1, -1)

    points_new = (
        rotation_matrix(angle_deg, dim, axis) @ (points - center).T
    ).T + center

    if return_mesh:
        return Mesh(points_new, cells, cell_type)
    else:
        return points_new, cells


def revolve(mesh, n=11, phi=180, axis=0):
    "Revolve 2d quad to 3d hexahedron mesh."

    if isinstance(mesh, Mesh):
        points = mesh.points
        cells = mesh.cells
        return_mesh = True
    else:
        points, cells = mesh
        return_mesh = False

    dim = points.shape[1]
    if dim == 1:
        sl = slice(None, None, -1)
        cell_type_new = "quad"
    elif dim == 2:
        sl = slice(None, None, None)
        cell_type_new = "hexahedron"
    else:
        raise ValueError("Revolution of a 3d mesh is not supported.")

    if abs(phi) > 360:
        raise ValueError("phi must be within |phi| <= 360 degree.")

    p = np.pad(points, ((0, 0), (0, 1)))
    R = rotation_matrix

    points_new = np.vstack(
        [(R(angle, dim + 1) @ p.T).T for angle in np.linspace(0, phi, n)]
    )

    c = [cells + len(p) * a for a in np.arange(n)]

    if phi == 360:
        c[-1] = c[0]
        points_new = points_new[: len(points_new) - len(points)]

    cells_new = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    if return_mesh:
        return Mesh(points_new, cells_new, cell_type_new)
    else:
        return points_new, cells_new


def sweep(mesh, decimals=None):
    """Sweep duplicated points and update cells.
    WARNING: This function sorts points!!!"""

    if isinstance(mesh, Mesh):
        points = mesh.points
        cells = mesh.cells
        cell_type = mesh.cell_type
        return_mesh = True
    else:
        points, cells = mesh
        return_mesh = False

    if decimals is None:
        points_rounded = points
    else:
        points_rounded = np.round(points, decimals)

    points_new, index, inverse, counts = np.unique(
        points_rounded, True, True, True, axis=0
    )

    original = np.arange(len(points))

    mask = inverse != original
    find = original[mask]
    replace = inverse[mask]

    cells_new = cells.copy()

    for i, j in zip(find, replace):
        cells_new[cells == i] = j

    if return_mesh:
        return Mesh(points_new, cells_new, cell_type)
    else:
        return points_new, cells_new


def convert(
    mesh, order=0, calc_points=False, calc_midfaces=False, calc_midvolumes=False
):
    """Convert mesh to a given order (only order=0 and order=2
    from order=1 are supported)."""

    if mesh.cell_type not in ["triangle", "tetra", "quad", "hexahedron"]:
        raise NotImplementedError("Cell type not supported for conversion.")

    if order == 0:

        if calc_points:
            points = np.stack(
                [np.mean(mesh.points[cell], axis=0) for cell in mesh.cells]
            )
        else:
            points = np.zeros((mesh.ncells, mesh.dim), dtype=int)

        cells = np.arange(mesh.ncells).reshape(-1, 1)
        cell_type = mesh.cell_type

    elif order == 2:

        points, cells, cell_type = add_midpoints_edges(
            mesh.points, mesh.cells, mesh.cell_type
        )

        if calc_midfaces:
            points, cells, cell_type = add_midpoints_faces(points, cells, cell_type)

        if calc_midvolumes:
            points, cells, cell_type = add_midpoints_volumes(points, cells, cell_type)

    else:
        raise NotImplementedError("Unsupported order conversion.")

    return Mesh(points, cells, cell_type)


def collect_edges(points, cells, cell_type):
    """ "Collect all unique edges,
    calculate and return midpoints on edges as well as the additional
    cells array."""

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

    return points_edges, cells_edges


def collect_faces(points, cells, cell_type):
    """ "Collect all unique faces,
    calculate and return midpoints on faces as well as the additional
    cells array."""

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
        i = [
            0,
        ]
        j = [
            1,
        ]
        k = [
            2,
        ]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k]

    if "tetra" in cell_type:
        # k-th face is (i[k], j[k], k[k])
        # ordering?
        i = [0, 0, 0, 1]
        j = [1, 1, 2, 2]
        k = [2, 3, 3, 3]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k]

    elif "quad" in cell_type:
        # k-th edge is (i[k], j[k], k[k], l[k])
        i = [
            0,
        ]
        j = [
            1,
        ]
        k = [
            2,
        ]
        l = [
            3,
        ]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k], cells[:, l]

    elif "hexahedron" in cell_type:
        # k-th edge is (i[k], j[k], k[k], l[k])
        i = [0, 1, 1, 2, 0, 4]
        j = [3, 2, 0, 3, 1, 5]
        k = [7, 6, 4, 7, 2, 6]
        l = [4, 5, 5, 6, 3, 7]

        faces_to_stack = cells[:, i], cells[:, j], cells[:, k], cells[:, l]

    # sort points of edges
    faces = np.sort(np.dstack(faces_to_stack).reshape(-1, len(faces_to_stack)), axis=1)

    # obtain unique edges and inverse mapping
    faces_unique, inverse = np.unique(faces, False, True, False, 0)

    # calculate midpoints on edges as mean
    points_faces = np.mean(points[faces_unique.T], axis=0)

    # create the additionals cells array
    cells_faces = inverse.reshape(len(cells), -1)

    return points_faces, cells_faces


def collect_volumes(points, cells, cell_type):
    """ "Collect all volumes,
    calculate and return midpoints on volumes as well as the additional
    cells array."""

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

    return points_volumes, cells_volumes


def add_midpoints_edges(points, cells, cell_type):
    """ "Add midpoints on edges for given points and cells
    and update cell_type accordingly."""

    cell_types_new = {
        "triangle": "triangle6",
        "tetra": "tetra10",
        "quad": "quad8",
        "hexahedron": "hexahedron20",
    }

    # collect edges
    points_edges, cells_edges = collect_edges(
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

    return points_new, cells_new, cell_types_new[cell_type]


def add_midpoints_faces(points, cells, cell_type):
    """ "Add midpoints on faces for given points and cells
    and update cell_type accordingly."""

    cell_types_new = {
        None: None,
        "triangle": None,
        "triangle6": "triangle7",
        "tetra10": "tetra14",
        "quad": None,
        "quad8": "quad9",
        "hexahedron": None,
        "hexahedron20": "hexahedron26",
    }

    # collect faces
    points_faces, cells_faces = collect_faces(
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

    return points_new, cells_new, cell_types_new[cell_type]


def add_midpoints_volumes(points, cells, cell_type):
    """ "Add midpoints on volumes for given points and cells
    and update cell_type accordingly."""

    cell_types_new = {
        None: None,
        "tetra": None,
        "tetra10": None,
        "tetra14": "tetra15",
        "hexahedron": None,
        "hexahedron20": None,
        "hexahedron26": "hexahedron27",
    }

    # collect volumes
    points_volumes, cells_volumes = collect_volumes(
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

    return points_new, cells_new, cell_types_new[cell_type]
