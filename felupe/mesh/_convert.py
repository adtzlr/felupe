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

from ._helpers import mesh_or_data
from ._mesh import Mesh


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
    """Convert mesh to a given order (only order=0 and order=2
    from order=1 are supported)."""

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

    return points_edges, cells_edges, cell_type


@mesh_or_data
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

    return points_faces, cells_faces, cell_type


@mesh_or_data
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

    return points_volumes, cells_volumes, cell_type


@mesh_or_data
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

    return points_new, cells_new, cell_types_new[cell_type]


@mesh_or_data
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

    return points_new, cells_new, cell_types_new[cell_type]


@mesh_or_data
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

    return points_new, cells_new, cell_types_new[cell_type]
