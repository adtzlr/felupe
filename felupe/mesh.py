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


class Mesh:
    def __init__(self, points, cells, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells)
        self.cell_type = cell_type

        self.update(self.cells)

    def update(self, cells):
        self.cells = cells

        # obtain dimensions
        self.npoints, self.ndim = self.points.shape
        self.ndof = self.points.size
        self.ncells = self.cells.shape[0]

        # get number of cells per point
        points_in_cell, self.cells_per_point = np.unique(cells, return_counts=True)

        # check if there are points without cells
        if self.npoints != len(self.cells_per_point):
            self.point_has_cell = np.isin(np.arange(self.npoints), points_in_cell)
            # update "cells_per_point" ... cells per point
            cells_per_point = -np.ones(self.npoints, dtype=int)
            cells_per_point[points_in_cell] = self.cells_per_point
            self.cells_per_point = cells_per_point

            self.points_without_cells = np.arange(self.npoints)[~self.point_has_cell]
            self.points_with_cells = np.arange(self.npoints)[self.point_has_cell]
        else:
            self.points_without_cells = np.array([], dtype=int)
            self.points_with_cells = np.arange(self.npoints)

    def save(self, filename="mesh.vtk"):
        "Export mesh as VTK file."

        if self.cell_type is None:
            raise TypeError("Cell type missing.")
        else:
            import meshio

        cells = {self.cell_type: self.cells}
        meshio.Mesh(self.points, cells).write(filename)


class Cube(Mesh):
    def __init__(self, a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
        self.a = a
        self.b = b
        self.n = n

        points, cells = cube_hexa(a, b, n)
        cell_type = "hexahedron"

        super().__init__(points, cells, cell_type)


class Rectangle(Mesh):
    def __init__(self, a=(0, 0), b=(1, 1), n=(2, 2)):
        self.a = a
        self.b = b
        self.n = n

        points, cells = rectangle_quad(a, b, n)
        cell_type = "quad"

        super().__init__(points, cells, cell_type)


class Line(Mesh):
    def __init__(self, a=0, b=1, n=2):
        self.a = a
        self.b = b
        self.n = n

        points, cells = line_line(a, b, n)

        cell_type = "line"

        super().__init__(points, cells, cell_type)


class CubeAdvanced(Cube):
    def __init__(
        self,
        n=5,
        L=1,
        B=1,
        H=1,
        dL=0,
        dB=0,
        exponent=4,
        symmetry=(False, False, False),
        L0=0,
        B0=0,
    ):

        a = -np.ones(3)
        symmetry = np.array(symmetry, dtype=bool)
        a[list(symmetry)] = 0
        super().__init__(a, (1, 1, 1), n)

        if L0 > 0 or B0 > 0:
            mask = np.logical_or(self.points[:, 0] > L0 / 2, self.points[:, 1] > B0 / 2)
            keep = np.arange(self.npoints)[mask]
            select = np.array([np.all(np.isin(cell, keep)) for cell in self.cells])
            self.cells = self.cells[select]

        z = self.points.copy()
        z[:, 0] *= L / 2 * (1 + 2 * dL / L * self.points[:, 2] ** exponent)
        z[:, 1] *= B / 2 * (1 + 2 * dB / B * self.points[:, 2] ** exponent)
        z[:, 2] *= H / 2
        self.points = z
        self.update(self.cells)


class CylinderAdvanced(Mesh):
    def __init__(
        self,
        D=10,
        H=1,
        n=(13, 13, 9),
        d=2,
        phi=180,
        dD=1,
        dd=1,
        k=4,
        align=True,
        symmetry=False,
    ):

        R = D / 2
        r = d / 2
        # rm = (R + r) / 2
        dr = R - r

        if symmetry:
            a = 0
        else:
            a = -1

        N, C = expand(line_line(a=a, b=1, n=n[2]), n[0], z=2)
        N[:, 1] -= 1

        bottom = N[:, 1] < 0
        top = N[:, 1] > 0
        Nb = N[bottom]
        Nt = N[top]

        Nb[:, 1] *= 1 + dd / dr * Nb[:, 0] ** k
        Nt[:, 1] *= 1 + dD / dr * Nt[:, 0] ** k

        N[bottom] = Nb
        N[top] = Nt

        N[:, 1] += 1
        N[:, 1] *= dr / 2
        N[:, 1] += r

        N[:, 0] *= H / 2
        N[:, 0] += H / 2

        points, cells = revolve((N, C), n[1], -phi, axis=0)
        cell_type = "hexahedron"

        if align:
            points, cells = rotate(rotate((points, cells), 90, 1), 90, 2)

        super().__init__(points, cells, cell_type)


class Cylinder(CylinderAdvanced):
    def __init__(self, D=2, H=1, n=(3, 9, 3), phi=360):

        super().__init__(D, H, n, d=0, phi=phi, dD=0, dd=0, k=4, align=True)


class CubeArbitraryOderHexahedron(Mesh):
    def __init__(self, a=(0, 0, 0), b=(1, 1, 1), order=2):
        zv, yv, xv = np.meshgrid(
            np.linspace(a[2], b[2], order + 1),
            np.linspace(a[1], b[1], order + 1),
            np.linspace(a[0], b[0], order + 1),
            indexing="ij",
        )

        points = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T

        # search vertices
        xmin = min(points[:, 0])
        ymin = min(points[:, 1])
        zmin = min(points[:, 2])

        xmax = max(points[:, 0])
        ymax = max(points[:, 1])
        zmax = max(points[:, 2])

        def search_vertice(p, x, y, z):
            return np.where(
                np.logical_and(np.logical_and(p[:, 0] == x, p[:, 1] == y), p[:, 2] == z)
            )[0][0]

        def search_edge(p, a, b, x, y):
            return np.where(np.logical_and(p[:, a] == x, p[:, b] == y))[0][1:-1]

        def search_face(p, a, x, vertices, edges):
            face = np.where(points[:, a] == x)[0]
            mask = np.zeros_like(p[:, 0], dtype=bool)
            mask[face] = 1
            mask[np.hstack((vertices, edges))] = 0
            return np.arange(len(p[:, 0]))[mask]

        v1 = search_vertice(points, xmin, ymin, zmin)
        v2 = search_vertice(points, xmax, ymin, zmin)
        v3 = search_vertice(points, xmax, ymax, zmin)
        v4 = search_vertice(points, xmin, ymax, zmin)

        v5 = search_vertice(points, xmin, ymin, zmax)
        v6 = search_vertice(points, xmax, ymin, zmax)
        v7 = search_vertice(points, xmax, ymax, zmax)
        v8 = search_vertice(points, xmin, ymax, zmax)

        vertices = [v1, v2, v3, v4, v5, v6, v7, v8]

        mask1 = np.ones_like(points[:, 0], dtype=bool)
        mask1[vertices] = 0
        # points_no_verts = points[mask1]

        e1 = search_edge(points, 1, 2, ymin, zmin)
        e2 = search_edge(points, 0, 2, xmax, zmin)
        e3 = search_edge(points, 1, 2, ymax, zmin)
        e4 = search_edge(points, 0, 2, xmin, zmin)

        e5 = search_edge(points, 1, 2, ymin, zmax)
        e6 = search_edge(points, 0, 2, xmax, zmax)
        e7 = search_edge(points, 1, 2, ymax, zmax)
        e8 = search_edge(points, 0, 2, xmin, zmax)

        e9 = search_edge(points, 0, 1, xmin, ymin)
        e10 = search_edge(points, 0, 1, xmax, ymin)
        e11 = search_edge(points, 0, 1, xmin, ymax)
        e12 = search_edge(points, 0, 1, xmax, ymax)

        edges = np.hstack((e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12))

        mask2 = np.ones_like(points[:, 0], dtype=bool)
        mask2[np.hstack((vertices, edges))] = 0
        # points_no_verts_edges = points[mask2]

        f1 = search_face(points, 0, xmin, vertices, edges)
        f2 = search_face(points, 0, xmax, vertices, edges)
        f3 = search_face(points, 1, ymin, vertices, edges)
        f4 = search_face(points, 1, ymax, vertices, edges)
        f5 = search_face(points, 2, zmin, vertices, edges)
        f6 = search_face(points, 2, zmax, vertices, edges)

        faces = np.hstack((f1, f2, f3, f4, f5, f6))

        mask3 = np.ones_like(points[:, 0], dtype=bool)
        mask3[np.hstack((vertices, edges, faces))] = 0
        volume = np.arange(len(points))[mask3]

        cells = np.hstack((vertices, edges, faces, volume)).reshape(1, -1)

        super().__init__(points, cells, cell_type="VTK_LAGRANGE_HEXAHEDRON")


class RectangleArbitraryOderQuad(Mesh):
    def __init__(self, a=(0, 0), b=(1, 1), order=2):
        yv, xv = np.meshgrid(
            np.linspace(a[1], b[1], order + 1),
            np.linspace(a[0], b[0], order + 1),
            indexing="ij",
        )

        points = np.vstack((xv.flatten(), yv.flatten())).T

        # search vertices
        xmin = min(points[:, 0])
        ymin = min(points[:, 1])

        xmax = max(points[:, 0])
        ymax = max(points[:, 1])

        def search_vertice(p, x, y):
            return np.where(np.logical_and(p[:, 0] == x, p[:, 1] == y))[0][0]

        def search_edge(p, a, x):
            return np.where(p[:, a] == x)[0][1:-1]

        def search_face(p, a, x, vertices, edges):
            face = np.where(points[:, a] == x)[0]
            mask = np.zeros_like(p[:, 0], dtype=bool)
            mask[face] = 1
            mask[np.hstack((vertices, edges))] = 0
            return np.arange(len(p[:, 0]))[mask]

        v1 = search_vertice(points, xmin, ymin)
        v2 = search_vertice(points, xmax, ymin)
        v3 = search_vertice(points, xmax, ymax)
        v4 = search_vertice(points, xmin, ymax)

        vertices = [v1, v2, v3, v4]

        mask1 = np.ones_like(points[:, 0], dtype=bool)
        mask1[vertices] = 0
        # points_no_verts = points[mask1]

        e1 = search_edge(points, 1, ymin)
        e2 = search_edge(points, 0, xmax)
        e3 = search_edge(points, 1, ymax)
        e4 = search_edge(points, 0, xmin)

        edges = np.hstack((e1, e2, e3, e4))

        mask2 = np.ones_like(points[:, 0], dtype=bool)
        mask2[np.hstack((vertices, edges))] = 0
        # points_no_verts_edges = points[mask2]

        face = np.arange(len(points))[mask2]

        cells = np.hstack((vertices, edges, face)).reshape(1, -1)

        super().__init__(points, cells, cell_type="VTK_LAGRANGE_QUADRILATERAL")


# line, rectangle (based on line) and cube (based on rectangle) generators
# ------------------------------------------------------------------------


def line_line(a=0, b=1, n=2):
    "Line generator."
    points = np.linspace(a, b, n).reshape(-1, 1)
    cells = np.repeat(np.arange(n), 2)[1:-1].reshape(-1, 2)

    return points, cells


def rectangle_quad(a=(0, 0), b=(1, 1), n=(2, 2)):
    "Rectangle generator."
    dim = 2
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of points per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    line = line_line(a[0], b[0], n[0])

    points, cells = expand(line, n[-1], b[-1] - a[-1])
    points[:, -1] += a[-1]

    return points, cells


def cube_hexa(a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
    "Cube generator."
    dim = 3
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of points per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    rectangle = rectangle_quad(a[:-1], b[:-1], n[:-1])

    points, cells = expand(rectangle, n[-1], b[-1] - a[-1])
    points[:, -1] += a[-1]

    return points, cells


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
    """Convert mesh to a given order (currently only order=0 and order=2
    from order=1 are supported)."""

    if mesh.cell_type not in ["triangle", "tetra", "quad", "hexahedron"]:
        raise NotImplementedError("Cell type not supported for conversion.")

    if order == 0:

        if calc_points:
            points = np.stack(
                [np.mean(mesh.points[cell], axis=0) for cell in mesh.cells]
            )
        else:
            points = np.zeros((mesh.ncells, mesh.ndim), dtype=int)

        cells = np.arange(mesh.ncells).reshape(-1, 1)
        cell_type = mesh.cell_type

    elif order == 1:

        points = mesh.points
        cells = mesh.cells
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


def fix(points, cells, cell_type):
    "Fixes cells array of tetrahedrals to ensure cell volume > 0."

    if cell_type == "tetra":

        # extract point 0 of all cells and reshape and repeat
        p0 = points[cells][:, 0].reshape(len(cells), 1, 3)

        # calculate vertice "i" as v[:, i]
        v = points[cells] - np.repeat(p0, 4, axis=1)

        # calculate volume by the triple product
        volume = np.einsum("...i,...i->...", v[:, 3], np.cross(v[:, 1], v[:, 2]))

        # permute point entries for cells with volume < 0
        cells[volume < 0] = cells[volume < 0][:, [0, 2, 1, 3]]

    return points, cells


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
