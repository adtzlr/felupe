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
    def __init__(self, nodes, connectivity, etype=None):
        self.nodes = nodes
        self.connectivity = connectivity
        self.etype = etype

        self.update(self.connectivity)

    def update(self, connectivity):
        self.connectivity = connectivity

        # obtain dimensions
        self.nnodes, self.ndim = self.nodes.shape
        self.ndof = self.nodes.size
        self.nelements = self.connectivity.shape[0]

        # get number of elements per node
        nodes_in_conn, self.elements_per_node = np.unique(
            connectivity, return_counts=True
        )

        # check if there are nodes without element connectivity
        if self.nnodes != len(self.elements_per_node):
            self.node_has_element = np.isin(np.arange(self.nnodes), nodes_in_conn)
            # update "epn" ... elements per node
            epn = -np.ones(self.nnodes, dtype=int)
            epn[nodes_in_conn] = self.elements_per_node
            self.elements_per_node = epn

            self.nodes_without_elements = np.arange(self.nnodes)[~self.node_has_element]
            self.nodes_with_elements = np.arange(self.nnodes)[self.node_has_element]
        else:
            self.nodes_without_elements = np.array([], dtype=int)
            self.nodes_with_elements = np.arange(self.nnodes)

    def save(self, filename="mesh.vtk"):
        "Export mesh as VTK file."

        if self.etype is None:
            raise TypeError("Element type missing.")
        else:
            import meshio

        cells = {self.etype: self.connectivity}
        meshio.Mesh(self.nodes, cells).write(filename)


class Cube(Mesh):
    def __init__(self, a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
        self.a = a
        self.b = b
        self.n = n

        nodes, connectivity = cube_hexa(a, b, n)
        etype = "hexahedron"

        super().__init__(nodes, connectivity, etype)
        # self.edgenodes = 8


class Rectangle(Mesh):
    def __init__(self, a=(0, 0), b=(1, 1), n=(2, 2)):
        self.a = a
        self.b = b
        self.n = n

        nodes, connectivity = rectangle_quad(a, b, n)
        etype = "quad"

        super().__init__(nodes, connectivity, etype)
        # self.edgenodes = 4


class Line(Mesh):
    def __init__(self, a=0, b=1, n=2):
        self.a = a
        self.b = b
        self.n = n

        nodes, connectivity = line_line(a, b, n)

        etype = "line"

        super().__init__(nodes, connectivity, etype)
        # self.edgenodes = 2


class CubeQuadratic(Mesh):
    def __init__(self, a=(0, 0, 0), b=(1, 1, 1)):
        self.a = a
        self.b = b
        self.n = (3, 3, 3)

        nodes, connectivity = cube_hexa(a, b, n=self.n)
        etype = "hexahedron20"

        super().__init__(nodes, connectivity, etype)
        # self.edgenodes = 8

        self.nodes = self.nodes[
            [0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15]
        ]
        self.connectivity = np.arange(20).reshape(1, -1)
        self.update(self.connectivity)


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
            mask = np.logical_or(self.nodes[:, 0] > L0 / 2, self.nodes[:, 1] > B0 / 2)
            keep = np.arange(self.nnodes)[mask]
            select = np.array(
                [np.all(np.isin(conn, keep)) for conn in self.connectivity]
            )
            self.connectivity = self.connectivity[select]

        z = self.nodes.copy()
        z[:, 0] *= L / 2 * (1 + 2 * dL / L * self.nodes[:, 2] ** exponent)
        z[:, 1] *= B / 2 * (1 + 2 * dB / B * self.nodes[:, 2] ** exponent)
        z[:, 2] *= H / 2
        self.nodes = z
        self.update(self.connectivity)


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

        if phi == 360:
            sweep_nodes = True
        else:
            sweep_nodes = False

        nodes, connectivity = revolve(
            (N, C), n[1], -phi, axis=0, sweep_nodes=sweep_nodes
        )
        etype = "hexahedron"

        if align:
            nodes, connectivity = rotate(rotate((nodes, connectivity), 90, 1), 90, 2)

        super().__init__(nodes, connectivity, etype)
        # self.edgenodes = 8


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

        connectivity = np.hstack((vertices, edges, faces, volume)).reshape(1, -1)

        super().__init__(points, connectivity, etype="VTK_LAGRANGE_HEXAHEDRON")


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

        connectivity = np.hstack((vertices, edges, face)).reshape(1, -1)

        super().__init__(points, connectivity, etype="VTK_LAGRANGE_QUADRILATERAL")


# line, rectangle (based on line) and cube (based on rectangle) generators
# ------------------------------------------------------------------------


def line_line(a=0, b=1, n=2):
    "Line generator."
    nodes = np.linspace(a, b, n).reshape(-1, 1)
    connectivity = np.repeat(np.arange(n), 2)[1:-1].reshape(-1, 2)

    return nodes, connectivity


def rectangle_quad(a=(0, 0), b=(1, 1), n=(2, 2)):
    "Rectangle generator."
    dim = 2
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of nodes per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    line = line_line(a[0], b[0], n[0])

    nodes, connectivity = expand(line, n[-1], b[-1] - a[-1])
    nodes[:, -1] += a[-1]

    return nodes, connectivity


def cube_hexa(a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
    "Cube generator."
    dim = 3
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of nodes per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    rectangle = rectangle_quad(a[:-1], b[:-1], n[:-1])

    nodes, connectivity = expand(rectangle, n[-1], b[-1] - a[-1])
    nodes[:, -1] += a[-1]

    return nodes, connectivity


def expand(mesh, n=11, z=1):
    "Expand 1d line to 2d quad or 2d quad to 3d hexahedron mesh."

    if isinstance(mesh, Mesh):
        Nodes = mesh.nodes
        Connectivity = mesh.connectivity
        return_mesh = True
    else:
        Nodes, Connectivity = mesh
        return_mesh = False

    Dim = Nodes.shape[1]
    if Dim == 1:
        sl = slice(None, None, -1)
        etype = "quad"
    elif Dim == 2:
        sl = slice(None, None, None)
        etype = "hexahedron"
    else:
        raise ValueError("Expansion of a 3d mesh is not supported.")

    p = np.pad(Nodes, (0, 1))[:-1]
    zeros = np.zeros(Dim)
    nodes = np.vstack([p + np.array([*zeros, h]) for h in np.linspace(0, z, n)])

    c = [Connectivity + len(p) * a for a in np.arange(n)]
    connectivity = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    if return_mesh:
        return Mesh(nodes, connectivity, etype)
    else:
        return nodes, connectivity


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
        Nodes = mesh.nodes
        Connectivity = mesh.connectivity
        Etype = mesh.etype
        return_mesh = True
    else:
        Nodes, Connectivity = mesh
        return_mesh = False

    dim = Nodes.shape[1]

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.array(center)
    center = center.reshape(1, -1)

    nodes = (rotation_matrix(angle_deg, dim, axis) @ (Nodes - center).T).T + center

    if return_mesh:
        return Mesh(nodes, Connectivity, Etype)
    else:
        return nodes, Connectivity


def revolve(mesh, n=11, phi=180, axis=0, sweep_nodes=False, sweep_decimals=6):
    "Revolve 2d quad to 3d hexahedron mesh."

    if isinstance(mesh, Mesh):
        Nodes = mesh.nodes
        Connectivity = mesh.connectivity
        return_mesh = True
    else:
        Nodes, Connectivity = mesh
        return_mesh = False

    Dim = Nodes.shape[1]
    if Dim == 1:
        sl = slice(None, None, -1)
        etype = "quad"
    elif Dim == 2:
        sl = slice(None, None, None)
        etype = "hexahedron"
    else:
        raise ValueError("Revolution of a 3d mesh is not supported.")

    if abs(phi) > 360:
        raise ValueError("phi must be within |phi| <= 360 degree.")

    p = np.pad(Nodes, ((0, 0), (0, 1)))
    R = rotation_matrix

    nodes = np.vstack(
        [(R(alpha_deg, Dim + 1) @ p.T).T for alpha_deg in np.linspace(0, phi, n)]
    )

    c = [Connectivity + len(p) * a for a in np.arange(n)]

    # if abs(phi) == 360:
    #    nodes = nodes[: -len(p)]
    #    c[-1] = Connectivity

    connectivity = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    # WARNING: np.unique sorts the output!
    if sweep_nodes:
        nodes, connectivity = sweep((nodes, connectivity), decimals=sweep_decimals)

    if return_mesh:
        return Mesh(nodes, connectivity, etype)
    else:
        return nodes, connectivity


def sweep(mesh, decimals=None):
    "Sweep duplicated nodes and update connectivity."

    if isinstance(mesh, Mesh):
        Nodes = mesh.nodes
        Connectivity = mesh.connectivity
        etype = mesh.etype
        return_mesh = True
    else:
        Nodes, Connectivity = mesh
        return_mesh = False

    if decimals is None:
        Nodes_rounded = Nodes
    else:
        Nodes_rounded = np.round(Nodes, decimals)

    nodes, index, inverse, counts = np.unique(Nodes_rounded, True, True, True, axis=0)

    original = np.arange(len(Nodes))

    mask = inverse != original
    find = original[mask]
    replace = inverse[mask]

    connectivity = Connectivity.copy()

    for i, j in zip(find, replace):
        connectivity[Connectivity == i] = j

    if return_mesh:
        return Mesh(nodes, connectivity, etype)
    else:
        return nodes, connectivity


def convert(mesh, order=0, calc_nodes=False):
    "Convert mesh to a given order (currently only order=0 supported)."

    if order != 0:
        raise NotImplementedError("Unsupported order conversion.")

    if calc_nodes:
        nodes = np.stack(
            [np.mean(mesh.nodes[conn], axis=0) for conn in mesh.connectivity]
        )
    else:
        nodes = np.zeros((mesh.nelements, mesh.ndim), dtype=int)

    connectivity = np.arange(mesh.nelements).reshape(-1, 1)
    etype = "None"
    return Mesh(nodes, connectivity, etype)
