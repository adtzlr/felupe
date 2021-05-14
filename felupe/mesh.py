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

import meshio

import meshzoo

from copy import deepcopy


def convert(mesh, order=0, calc_nodes=False):
    "Convert mesh to a given order (only order=0 supported)."

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
        self.edgenodes = 8


class Rectangle(Mesh):
    def __init__(self, a=(0, 0), b=(1, 1), n=(2, 2)):
        self.a = a
        self.b = b
        self.n = n

        nodes, connectivity = rectangle_quad(a, b, n)
        etype = "quad"

        super().__init__(nodes, connectivity, etype)
        self.edgenodes = 4


class Line(Mesh):
    def __init__(self, a=0, b=1, n=2):
        self.a = a
        self.b = b
        self.n = n

        nodes = np.linspace(a, b, n)
        connectivity = np.repeat(np.arange(n), 2)[1:-1].reshape(-1, 2)

        etype = "line"

        super().__init__(nodes, connectivity, etype)
        self.edgenodes = 2


class CubeQuadratic(Mesh):
    def __init__(self, a=(0, 0, 0), b=(1, 1, 1)):
        self.a = a
        self.b = b
        self.n = (3, 3, 3)

        nodes, connectivity = cube_hexa(a, b, n=self.n)
        etype = "hexahedron"

        super().__init__(nodes, connectivity, etype)
        self.edgenodes = 8

        self.nodes = self.nodes[
            [0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15]
        ]
        self.connectivity = np.arange(20).reshape(1, -1)
        self.update(self.connectivity)


class ScaledCube(Cube):
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


class CylinderOld(Cube):
    def __init__(
        self,
        a=(-1, -1, -1),
        b=(1, 1, 1),
        n=5,
        D=1,
        H=1,
        dD=0,
        exponent=4,
        symmetry=(False, False, False),
        switch=0.5,
    ):

        a = np.array(a)
        a[symmetry] = 0
        super().__init__(a, b, n)

        z = self.nodes.copy()

        r = np.sqrt(self.nodes[:, 0] ** 2 + self.nodes[:, 1] ** 2)
        mask = np.logical_or(
            abs(self.nodes[:, 0]) > switch, abs(self.nodes[:, 1]) > switch
        )
        r[~mask] = (r[~mask] - 2 / 3) / r[~mask].max() * r[~mask] + 2 / 3
        z[:, 0] /= 0.05 + r
        z[:, 1] /= 0.05 + r
        z[:, 0] *= D / 2 * (1 + 2 * dD / D * self.nodes[:, 2] ** exponent)
        z[:, 1] *= D / 2 * (1 + 2 * dD / D * self.nodes[:, 2] ** exponent)
        z[:, 2] *= H / 2

        self.nodes = z
        self.update(self.connectivity)


class HollowCylinder(Mesh):
    def __init__(self, D=10, H=1, n=(13, 13, 9), d=2, phi=180, dD=1, dd=1, k=4):

        R = D / 2
        r = d / 2
        # rm = (R + r) / 2
        dr = R - r

        N, C = expand(line_line(a=-1, b=1, n=n[0]), n[2], z=2)
        N[:, 1] -= 1

        left = N[:, 0] < 0
        right = N[:, 0] > 0
        Nl = N[left]
        Nr = N[right]

        Nl[:, 0] *= 1 + dd / dr * Nl[:, 1] ** k
        Nr[:, 0] *= 1 + dD / dr * Nr[:, 1] ** k

        N[left] = Nl
        N[right] = Nr

        N[:, 0] += 1
        N[:, 0] *= dr / 2
        N[:, 0] += r

        N[:, 1] *= H / 2
        N[:, 1] += H / 2

        nodes, connectivity = revolve((N, C), n[1], -phi, axis=1)
        etype = "hexahedron"

        super().__init__(nodes, connectivity, etype)
        self.edgenodes = 8


class Cylinder(HollowCylinder):
    def __init__(self, D=2, H=1, n=(13, 25, 2), phi=360):

        super().__init__(D, H, n, d=0, phi=phi, dD=0, dd=0, k=4)


def cube_hexa(a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):

    dim = 3
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of nodes per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    rectangle = rectangle_quad(a[:-1], b[:-1], n[:-1])

    nodes, connectivity = expand(rectangle, n[-1], b[-1] - a[-1])
    nodes[:, -1] += a[-1]

    return nodes, connectivity


def rectangle_quad(a=(0, 0), b=(1, 1), n=(2, 2)):

    dim = 2
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of nodes per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    line = line_line(a[0], b[0], n[0])

    nodes, connectivity = expand(line, n[-1], b[-1] - a[-1])
    nodes[:, -1] += a[-1]

    return nodes, connectivity


def line_line(a=0, b=1, n=2):

    nodes = np.linspace(a, b, n).reshape(-1, 1)
    connectivity = np.repeat(np.arange(n), 2)[1:-1].reshape(-1, 2)

    return nodes, connectivity


def expand(mesh, n=11, z=1):
    "Expand 2d quad to 3d hexahedron mesh."

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


def revolve(mesh, n=11, phi=180, axis=0):
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

    def R(alpha_deg, dim=3):
        a = np.deg2rad(alpha_deg)
        rotation_matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        if dim == 3:
            # rotation_matrix = np.pad(rotation_matrix, (1, 0))
            # rotation_matrix[0, 0] = 1
            rotation_matrix = np.insert(
                rotation_matrix, [axis], np.zeros((1, 2)), axis=0
            )
            rotation_matrix = np.insert(
                rotation_matrix, [axis], np.zeros((3, 1)), axis=1
            )
            rotation_matrix[axis, axis] = 1

        return rotation_matrix

    p = np.pad(Nodes, (0, 1))[:-1]

    nodes = np.vstack(
        [(R(alpha_deg, Dim + 1) @ p.T).T for alpha_deg in np.linspace(0, phi, n)]
    )

    c = [Connectivity + len(p) * a for a in np.arange(n)]

    if abs(phi) == 360:
        nodes = nodes[: -len(p)]
        c[-1] = Connectivity

    connectivity = np.vstack([np.hstack((a, b[:, sl])) for a, b in zip(c[:-1], c[1:])])

    nodes, connectivity = sweep((nodes, connectivity), decimals=6)

    if return_mesh:
        return Mesh(nodes, connectivity, etype)
    else:
        return nodes, connectivity


def sweep(mesh, decimals=6):

    if isinstance(mesh, Mesh):
        Nodes = mesh.nodes
        Connectivity = mesh.connectivity
        etype = mesh.etype
        return_mesh = True
    else:
        Nodes, Connectivity = mesh
        return_mesh = False

    nodes, index, inverse, counts = np.unique(
        np.round(Nodes, decimals), True, True, True, axis=0
    )

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
