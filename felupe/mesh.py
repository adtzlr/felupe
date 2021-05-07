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
import meshzoo


class Mesh:
    def __init__(self, nodes, connectivity, etype):
        self.nodes = nodes
        self.connectivity = connectivity
        self.etype = etype

        self.update(self.connectivity)

    def update(self, connectivity):
        self.connectivity = connectivity
        self.nnodes, self.ndim = self.nodes.shape
        self.ndof = self.nodes.size
        self.nelements = self.connectivity.shape[0]

        nodes_in_conn, self.elements_per_node = np.unique(
            connectivity, return_counts=True
        )

        if self.nnodes != len(self.elements_per_node):
            self.node_has_element = np.isin(np.arange(self.nnodes), nodes_in_conn)
            epn = -np.ones(self.nnodes, dtype=int)
            epn[nodes_in_conn] = self.elements_per_node
            self.elements_per_node = epn
            self.nodes_without_elements = np.arange(self.nnodes)[~self.node_has_element]
            self.nodes_with_elements = np.arange(self.nnodes)[self.node_has_element]
        else:
            self.nodes_without_elements = np.array([], dtype=int)
            self.nodes_with_elements = np.arange(self.nnodes)


class Cube(Mesh):
    def __init__(self, a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
        self.a = a
        self.b = b
        self.n = n

        nodes, connectivity = meshzoo.cube_hexa(a, b, n)
        etype = "hexahedron"

        super().__init__(nodes, connectivity, etype)
        self.edgenodes = 8


class Rectangle(Mesh):
    def __init__(self, a=(0, 0), b=(1, 1), n=(2, 2)):
        self.a = a
        self.b = b
        self.n = n

        nodes, connectivity = meshzoo.rectangle_quad(a, b, n)
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

        nodes, connectivity = meshzoo.cube_hexa(a, b, n=self.n)
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


class Cylinder(Cube):
    def __init__(
        self,
        a=(-1, -1, -1),
        b=(1, 1, 1),
        n=5,
        D=1,
        dD=0,
        exponent=4,
        symmetry=(False, False, False),
        switch=0.5,
    ):

        a = np.array(a)
        a[symmetry] = 0
        super().__init__(a, b, n)

        z = self.nodes.copy()

        r = np.sqrt(mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2)
        mask = np.logical_or(
            abs(mesh.nodes[:, 0]) > switch, abs(mesh.nodes[:, 1]) > switch
        )
        r[~mask] = (r[~mask] - 2 / 3) / r[~mask].max() * r[~mask] + 2 / 3
        z[:, 0] /= 0.05 + r
        z[:, 1] /= 0.05 + r
        z[:, 0] *= D / 2 * (1 + 2 * dD / D * mesh.nodes[:, 2] ** exponent)
        z[:, 1] *= D / 2 * (1 + 2 * dD / D * mesh.nodes[:, 2] ** exponent)
        z[:, 2] *= H / 2

        self.nodes = z
        self.update(self.connectivity)
