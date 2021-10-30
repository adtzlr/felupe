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
import felupe as fe


def test_region():

    mesh = fe.Rectangle()
    r = fe.RegionQuad(mesh)
    r = fe.RegionConstantQuad(mesh)

    mesh = fe.Cube()
    r = fe.RegionHexahedron(mesh)
    r = fe.RegionConstantHexahedron(mesh)

    mesh2 = fe.mesh.convert(mesh, 2, True, False, False)
    r = fe.RegionQuadraticHexahedron(mesh2)

    mesh3 = fe.mesh.convert(mesh, 2, True, True, True)
    r = fe.RegionTriQuadraticHexahedron(mesh3)

    triangle = fe.Triangle()
    points = triangle.points
    cells = np.arange(3).reshape(1, -1)
    mesh = fe.Mesh(points, cells, "triangle")
    r = fe.RegionTriangle(mesh)

    mesh2 = fe.mesh.convert(mesh, 2, True, False, False)
    r = fe.RegionQuadraticTriangle(mesh2)

    tetra = fe.Tetra()
    points = tetra.points
    cells = np.arange(4).reshape(1, -1)
    mesh = fe.Mesh(points, cells, "tetra")
    r = fe.RegionTetra(mesh)

    mesh2 = fe.mesh.convert(mesh, 2, True, False, False)
    r = fe.RegionQuadraticTetra(mesh2)

    triangle = fe.TriangleMINI()
    points = triangle.points
    cells = np.arange(4).reshape(1, -1)
    mesh = fe.Mesh(points, cells, "triangle-mini")
    r = fe.RegionTriangleMINI(mesh)

    tetra = fe.TetraMINI()
    points = tetra.points
    cells = np.arange(5).reshape(1, -1)
    mesh = fe.Mesh(points, cells, "tetra-mini")
    r = fe.RegionTetraMINI(mesh)

    order = 5
    dim = 3
    u = np.linspace(-1, 1, order + 1)
    z, y, x = np.meshgrid(u, u, u, indexing="ij")
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    cells = np.arange(len(points)).reshape(1, -1)
    mesh = fe.Mesh(points, cells, "lagrange")
    r = fe.RegionLagrange(mesh, order, dim)


if __name__ == "__main__":
    test_region()
