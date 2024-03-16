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
import pytest

import felupe as fem


def test_region():
    mesh = fem.Rectangle()
    r = fem.RegionQuad(mesh)

    r.plot(off_screen=True)

    r = fem.RegionQuadBoundary(mesh)
    r = fem.RegionQuadBoundary(mesh, ensure_3d=True)
    r = fem.RegionConstantQuad(mesh)

    mesh2 = fem.mesh.convert(mesh, 2, True, False, False)
    r = fem.RegionQuadraticQuad(mesh2)
    f = fem.FieldsMixed(r)

    r = fem.RegionQuadraticQuadBoundary(mesh2, ensure_3d=True)

    mesh3 = fem.mesh.convert(mesh, 2, True, True, False)
    r = fem.RegionQuad(mesh3)
    r = fem.RegionQuadraticQuad(mesh3)
    r = fem.RegionBiQuadraticQuad(mesh3)
    f = fem.FieldsMixed(r)

    r = fem.RegionBiQuadraticQuadBoundary(mesh3)

    mesh.cell_type = "some_fancy_cell_type"
    with pytest.raises(NotImplementedError):
        r = fem.RegionBoundary(
            mesh, fem.Quad(), fem.GaussLegendreBoundary(order=1, dim=2)
        )

    mesh = fem.Cube()
    r = fem.RegionHexahedron(mesh)
    r = fem.RegionHexahedronBoundary(mesh)
    r = fem.RegionHexahedronBoundary(mesh, only_surface=False)
    r = fem.RegionHexahedronBoundary(mesh, mask=[0, 3])
    r = fem.RegionHexahedronBoundary(mesh, only_surface=False, mask=[0, 3])

    m = r.mesh_faces()
    assert np.allclose(m.points, mesh.points)
    assert m.cell_type == "quad"

    r = fem.RegionConstantHexahedron(mesh)

    mesh2 = fem.mesh.convert(mesh, 2, True, False, False)
    r = fem.RegionQuadraticHexahedron(mesh2)
    f = fem.FieldsMixed(r)

    r = fem.RegionQuadraticHexahedronBoundary(mesh2)

    mesh3 = fem.mesh.convert(mesh, 2, True, True, True)
    r = fem.RegionHexahedron(mesh3)
    r = fem.RegionQuadraticHexahedron(mesh3)
    r = fem.RegionTriQuadraticHexahedron(mesh3)
    f = fem.FieldsMixed(r)

    r = fem.RegionTriQuadraticHexahedronBoundary(mesh3)

    triangle = fem.Triangle()
    points = triangle.points
    cells = np.arange(3).reshape(1, -1)
    mesh = fem.Mesh(points, cells, "triangle")
    r = fem.RegionTriangle(mesh)

    mesh2 = fem.mesh.convert(mesh, 2, True, False, False)
    r = fem.RegionTriangle(mesh2)
    r = fem.RegionQuadraticTriangle(mesh2)
    f = fem.FieldsMixed(r)

    tetra = fem.Tetra()
    points = tetra.points
    cells = np.arange(4).reshape(1, -1)
    mesh = fem.Mesh(points, cells, "tetra")
    r = fem.RegionTetra(mesh)

    mesh2 = fem.mesh.convert(mesh, 2, True, False, False)
    r = fem.RegionTetra(mesh2)
    r = fem.RegionQuadraticTetra(mesh2)
    f = fem.FieldsMixed(r)

    triangle = fem.TriangleMINI()
    points = triangle.points
    cells = np.arange(4).reshape(1, -1)
    mesh = fem.Mesh(points, cells, "triangle-mini")
    r = fem.RegionTriangleMINI(mesh)
    f = fem.FieldsMixed(r)

    tetra = fem.TetraMINI()
    points = tetra.points
    cells = np.arange(5).reshape(1, -1)
    mesh = fem.Mesh(points, cells, "tetra-mini")
    r = fem.RegionTetraMINI(mesh)
    f = fem.FieldsMixed(r)

    order = 5
    dim = 3
    u = np.linspace(-1, 1, order + 1)
    z, y, x = np.meshgrid(u, u, u, indexing="ij")
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    cells = np.arange(len(points)).reshape(1, -1)
    mesh = fem.Mesh(points, cells, "lagrange")
    r = fem.RegionLagrange(mesh, order, dim, permute=False)
    assert not np.any(r.dV <= 0)

    order = 5
    dim = 2
    mesh = fem.mesh.RectangleArbitraryOrderQuad(order=order)
    r = fem.RegionLagrange(mesh, order, dim)
    assert not np.any(r.dV <= 0)

    order = 5
    dim = 3
    mesh = fem.mesh.CubeArbitraryOrderHexahedron(order=order)
    r = fem.RegionLagrange(mesh, order, dim)
    assert not np.any(r.dV <= 0)


if __name__ == "__main__":
    test_region()
