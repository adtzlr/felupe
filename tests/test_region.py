# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import felupe as fem


def test_region():

    mesh = fem.Point()
    r = fem.RegionVertex(mesh)

    assert r.h.flags["C_CONTIGUOUS"]

    mesh = fem.Rectangle()
    r = fem.RegionQuad(mesh, uniform=True)

    assert r.h.flags["C_CONTIGUOUS"]
    assert r.dhdX.flags["C_CONTIGUOUS"]

    r.plot(off_screen=True)

    r = fem.RegionQuadBoundary(mesh)
    r = fem.RegionQuadBoundary(mesh, ensure_3d=True)
    r = fem.RegionConstantQuad(mesh)

    mesh2 = fem.mesh.convert(mesh, 2, True, False, False)
    r = fem.RegionQuadraticQuad(mesh2)
    f = fem.FieldsMixed(r)

    r = fem.RegionQuadraticQuadBoundary(mesh2, ensure_3d=True)

    assert r.h.flags["C_CONTIGUOUS"]
    assert r.dhdX.flags["C_CONTIGUOUS"]

    mesh3 = fem.mesh.convert(mesh, 2, True, True, False)
    r = fem.RegionQuad(mesh3)
    r = fem.RegionQuadraticQuad(mesh3)
    r = fem.RegionBiQuadraticQuad(mesh3)
    f = fem.FieldsMixed(r)

    r = fem.RegionBiQuadraticQuadBoundary(mesh3)

    assert r.h.flags["C_CONTIGUOUS"]
    assert r.dhdX.flags["C_CONTIGUOUS"]

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

    assert r.h.flags["C_CONTIGUOUS"]
    assert r.dhdX.flags["C_CONTIGUOUS"]

    triangle = fem.Triangle()
    points = triangle.points
    cells = np.arange(3).reshape(1, -1)
    mesh = fem.Mesh(points, cells, "triangle")
    r = fem.RegionTriangle(mesh)

    mesh2 = fem.mesh.convert(mesh, 2, True, False, False)
    r = fem.RegionTriangle(mesh2)
    r = fem.RegionQuadraticTriangle(mesh2)
    f = fem.FieldsMixed(r)

    assert r.h.flags["C_CONTIGUOUS"]
    assert r.dhdX.flags["C_CONTIGUOUS"]

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

    assert r.h.flags["C_CONTIGUOUS"]
    assert r.dhdX.flags["C_CONTIGUOUS"]

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


def test_region_negative_volumes_cells():

    mesh = fem.Rectangle()
    mesh.cells = mesh.cells[:, ::-1]

    with pytest.warns():  # negative volumes of cells
        region = fem.RegionQuad(mesh)


def test_container_warning():

    mesh = fem.MeshContainer([fem.Rectangle()])
    element = fem.Quad()
    quadrature = fem.GaussLegendre(order=1, dim=2)
    with pytest.raises(TypeError):
        region = fem.Region(mesh, element, quadrature)

    mesh = fem.MeshContainer([fem.Rectangle()])
    with pytest.raises(TypeError):
        region = fem.RegionQuad(mesh)

    mesh = fem.MeshContainer([fem.Rectangle().add_midpoints_edges()])
    with pytest.raises(TypeError):
        region = fem.RegionQuadraticQuad(mesh)

    mesh = fem.MeshContainer([fem.Rectangle().triangulate()])
    with pytest.raises(TypeError):
        region = fem.RegionTriangle(mesh)

    mesh = fem.MeshContainer([fem.Cube()])
    with pytest.raises(TypeError):
        region = fem.RegionHexahedron(mesh)

    mesh = fem.MeshContainer([fem.Cube().add_midpoints_edges()])
    with pytest.raises(TypeError):
        region = fem.RegionQuadraticHexahedron(mesh)

    mesh = fem.MeshContainer([fem.Cube().triangulate()])
    with pytest.raises(TypeError):
        region = fem.RegionTetra(mesh)


def test_region_boundary_reload():
    "The quantities of a boundary region are kept by copy, astype and reload."

    for mesh, region_class in [
        (fem.Cube(n=3), fem.RegionHexahedronBoundary),
        (fem.Rectangle(n=4), fem.RegionQuadBoundary),
    ]:
        mask = mesh.points[:, 0] == 1
        region = region_class(mesh, mask=mask)

        dV = region.dV.sum()
        dA = region.dA.copy()
        normals = region.normals.copy()
        tangents = [tangent.copy() for tangent in region.tangents]

        for other in [
            region.copy(),
            region.astype(np.float64),
            region.astype(np.float32),
        ]:
            assert np.isclose(other.dV.sum(), dV)
            assert np.allclose(other.dA, dA)
            assert np.allclose(other.normals, normals)
            assert np.allclose(other.dA / other.dV, other.normals)

        region.reload()

        assert np.isclose(region.dV.sum(), dV)
        assert np.allclose(region.dA, dA)
        assert np.allclose(region.normals, normals)
        assert all(np.allclose(new, old) for new, old in zip(region.tangents, tangents))
        assert np.allclose(region.dA / region.dV, region.normals)

    # a pressure of one on the face x=1 of a unit cube integrates to a force of -1
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedronBoundary(mesh, mask=mesh.points[:, 0] == 1)

    def force(region):
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        vector = fem.SolidBodyPressure(field, pressure=1.0).assemble.vector().toarray()
        return vector.reshape(-1, 3).sum(axis=0)

    reference = force(region)

    assert np.allclose(reference, [-1.0, 0.0, 0.0])
    assert np.allclose(force(region.copy()), reference)
    assert np.allclose(force(region.astype(np.float32)), reference)

    region.reload()

    assert np.allclose(force(region), reference)


if __name__ == "__main__":
    test_region()
    test_region_negative_volumes_cells()
    test_container_warning()
    test_region_boundary_reload()
