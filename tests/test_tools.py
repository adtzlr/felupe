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
import os

import numpy as np
import pytest

import felupe as fem


def pre():
    r = fem.RegionHexahedron(fem.Cube(n=3))

    u = fem.Field(r, dim=3)
    p = fem.Field(r)
    J = fem.Field(r, values=1)

    f = fem.FieldContainer((u, p, J))

    return r, f, (u, p, J)


def test_solve():
    r, _, (u, p, J) = pre()
    f = fem.FieldContainer([u])

    W = fem.constitution.NeoHooke(1, 3)

    F = f.extract()

    bounds = fem.dof.symmetry(u)
    dof0, dof1 = fem.dof.partition(f, bounds)

    ext0 = fem.dof.apply(f, bounds, dof0)

    L = fem.IntegralForm(W.gradient(F)[:-1], f, r.dV)
    a = fem.IntegralForm(W.hessian(F), f, r.dV, f)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fem.tools.solve(A, b, f, dof0, dof1, f.offsets, ext0)
    assert dx[0].shape == f[0].values.ravel().shape

    fem.tools.save(r, f)
    fem.tools.save(r, f, forces=b)
    fem.tools.save(
        r,
        f,
        forces=b,
        gradient=W.gradient(F),
    )

    for b in [L.assemble(), L.assemble().toarray(), L.assemble().toarray()[:, 0]]:
        force = fem.tools.force(f, b, bounds["symx"])
        moment = fem.tools.moment(f, b, bounds["symx"])

    for a in [2, 3, 4, 5]:
        curve = fem.tools.curve(np.arange(a), np.ones(a) * force[0])

    s = fem.math.dot(W.gradient(F)[0], fem.math.inv(fem.math.cof(F[0])))

    cauchy = fem.tools.project(fem.math.tovoigt(s), region=r, average=True)
    assert cauchy.shape == (r.mesh.npoints, 6)

    cauchy = fem.tools.project(fem.math.tovoigt(s), region=r, average=False)
    assert cauchy.shape == (r.mesh.cells.size, 6)

    cauchy = fem.tools.project(fem.math.tovoigt(s), region=r, mean=True)
    assert cauchy.shape == (r.mesh.npoints, 6)

    cauchy = fem.tools.topoints(s, region=r)
    assert cauchy.shape == (r.mesh.npoints, 3, 3)

    cauchy = fem.tools.topoints(s[:2, :2], region=r)
    assert cauchy.shape == (r.mesh.npoints, 2, 2)

    cauchy = fem.tools.topoints(s[0, 0], region=r)
    assert cauchy.shape == (r.mesh.npoints,)


def test_solve_mixed():
    r, f, fields = pre()
    u = fields[0]

    f = fem.FieldContainer(fields)

    F, p, J = f.extract()

    W = fem.NeoHooke(1, 3)
    W_mixed = fem.ThreeFieldVariation(W)

    F = u.extract()

    bounds = fem.dof.symmetry(u)
    dof0, dof1 = fem.dof.partition(f, bounds)

    ext0 = fem.dof.apply(f, bounds, dof0)

    L = fem.IntegralForm(W_mixed.gradient([F, p, J])[:-1], f, r.dV)
    a = fem.IntegralForm(W_mixed.hessian([F, p, J]), f, r.dV, f)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fem.tools.solve(A, b, f, dof0, dof1, f.offsets, ext0)

    assert dx[0].shape == u.values.ravel().shape
    assert dx[1].shape == fields[1].values.ravel().shape
    assert dx[2].shape == fields[2].values.ravel().shape

    fem.tools.save(r, f)
    fem.tools.save(r, f, forces=b)
    fem.tools.save(
        r,
        f,
        forces=b,
        gradient=W_mixed.gradient([F, p, J]),
    )

    force = fem.tools.force(f, b, bounds["symx"])
    moment = fem.tools.moment(f, b, bounds["symx"])

    for a in [2, 3, 4, 5]:
        curve = fem.tools.curve(np.arange(a), np.ones(a) * force[0])


def test_newton_simple():
    def fun(x):
        return (x - 3) ** 2

    def jac(x):
        return np.array([2 * (x - 3)])

    x0 = np.array([3.1])

    os.environ["FELUPE_VERBOSE"] = "true"

    res = fem.tools.newtonrhapson(x0, fun, jac, solve=np.linalg.solve, maxiter=32)

    os.environ.pop("FELUPE_VERBOSE")

    assert abs(res.fun) < 1e-6
    assert np.isclose(res.x, 3, rtol=1e-2)

    with pytest.raises(ValueError):
        res = fem.tools.newtonrhapson(
            x0, fun, jac, solve=np.linalg.solve, maxiter=4, verbose=False
        )

    with pytest.raises(ValueError):
        x0 = np.array([np.nan])
        res = fem.tools.newtonrhapson(x0, fun, jac, solve=np.linalg.solve)


def test_newton():
    # create a hexahedron-region on a cube
    region = fem.RegionHexahedron(fem.Cube(n=6))

    # add a displacement field and apply a uniaxial elongation on the cube
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])
    boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    umat = fem.NeoHooke(mu=1.0, bulk=2.0)

    for kwargs in [{"parallel": True, "umat": umat}]:
        # newton-rhapson procedure
        res = fem.newtonrhapson(
            field,
            verbose=True,
            kwargs=kwargs,
            **loadcase,
        )


def test_newton_plane():
    # create a quad-region on a rectangle
    region = fem.RegionQuad(fem.Rectangle(n=6))

    # add a displacement field and apply a uniaxial elongation on the rectangle
    displacement = fem.Field(region, dim=2)
    field = fem.FieldContainer([displacement])
    boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    umat = fem.LinearElasticPlaneStress(E=1.0, nu=0.3)

    # newton-rhapson procedure
    res = fem.newtonrhapson(
        field,
        verbose=True,
        kwargs=dict(umat=umat),
        **loadcase,
    )

    # define the constitutive material behavior
    umat = fem.LinearElasticPlaneStrain(E=1.0, nu=0.3)

    # newton-rhapson procedure
    res = fem.newtonrhapson(
        field,
        verbose=True,
        kwargs=dict(umat=umat),
        **loadcase,
    )


def test_newton_linearelastic():
    # create a hexahedron-region on a cube
    region = fem.RegionHexahedron(fem.Cube(n=6))

    # add a displacement field and apply a uniaxial elongation on the cube
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])
    boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    umat = fem.LinearElastic(E=1.0, nu=0.3)

    # newton-rhapson procedure
    res = fem.newtonrhapson(
        field,
        verbose=True,
        kwargs={"umat": umat, "grad": True, "sym": True, "add_identity": False},
        **loadcase,
    )


def test_newton_mixed():
    # create a hexahedron-region on a cube
    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    region0 = fem.RegionConstantHexahedron(fem.mesh.convert(mesh, 0))

    # add a displacement field and apply a uniaxial elongation on the cube
    u = fem.Field(region, dim=3)
    p = fem.Field(region0)
    J = fem.Field(region0, values=1)
    field = fem.FieldContainer((u, p, J))

    assert len(field) == 3

    boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, clamped=True)

    # deformation gradient
    F = field.extract(grad=True, sym=False, add_identity=True)

    # define the constitutive material behavior
    nh = fem.NeoHooke(mu=1.0, bulk=2.0)
    umat = fem.ThreeFieldVariation(nh)

    # newton-rhapson procedure
    res = fem.newtonrhapson(x0=field, kwargs=dict(umat=umat), **loadcase)


def test_newton_body():
    # create a hexahedron-region on a cube
    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    region0 = fem.RegionConstantHexahedron(fem.mesh.convert(mesh, 0))

    # add a displacement field and apply a uniaxial elongation on the cube
    u = fem.Field(region, dim=3)
    p = fem.Field(region0)
    J = fem.Field(region0, values=1)
    field = fem.FieldContainer((u, p, J))

    boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    nh = fem.NeoHooke(mu=1.0, bulk=2.0)
    umat = fem.ThreeFieldVariation(nh)
    body = fem.SolidBody(umat, field)

    # create a region, a field and a body for a pressure boundary
    regionp = fem.RegionHexahedronBoundary(mesh, only_surface=True)
    fieldp = fem.FieldContainer([fem.Field(regionp, dim=3)])
    bodyp = fem.SolidBodyPressure(fieldp, pressure=1.0)

    # newton-rhapson procedure
    res = fem.newtonrhapson(
        items=[body, bodyp],
        kwargs={},
        **loadcase,
    )


def test_project():
    # rectangle (triangle)
    mesh = fem.Rectangle(n=2).triangulate()
    mesh.update(points=np.vstack([mesh.points, [1000, 1000]]))
    region = fem.RegionTriangle(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.project(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.project(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    # rectangle (quadratic triangle)
    mesh = fem.Rectangle(n=2).triangulate().add_midpoints_edges()
    region = fem.RegionQuadraticTriangle(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.project(values, region, average=True)

    # cube (tetra)
    mesh = fem.Cube(n=2).triangulate()
    region = fem.RegionTetra(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.project(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.project(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # cube (quadratic tetra)
    mesh = fem.Cube(n=2).triangulate().add_midpoints_edges()
    region = fem.RegionQuadraticTetra(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.project(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.project(values, region, average=True)


def test_topoints():
    mesh = fem.Rectangle(n=2).triangulate()
    region = fem.RegionTriangle(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    # single quadrature-point values
    data = fem.topoints(values, region)
    assert data.shape == (mesh.npoints, 3, 3)

    mesh = fem.Rectangle(n=3).convert(2, 1)
    region = fem.RegionQuadraticQuad(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    # trim values array to number of points-per-cell
    data = fem.topoints(values, region)
    assert data.shape == (mesh.npoints, 3, 3)

    data = fem.topoints(values, region, average=False)
    assert data.shape == (mesh.cells.size, 3, 3)

    data = fem.topoints(values, region, mean=True)
    assert data.shape == (mesh.npoints, 3, 3)

    data = fem.topoints(values, region, average=False, mean=True)
    assert data.shape == (mesh.cells.size, 3, 3)


def test_extrapolate():
    # rectangle (triangle)
    mesh = fem.Rectangle(n=2)
    mesh.update(points=np.vstack([mesh.points, [1000, 1000]]))
    region = fem.RegionQuad(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.tools.extrapolate(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected[:-1]])

    # rectangle (quadratic triangle)
    mesh = fem.Rectangle(n=2).add_midpoints_edges()
    region = fem.RegionQuadraticQuad(mesh)
    field = fem.FieldAxisymmetric(region, dim=2)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.tools.extrapolate(values, region, average=True)

    # cube (tetra)
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=False)
    assert projected.shape == (mesh.cells.size, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.tools.extrapolate(values, region, average=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # cube (quadratic tetra)
    mesh = fem.Cube(n=2).add_midpoints_edges()
    region = fem.RegionQuadraticHexahedron(mesh)
    field = fem.Field(region, dim=3)
    values = field.extract()

    projected = fem.tools.extrapolate(values, region, average=True, mean=True)
    assert projected.shape == (mesh.npoints, 3, 3)
    assert not np.any(np.isnan(projected))
    assert np.all([np.allclose(np.eye(3), res) for res in projected])

    # this is wrong
    with pytest.raises(ValueError):
        projected = fem.tools.extrapolate(values, region, average=True)


if __name__ == "__main__":
    test_solve()
    test_solve_mixed()
    test_newton_simple()
    test_newton()
    test_newton_mixed()
    test_newton_plane()
    test_newton_linearelastic()
    test_newton_body()
    test_project()
    test_topoints()
    test_extrapolate()
