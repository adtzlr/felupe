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

import pytest


def pre():

    r = fe.RegionHexahedron(fe.Cube(n=3))

    u = fe.Field(r, dim=3)
    p = fe.Field(r)
    J = fe.Field(r, values=1)

    f = fe.FieldContainer((u, p, J))

    return r, f, (u, p, J)


def test_solve_check():

    r, _, (u, p, J) = pre()
    f = fe.FieldContainer([u])

    W = fe.constitution.NeoHooke(1, 3)

    F = f.extract()

    bounds = fe.dof.symmetry(u)
    dof0, dof1 = fe.dof.partition(f, bounds)

    ext0 = fe.dof.apply(f, bounds, dof0)

    L = fe.IntegralForm(W.gradient(F)[:-1], f, r.dV)
    a = fe.IntegralForm(W.hessian(F), f, r.dV, f)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fe.tools.solve(A, b, f, dof0, dof1, f.offsets, ext0)
    assert dx[0].shape == f[0].values.ravel().shape

    fe.tools.check(dx, f, b, dof1, dof0, verbose=0)
    fe.tools.check(dx, f, b, dof1, dof0, verbose=1)

    fe.tools.save(r, f)
    fe.tools.save(r, f, r=b)
    fe.tools.save(
        r,
        f,
        r=b,
        gradient=W.gradient(F),
    )

    for b in [L.assemble(), L.assemble().toarray(), L.assemble().toarray()[:, 0]]:
        force = fe.tools.force(f, b, bounds["symx"])
        moment = fe.tools.moment(f, b, bounds["symx"])

    for a in [2, 3, 4, 5]:
        curve = fe.tools.curve(np.arange(a), np.ones(a) * force[0])

    s = fe.math.dot(W.gradient(F)[0], fe.math.inv(fe.math.cof(F[0])))

    cauchy = fe.tools.project(fe.math.tovoigt(s), region=r, average=True)
    assert cauchy.shape == (r.mesh.npoints, 6)

    cauchy = fe.tools.project(fe.math.tovoigt(s), region=r, average=False)
    assert cauchy.shape == (r.mesh.cells.size, 6)

    cauchy = fe.tools.project(fe.math.tovoigt(s), region=r, mean=True)
    assert cauchy.shape == (r.mesh.npoints, 6)


def test_solve_mixed_check():

    r, f, fields = pre()
    u = fields[0]

    f = fe.FieldContainer(fields)

    F, p, J = f.extract()

    W = fe.NeoHooke(1, 3)
    W_mixed = fe.ThreeFieldVariation(W)

    F = u.extract()

    bounds = fe.dof.symmetry(u)
    dof0, dof1 = fe.dof.partition(f, bounds)

    ext0 = fe.dof.apply(f, bounds, dof0)

    L = fe.IntegralForm(W_mixed.gradient([F, p, J])[:-1], f, r.dV)
    a = fe.IntegralForm(W_mixed.hessian([F, p, J]), f, r.dV, f)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fe.tools.solve(A, b, f, dof0, dof1, f.offsets, ext0)

    assert dx[0].shape == u.values.ravel().shape
    assert dx[1].shape == fields[1].values.ravel().shape
    assert dx[2].shape == fields[2].values.ravel().shape

    fe.tools.check(dx, f, b, dof1, dof0, verbose=0)
    fe.tools.check(dx, f, b, dof1, dof0, verbose=1)

    fe.tools.save(r, f)
    fe.tools.save(r, f, r=b)
    fe.tools.save(
        r,
        f,
        r=b,
        gradient=W_mixed.gradient([F, p, J]),
    )

    force = fe.tools.force(f, b, bounds["symx"])
    moment = fe.tools.moment(f, b, bounds["symx"])

    for a in [2, 3, 4, 5]:
        curve = fe.tools.curve(np.arange(a), np.ones(a) * force[0])


def test_newton_simple():
    def fun(x):
        return (x - 3) ** 2

    def jac(x):
        return np.array([2 * (x - 3)])

    x0 = np.array([3.1])

    res = fe.tools.newtonrhapson(
        x0, fun, jac, solve=np.linalg.solve, maxiter=32, verbose=True, timing=False
    )

    res = fe.tools.newtonrhapson(
        x0, fun, jac, solve=np.linalg.solve, maxiter=32, verbose=True, timing=True
    )

    assert abs(res.fun) < 1e-6
    assert np.isclose(res.x, 3, rtol=1e-2)

    with pytest.raises(ValueError):
        res = fe.tools.newtonrhapson(
            x0, fun, jac, solve=np.linalg.solve, maxiter=4, verbose=False
        )

    with pytest.raises(ValueError):
        x0 = np.array([np.nan])
        res = fe.tools.newtonrhapson(x0, fun, jac, solve=np.linalg.solve)


def test_newton():

    # create a hexahedron-region on a cube
    region = fe.RegionHexahedron(fe.Cube(n=6))

    # add a displacement field and apply a uniaxial elongation on the cube
    displacement = fe.Field(region, dim=3)
    field = fe.FieldContainer([displacement])
    boundaries, loadcase = fe.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    umat = fe.NeoHooke(mu=1.0, bulk=2.0)

    for kwargs in [{"parallel": True}, {"jit": True}]:

        # newton-rhapson procedure
        res = fe.newtonrhapson(
            field,
            umat=umat,
            timing=True,
            verbose=True,
            kwargs=kwargs,
            **loadcase,
        )


def test_newton_plane():

    # create a quad-region on a rectangle
    region = fe.RegionQuad(fe.Rectangle(n=6))

    # add a displacement field and apply a uniaxial elongation on the rectangle
    displacement = fe.Field(region, dim=2)
    field = fe.FieldContainer([displacement])
    boundaries, loadcase = fe.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    umat = fe.LinearElasticPlaneStress(E=1.0, nu=0.3)

    # newton-rhapson procedure
    res = fe.newtonrhapson(
        field,
        umat=umat,
        timing=True,
        verbose=True,
        kwargs={},
        **loadcase,
    )

    # define the constitutive material behavior
    umat = fe.LinearElasticPlaneStrain(E=1.0, nu=0.3)

    # newton-rhapson procedure
    res = fe.newtonrhapson(
        field,
        umat=umat,
        timing=True,
        verbose=True,
        kwargs={},
        **loadcase,
    )


def test_newton_linearelastic():

    # create a hexahedron-region on a cube
    region = fe.RegionHexahedron(fe.Cube(n=6))

    # add a displacement field and apply a uniaxial elongation on the cube
    displacement = fe.Field(region, dim=3)
    field = fe.FieldContainer([displacement])
    boundaries, loadcase = fe.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    umat = fe.LinearElastic(E=1.0, nu=0.3)

    # newton-rhapson procedure
    res = fe.newtonrhapson(
        field,
        umat=umat,
        timing=True,
        verbose=True,
        kwargs={"grad": True, "sym": True, "add_identity": False},
        **loadcase,
    )


def test_newton_mixed():

    # create a hexahedron-region on a cube
    mesh = fe.Cube(n=6)
    region = fe.RegionHexahedron(mesh)
    region0 = fe.RegionConstantHexahedron(fe.mesh.convert(mesh, 0))

    # add a displacement field and apply a uniaxial elongation on the cube
    u = fe.Field(region, dim=3)
    p = fe.Field(region0)
    J = fe.Field(region0, values=1)
    field = fe.FieldContainer((u, p, J))

    assert len(field) == 3

    boundaries, loadcase = fe.dof.uniaxial(field, move=0.2, clamped=True)

    # deformation gradient
    F = field.extract(grad=True, sym=False, add_identity=True)

    # define the constitutive material behavior
    nh = fe.NeoHooke(mu=1.0, bulk=2.0)
    umat = fe.ThreeFieldVariation(nh)

    # newton-rhapson procedure
    res = fe.newtonrhapson(x0=field, umat=umat, kwargs={}, **loadcase)


def test_newton_body():

    # create a hexahedron-region on a cube
    mesh = fe.Cube(n=6)
    region = fe.RegionHexahedron(mesh)
    region0 = fe.RegionConstantHexahedron(fe.mesh.convert(mesh, 0))

    # add a displacement field and apply a uniaxial elongation on the cube
    u = fe.Field(region, dim=3)
    p = fe.Field(region0)
    J = fe.Field(region0, values=1)
    field = fe.FieldContainer((u, p, J))

    boundaries, loadcase = fe.dof.uniaxial(field, move=0.2, clamped=True)

    # define the constitutive material behavior
    nh = fe.NeoHooke(mu=1.0, bulk=2.0)
    umat = fe.ThreeFieldVariation(nh)
    body = fe.SolidBody(umat, field)

    # create a region, a field and a body for a pressure boundary
    regionp = fe.RegionHexahedronBoundary(mesh, only_surface=True)
    fieldp = fe.FieldContainer([fe.Field(regionp, dim=3)])
    bodyp = fe.SolidBodyPressure(fieldp, pressure=1.0)

    # newton-rhapson procedure
    res = fe.newtonrhapson(
        items=[body, bodyp],
        kwargs={},
        **loadcase,
    )


if __name__ == "__main__":
    test_solve_check()
    test_solve_mixed_check()
    test_newton_simple()
    test_newton()
    test_newton_mixed()
    test_newton_plane()
    test_newton_linearelastic()
    test_newton_body()
