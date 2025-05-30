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
from scipy.sparse import csr_matrix

import felupe as fem


def test_simple():
    umat = fem.LinearElastic(E=1, nu=0.3)

    m = fem.Cube(n=3)
    r = fem.RegionHexahedron(m, uniform=True)
    u = fem.FieldContainer([fem.Field(r, dim=3)])

    b = fem.SolidBody(umat, u, density=1.0)
    r = b.assemble.vector()

    K = b.assemble.matrix()
    M = b.assemble.mass()
    r = b.assemble.vector(u)
    F = b.results.kinematics
    P = b.results.stress
    s = b.evaluate.cauchy_stress()
    t = b.evaluate.kirchhoff_stress()
    C = b.results.elasticity

    assert K.shape == (81, 81)
    assert M.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F[0].shape == (3, 3, 8, 8)
    assert P[0].shape == (3, 3, 8, 8)
    assert s.shape == (3, 3, 8, 8)
    assert t.shape == (3, 3, 8, 8)
    assert C[0].shape == (3, 3, 3, 3, 1, 1)


def test_pressure():
    umat = fem.LinearElastic(E=1, nu=0.3)

    m = fem.Cube(n=3)
    h = fem.RegionHexahedron(m)
    u = fem.Field(h, dim=3)

    np.random.seed(156)
    u.values = np.random.rand(*u.values.shape) / 10
    v = fem.FieldContainer([u])

    s = fem.RegionHexahedronBoundary(m)
    p = fem.Field(s, dim=3)
    q = fem.FieldContainer([p])

    b = fem.SolidBody(umat, v)
    z = fem.SolidBodyPressure(q)
    c = fem.SolidBodyPressure(q, pressure=1.0)

    assert z.results.pressure == 0
    assert c.results.pressure == 1.0

    r = b.assemble.vector()
    K = b.assemble.matrix()
    r = b.assemble.vector(v)
    F = b.results.kinematics
    s = b.results.stress
    C = b.results.elasticity

    assert K.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F[0].shape == (3, 3, 8, 8)
    assert s[0].shape == (3, 3, 8, 8)
    assert C[0].shape == (3, 3, 3, 3, 1, 1)

    r = c.assemble.vector()
    K = c.assemble.matrix()
    K = c.assemble.matrix(q, resize=b.assemble.matrix(), pressure=2.0)
    r = c.assemble.vector(q)
    r = c.assemble.vector(q, resize=b.assemble.vector(), pressure=2.0)
    F = c.results.kinematics

    assert K.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F[0].shape == (3, 3, 4, 24)

    c._update(v, q)
    assert np.allclose(v[0].values, q[0].values)

    np.random.seed(156)
    w = fem.FieldsMixed(h)
    w[0].values = np.random.rand(*w[0].values.shape) / 10

    c._update(w, v)
    assert np.allclose(w[0].values, v[0].values)


def pre(dim, bulk=2):
    umat = fem.NeoHooke(mu=1, bulk=bulk)

    m = fem.Cube(n=3)
    r = fem.RegionHexahedron(m)
    u = fem.Field(r, dim=dim)

    np.random.seed(156)
    u.values = np.random.rand(*u.values.shape) / 10

    return umat, fem.FieldContainer([u])


def pre_axi(bulk=2):
    umat = fem.NeoHooke(mu=1, bulk=bulk)

    m = fem.Rectangle(n=3)
    r = fem.RegionQuad(m)
    u = fem.FieldAxisymmetric(r)

    np.random.seed(156)
    u.values = np.random.rand(*u.values.shape) / 10

    return umat, fem.FieldContainer([u])


def pre_planestrain(bulk=2):
    umat = fem.NeoHooke(mu=1, bulk=bulk)

    m = fem.Rectangle(n=3)
    r = fem.RegionQuad(m)
    u = fem.FieldPlaneStrain(r)

    np.random.seed(156)
    u.values = np.random.rand(*u.values.shape) / 10

    return umat, fem.FieldContainer([u])


def pre_mixed(dim):
    umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=2))

    m = fem.Cube(n=3)
    r = fem.RegionHexahedron(m)
    u = fem.FieldsMixed(r, n=3)

    np.random.seed(156)
    u[0].values = np.random.rand(*u[0].values.shape) / 10
    u[1].values = np.random.rand(*u[1].values.shape) / 10
    u[2].values = np.random.rand(*u[2].values.shape) / 10

    return umat, u


def test_solidbody():
    umat, u = pre(dim=3)
    b = fem.SolidBody(umat=umat, field=u, statevars=np.ones(5))

    for parallel in [False, True]:
        kwargs = {"parallel": parallel}

        r1 = b.assemble.vector(u, **kwargs)
        assert r1.shape == (81, 1)

        r1b = b.results.force
        assert np.allclose(r1.toarray(), r1b.toarray())

        r2 = b.assemble.vector(**kwargs)
        assert np.allclose(r1.toarray(), r2.toarray())

        K1 = b.assemble.matrix(u, **kwargs)
        assert K1.shape == (81, 81)

        K1b = b.results.stiffness
        assert np.allclose(K1.toarray(), K1b.toarray())

        K2 = b.assemble.matrix(**kwargs)
        assert np.allclose(K1.toarray(), K2.toarray())

        P1 = b.results.stress
        P2 = b.evaluate.gradient()
        P2 = b.evaluate.gradient(u)
        assert np.allclose(P1, P2)

        A1 = b.results.elasticity
        A2 = b.evaluate.hessian()
        A2 = b.evaluate.hessian(u)
        assert np.allclose(A1, A2)

        F1 = b.results.kinematics
        F2 = b._extract(u)
        assert np.allclose(F1, F2)

        s1 = b.evaluate.cauchy_stress()
        s2 = b.evaluate.cauchy_stress(u)
        assert np.allclose(s1, s2)

        t1 = b.evaluate.kirchhoff_stress()
        t2 = b.evaluate.kirchhoff_stress(u)
        assert np.allclose(t1, t2)


def test_solidbody_incompressible():
    umat, u = pre(dim=3, bulk=None)
    b = fem.SolidBodyNearlyIncompressible(
        umat=umat, field=u, bulk=5000, statevars=np.ones(5)
    )

    umat = fem.OgdenRoxburgh(fem.NeoHooke(mu=1, bulk=5000), r=3, m=1, beta=0)
    ax = umat.plot(
        ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
        ps=None,
        bx=None,
    )

    umat = fem.OgdenRoxburgh(fem.NeoHooke(mu=1), r=3, m=1, beta=0)
    b = fem.SolidBodyNearlyIncompressible(
        umat=umat,
        field=u,
        bulk=5000,
        state=fem.StateNearlyIncompressible(u),
        density=1.0,
    )

    M = b.assemble.mass()
    assert M.shape == (81, 81)

    for parallel in [False, True]:
        kwargs = {"parallel": parallel}

        r1 = b.assemble.vector(u, **kwargs)
        assert r1.shape == (81, 1)

        r1b = b.results.force
        assert np.allclose(r1.toarray(), r1b.toarray())

        r2 = b.assemble.vector(**kwargs)
        assert np.allclose(r1.toarray(), r2.toarray())

        K1 = b.assemble.matrix(u, **kwargs)
        assert K1.shape == (81, 81)

        K1b = b.results.stiffness
        assert np.allclose(K1.toarray(), K1b.toarray())

        K2 = b.assemble.matrix(**kwargs)
        assert np.allclose(K1.toarray(), K2.toarray())

        P1 = b.results.stress
        P2 = b.evaluate.gradient()
        P2 = b.evaluate.gradient(u)
        assert np.allclose(P1, P2)

        A1 = b.results.elasticity
        A2 = b.evaluate.hessian()
        A2 = b.evaluate.hessian(u)
        assert np.allclose(A1, A2)

        F1 = b.results.kinematics
        F2 = b._extract(u)
        assert np.allclose(F1, F2)

        p1 = b.evaluate.stress()
        p2 = b.evaluate.stress(u)
        assert np.allclose(p1, p2)

        s1 = b.evaluate.cauchy_stress()
        s2 = b.evaluate.cauchy_stress(u)
        assert np.allclose(s1, s2)

        t1 = b.evaluate.kirchhoff_stress()
        t2 = b.evaluate.kirchhoff_stress(u)
        assert np.allclose(t1, t2)


def test_solidbody_axi_incompressible():
    umat, u = pre_axi(bulk=None)
    b = fem.SolidBodyNearlyIncompressible(umat=umat, field=u, bulk=5000)
    b = fem.SolidBodyNearlyIncompressible(
        umat=umat, field=u, bulk=5000, state=fem.StateNearlyIncompressible(u)
    )

    for parallel in [False, True]:
        kwargs = {"parallel": parallel}

        r1 = b.assemble.vector(u, **kwargs)
        assert r1.shape == (18, 1)

        r2 = b.assemble.vector(**kwargs)
        assert np.allclose(r1.toarray(), r2.toarray())

        K1 = b.assemble.matrix(u, **kwargs)
        assert K1.shape == (18, 18)

        K2 = b.assemble.matrix(**kwargs)
        assert np.allclose(K1.toarray(), K2.toarray())

        P1 = b.results.stress
        P2 = b.evaluate.gradient()
        P2 = b.evaluate.gradient(u)
        assert np.allclose(P1, P2)

        A1 = b.results.elasticity
        A2 = b.evaluate.hessian()
        A2 = b.evaluate.hessian(u)
        assert np.allclose(A1, A2)

        F1 = b.results.kinematics
        F2 = b._extract(u)
        assert np.allclose(F1, F2)

        s1 = b.evaluate.cauchy_stress()
        s2 = b.evaluate.cauchy_stress(u)
        assert np.allclose(s1, s2)

        t1 = b.evaluate.kirchhoff_stress()
        t2 = b.evaluate.kirchhoff_stress(u)
        assert np.allclose(t1, t2)


def test_solidbody_axi():
    umat, u = pre_axi()
    b = fem.SolidBody(umat=umat, field=u)

    for parallel in [False, True]:
        kwargs = {"parallel": parallel}

        r1 = b.assemble.vector(u, **kwargs)
        assert r1.shape == (18, 1)

        r2 = b.assemble.vector(**kwargs)
        assert np.allclose(r1.toarray(), r2.toarray())

        r3 = b.assemble.vector(**kwargs, block=False, apply=sum)
        assert np.allclose(r1.toarray(), r3.toarray())

        K1 = b.assemble.matrix(u, **kwargs)
        assert K1.shape == (18, 18)

        K2 = b.assemble.matrix(**kwargs)
        assert np.allclose(K1.toarray(), K2.toarray())

        K3 = b.assemble.matrix(**kwargs, block=False, apply=sum)
        assert np.allclose(K1.toarray(), K3.toarray())

        P1 = b.results.stress
        P2 = b.evaluate.gradient()
        P2 = b.evaluate.gradient(u)
        assert np.allclose(P1, P2)

        A1 = b.results.elasticity
        A2 = b.evaluate.hessian()
        A2 = b.evaluate.hessian(u)
        assert np.allclose(A1, A2)

        F1 = b.results.kinematics
        F2 = b._extract(u)
        assert np.allclose(F1, F2)

        s1 = b.evaluate.cauchy_stress()
        s2 = b.evaluate.cauchy_stress(u)
        assert np.allclose(s1, s2)

        t1 = b.evaluate.kirchhoff_stress()
        t2 = b.evaluate.kirchhoff_stress(u)
        assert np.allclose(t1, t2)


def test_solidbody_planestrain():
    umat, u = pre_planestrain(bulk=None)
    b = fem.SolidBodyNearlyIncompressible(umat=umat, field=u, bulk=5000)
    b = fem.SolidBodyNearlyIncompressible(
        umat=umat, field=u, bulk=5000, state=fem.StateNearlyIncompressible(u)
    )

    for parallel in [False, True]:
        kwargs = {"parallel": parallel}

        r1 = b.assemble.vector(u, **kwargs)
        assert r1.shape == (18, 1)

        r2 = b.assemble.vector(**kwargs)
        assert np.allclose(r1.toarray(), r2.toarray())

        K1 = b.assemble.matrix(u, **kwargs)
        assert K1.shape == (18, 18)

        K2 = b.assemble.matrix(**kwargs)
        assert np.allclose(K1.toarray(), K2.toarray())

        P1 = b.results.stress
        P2 = b.evaluate.gradient()
        P2 = b.evaluate.gradient(u)
        assert np.allclose(P1, P2)

        A1 = b.results.elasticity
        A2 = b.evaluate.hessian()
        A2 = b.evaluate.hessian(u)
        assert np.allclose(A1, A2)

        F1 = b.results.kinematics
        F2 = b._extract(u)
        assert np.allclose(F1, F2)

        s1 = b.evaluate.cauchy_stress()
        s2 = b.evaluate.cauchy_stress(u)
        assert np.allclose(s1, s2)

        t1 = b.evaluate.kirchhoff_stress()
        t2 = b.evaluate.kirchhoff_stress(u)
        assert np.allclose(t1, t2)


def test_solidbody_mixed():
    umat, u = pre_mixed(dim=3)
    b = fem.SolidBody(umat=umat, field=u)
    g = fem.SolidBodyForce(field=u, values=[9810, 0, 0], scale=7.85e-9)

    g.assemble.vector()

    for parallel in [False, True]:
        kwargs = {"parallel": parallel}

        r1 = b.assemble.vector(u, **kwargs)
        r1 = b.assemble.vector(u, items=3, **kwargs)
        assert r1.shape == (97, 1)

        r2 = b.assemble.vector(**kwargs)
        assert np.allclose(r1.toarray(), r2.toarray())

        K1 = b.assemble.matrix(u, **kwargs)
        K1 = b.assemble.matrix(u, items=6, **kwargs)
        assert K1.shape == (97, 97)

        K2 = b.assemble.matrix(**kwargs)
        assert np.allclose(K1.toarray(), K2.toarray())

        P1 = b.results.stress
        P2 = b.evaluate.gradient()
        P2 = b.evaluate.gradient(u)
        for p1, p2 in zip(P1, P2):
            assert np.allclose(p1, p2)

        A1 = b.results.elasticity
        A2 = b.evaluate.hessian()
        A2 = b.evaluate.hessian(u)
        for a1, a2 in zip(A1, A2):
            if a1 is None:
                assert a2 is None
            else:
                assert np.allclose(a1, a2)

        F1 = b.results.kinematics
        F2 = b._extract(u)
        for f1, f2 in zip(F1, F2):
            assert np.allclose(f1, f2)

        s1 = b.evaluate.cauchy_stress()
        s2 = b.evaluate.cauchy_stress(u)
        assert np.allclose(s1, s2)

        t1 = b.evaluate.kirchhoff_stress()
        t2 = b.evaluate.kirchhoff_stress(u)
        assert np.allclose(t1, t2)

        rg1 = g.assemble.vector(u, **kwargs)
        assert rg1.shape == (97, 1)

        Kg1 = g.assemble.matrix(u, **kwargs)
        assert Kg1.shape == (97, 97)

        rg2 = g.assemble.vector(**kwargs)
        assert rg1.shape == (97, 1)
        assert np.allclose(rg1.toarray(), rg2.toarray())

        Kg2 = g.assemble.matrix(**kwargs)
        assert Kg1.shape == (97, 97)
        assert np.allclose(Kg1.toarray(), Kg2.toarray())


def test_load():
    for axi in [False, True]:
        if axi:
            umat, field = pre(dim=3)
        else:
            umat, field = pre_axi()

        mask = field.region.mesh.points[:, 0] == 1
        values = 0.1

        if axi:
            values *= 0.025

        body = fem.SolidBody(umat, field)

        load = fem.PointLoad(field, mask, values=values, axisymmetric=axi)
        bounds = {"fix": fem.Boundary(field[0], fx=lambda x: x == 0)}
        dof0, dof1 = fem.dof.partition(field, bounds)

        res = fem.newtonrhapson(items=[body, load], dof0=dof0, dof1=dof1)

        assert res.success

    plotter = load.plot(plotter=field.region.mesh.plot(off_screen=True))

    load.update([0, 0.1])
    plotter = load.plot(plotter=field.region.mesh.plot(off_screen=True))


def test_view():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)

    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    umat = fem.NeoHooke(mu=1, bulk=2)
    solid = fem.SolidBody(umat, field)
    plotter = solid.plot("Principal Values of Cauchy Stress", off_screen=True)
    # img = solid.screenshot(transparent_background=True)
    # ax = solid.imshow()

    field = fem.FieldContainer([fem.Field(region, dim=2)])
    umat = fem.LinearElasticPlaneStress(E=1, nu=0.3)

    solid = fem.SolidBody(umat, field)
    plotter = solid.plot("Equivalent of Stress", off_screen=True)

    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=0)
    plotter = solid.plot("Kirchhoff Stress", off_screen=True)


def test_threefield():
    field = fem.FieldsMixed(fem.RegionHexahedron(fem.Cube(n=3)), n=3)
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
    umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
    solid = fem.SolidBody(umat, field)
    step = fem.Step(items=[solid], boundaries=boundaries)
    job = fem.Job(steps=[step]).evaluate()
    assert np.isclose(job.fnorms[0][-1], 0)

    field = fem.FieldsMixed(fem.RegionHexahedron(fem.Cube(n=3)), n=3)
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
    umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    solid = fem.SolidBody(umat, field)
    step = fem.Step(items=[solid], boundaries=boundaries)
    job = fem.Job(steps=[step]).evaluate()
    assert np.isclose(job.fnorms[0][-1], 0)


def test_solidbody_cauchy_stress():
    field = fem.FieldsMixed(fem.RegionHexahedron(fem.Cube(n=3)), n=3)
    region_stress = fem.RegionHexahedronBoundary(
        mesh=field.region.mesh,
        only_surface=True,
        mask=field.region.mesh.x == 1,
    )
    field_boundary = fem.FieldContainer([fem.Field(region_stress, dim=3)])
    boundaries = dict(left=fem.Boundary(field[0], fx=0))
    umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000)
    solid = fem.SolidBody(umat, field)

    resize_matrix = csr_matrix(([0.0], ([0], [0])), shape=(100, 100))
    resize_vector = csr_matrix(([0.0], ([0], [0])), shape=(100, 1))

    for cauchy_stress in [None, np.zeros((3, 3))]:
        stress = fem.SolidBodyCauchyStress(
            field=field_boundary, cauchy_stress=cauchy_stress
        )
        matrix = stress.assemble.matrix(field, resize=resize_matrix)
        vector = stress.assemble.vector(field, resize=resize_vector)
        stress._update(other_field=field, field=field_boundary)

        assert matrix.shape == (100, 100)
        assert vector.shape == (100, 1)

    table = (
        fem.math.linsteps([1], num=0, axis=2, axes=9)
        + fem.math.linsteps([1], num=0, axis=6, axes=9)
    ).reshape(-1, 3, 3)
    step = fem.Step(
        items=[solid, stress], ramp={stress: 1 * table}, boundaries=boundaries
    )
    fem.Job(steps=[step]).evaluate()

    assert np.isclose(field[0].values.max(), 0.971866)


def test_truss():

    mesh = fem.Mesh(
        points=[[0, 0], [1, 1], [2.0, 0]], cells=[[0, 1], [1, 2]], cell_type="line"
    )
    region = fem.RegionTruss(mesh)
    field = fem.Field(region, dim=2).as_container()
    boundaries = fem.BoundaryDict(fixed=fem.Boundary(field[0], fy=0))

    umat = fem.LinearElastic1D(E=np.ones(2))
    truss = fem.TrussBody(umat, field, area=np.ones(2))
    load = fem.PointLoad(field, [1])

    move = fem.math.linsteps([0, -0.1], num=5, axis=1, axes=2)
    step = fem.Step(items=[truss, load], ramp={load: move}, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate()

    assert np.isclose(field[0].values[1, 1], -0.16302376)

    hess = truss.evaluate.hessian(field)
    fem.TrussBody(umat, field, area=[1, 1], statevars=[0, 0])


if __name__ == "__main__":
    test_simple()
    test_solidbody()
    test_solidbody_cauchy_stress()
    test_solidbody_incompressible()
    test_solidbody_axi()
    test_solidbody_axi_incompressible()
    test_solidbody_planestrain()
    test_solidbody_mixed()
    test_pressure()
    test_load()
    test_view()
    test_threefield()
    test_truss
