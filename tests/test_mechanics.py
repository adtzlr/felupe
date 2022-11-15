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

import pytest
import felupe as fe
import numpy as np


def test_simple():

    umat = fe.LinearElastic(E=1, nu=0.3)

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    u = fe.FieldContainer([fe.Field(r, dim=3)])

    b = fe.SolidBody(umat, u)
    r = b.assemble.vector()

    K = b.assemble.matrix()
    r = b.assemble.vector(u)
    F = b.results.kinematics
    P = b.results.stress
    s = b.evaluate.cauchy_stress()
    t = b.evaluate.kirchhoff_stress()
    C = b.results.elasticity

    assert K.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F[0].shape == (3, 3, 8, 8)
    assert P[0].shape == (3, 3, 8, 8)
    assert s.shape == (3, 3, 8, 8)
    assert t.shape == (3, 3, 8, 8)
    assert C[0].shape == (3, 3, 3, 3, 1, 1)


def test_pressure():

    umat = fe.LinearElastic(E=1, nu=0.3)

    m = fe.Cube(n=3)
    h = fe.RegionHexahedron(m)
    u = fe.Field(h, dim=3)

    np.random.seed(156)
    u.values = np.random.rand(*u.values.shape) / 10
    v = fe.FieldContainer([u])

    s = fe.RegionHexahedronBoundary(m)
    p = fe.Field(s, dim=3)
    q = fe.FieldContainer([p])

    b = fe.SolidBody(umat, v)
    z = fe.SolidBodyPressure(q)
    c = fe.SolidBodyPressure(q, pressure=1.0)

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
    w = fe.FieldsMixed(h)
    w[0].values = np.random.rand(*w[0].values.shape) / 10

    c._update(w, v)
    assert np.allclose(w[0].values, v[0].values)


def pre(dim, bulk=2):

    umat = fe.NeoHooke(mu=1, bulk=bulk)

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    u = fe.Field(r, dim=dim)

    np.random.seed(156)
    u.values = np.random.rand(*u.values.shape) / 10

    return umat, fe.FieldContainer([u])


def pre_axi(bulk=2):

    umat = fe.NeoHooke(mu=1, bulk=bulk)

    m = fe.Rectangle(n=3)
    r = fe.RegionQuad(m)
    u = fe.FieldAxisymmetric(r)

    np.random.seed(156)
    u.values = np.random.rand(*u.values.shape) / 10

    return umat, fe.FieldContainer([u])


def pre_mixed(dim):

    umat = fe.ThreeFieldVariation(fe.NeoHooke(mu=1, bulk=2))

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    u = fe.FieldsMixed(r, n=3)

    np.random.seed(156)
    u[0].values = np.random.rand(*u[0].values.shape) / 10
    u[1].values = np.random.rand(*u[1].values.shape) / 10
    u[2].values = np.random.rand(*u[2].values.shape) / 10

    return umat, u


def test_solidbody():

    umat, u = pre(dim=3)
    b = fe.SolidBody(umat=umat, field=u, statevars=np.ones(5))

    for parallel in [False, True]:
        for jit in [False, True]:

            kwargs = {"parallel": parallel, "jit": jit}

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
    b = fe.SolidBodyNearlyIncompressible(
        umat=umat, field=u, bulk=5000, statevars=np.ones(5)
    )

    umat = fe.OgdenRoxburgh(fe.NeoHooke(mu=1), r=3, m=1, beta=0)
    b = fe.SolidBodyNearlyIncompressible(
        umat=umat, field=u, bulk=5000, state=fe.StateNearlyIncompressible(u)
    )

    for parallel in [False, True]:
        for jit in [False, True]:

            kwargs = {"parallel": parallel, "jit": jit}

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


def test_solidbody_axi():

    umat, u = pre_axi(bulk=None)
    b = fe.SolidBodyNearlyIncompressible(umat=umat, field=u, bulk=5000)
    b = fe.SolidBodyNearlyIncompressible(
        umat=umat, field=u, bulk=5000, state=fe.StateNearlyIncompressible(u)
    )

    for parallel in [False, True]:
        for jit in [False, True]:

            kwargs = {"parallel": parallel, "jit": jit}

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


def test_solidbody_axi_incompressible():

    umat, u = pre_axi()
    b = fe.SolidBody(umat=umat, field=u)

    for parallel in [False, True]:
        for jit in [False, True]:

            kwargs = {"parallel": parallel, "jit": jit}

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
    b = fe.SolidBody(umat=umat, field=u)
    g = fe.SolidBodyGravity(field=u, gravity=[9810, 0, 0], density=7.85e-9)

    g.assemble.vector()

    for parallel in [False, True]:
        for jit in [False, True]:

            kwargs = {"parallel": parallel, "jit": jit}

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

        body = fe.SolidBody(umat, field)
        load = fe.PointLoad(field, mask, values=values, axisymmetric=axi)

        bounds = {"fix": fe.Boundary(field[0], fx=lambda x: x == 0)}
        dof0, dof1 = fe.dof.partition(field, bounds)

        res = fe.newtonrhapson(items=[body, load], dof0=dof0, dof1=dof1)

        assert res.success


if __name__ == "__main__":
    test_simple()
    test_solidbody()
    test_solidbody_incompressible()
    test_solidbody_axi()
    test_solidbody_mixed()
    test_pressure()
    test_load()
