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
    u = fe.Field(r, dim=3)

    b = fe.SolidBody(umat, u)
    r = b.assemble.vector()

    K = b.assemble.matrix()
    r = b.assemble.vector(u)
    F = b.results.kinematics[0]
    P = b.results.stress
    s = b.evaluate.cauchy_stress()
    C = b.results.elasticity

    assert K.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F.shape == (3, 3, 8, 8)
    assert P.shape == (3, 3, 8, 8)
    assert s.shape == (3, 3, 8, 8)
    assert C.shape == (3, 3, 3, 3, 8, 8)


def test_pressure():

    umat = fe.LinearElastic(E=1, nu=0.3)

    m = fe.Cube(n=3)
    h = fe.RegionHexahedron(m)
    u = fe.Field(h, dim=3)

    u.values = np.random.rand(*u.values.shape) / 10

    s = fe.RegionHexahedronBoundary(m)
    v = fe.Field(s, dim=3)

    b = fe.SolidBody(umat, u)
    c = fe.SolidBodyPressure(v)

    r = b.assemble.vector()
    K = b.assemble.matrix()
    r = b.assemble.vector(u)
    F = b.results.kinematics[0]
    s = b.results.stress
    C = b.results.elasticity

    assert K.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F.shape == (3, 3, 8, 8)
    assert s.shape == (3, 3, 8, 8)
    assert C.shape == (3, 3, 3, 3, 8, 8)

    r = c.assemble.vector()
    K = c.assemble.matrix()
    K = c.assemble.matrix(v, resize=b.assemble.matrix())
    r = c.assemble.vector(v)
    r = c.assemble.vector(v, resize=b.assemble.vector())
    F = c.results.kinematics[0]

    assert K.shape == (81, 81)
    assert r.shape == (81, 1)
    assert F.shape == (3, 3, 4, 24)

    c.update(u, v)
    assert np.allclose(u.values, v.values)

    w = fe.FieldsMixed(h)
    w[0].values = np.random.rand(*w[0].values.shape) / 10

    c.update(w, v)
    assert np.allclose(w[0].values, v.values)


def pre(dim):

    umat = fe.NeoHooke(mu=1, bulk=2)

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    u = fe.Field(r, dim=dim)

    u.values = np.random.rand(*u.values.shape) / 10

    return umat, u


def pre_axi():

    umat = fe.NeoHooke(mu=1, bulk=2)

    m = fe.Rectangle(n=3)
    r = fe.RegionQuad(m)
    u = fe.FieldAxisymmetric(r)

    u.values = np.random.rand(*u.values.shape) / 10

    return umat, u


def pre_mixed(dim):

    umat = fe.ThreeFieldVariation(fe.NeoHooke(mu=1, bulk=2))

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    u = fe.FieldsMixed(r, n=3)

    u[0].values = np.random.rand(*u[0].values.shape) / 10
    u[1].values = np.random.rand(*u[1].values.shape) / 10
    u[2].values = np.random.rand(*u[2].values.shape) / 10

    return umat, u


def test_solidbody():

    umat, u = pre(dim=3)
    b = fe.SolidBody(umat=umat, field=u)

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


def test_solidbody_axi():

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


def test_solidbody_mixed():

    umat, u = pre_mixed(dim=3)
    b = fe.SolidBody(umat=umat, field=u)

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


if __name__ == "__main__":
    test_simple()
    test_solidbody()
    test_solidbody_axi()
    test_solidbody_mixed()
    test_pressure()
