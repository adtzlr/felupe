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

import felupe as fe


def pre():

    m = fe.mesh.Cube(n=3)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)

    u = fe.Field(r, dim=3)
    p = fe.Field(r)

    nh = fe.constitution.models.NeoHooke(1, 3)

    F = fe.tools.defgrad(u)
    P = nh.P(F)
    A = nh.A(F)

    return r, u, p, P, A


def pre_axi():

    m = fe.mesh.Rectangle(n=3)
    e = fe.element.Quad()
    q = fe.quadrature.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)

    u = fe.FieldAxisymmetric(r)

    nh = fe.constitution.models.NeoHooke(1, 3)

    F = fe.tools.defgrad(u)
    P = nh.P(F)
    A = nh.A(F)

    return r, u, P, A


def pre_mixed():

    m = fe.mesh.Cube(n=3)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)

    u = fe.Field(r, dim=3)
    p = fe.Field(r)
    J = fe.Field(r, values=1)

    nh = fe.constitution.models.NeoHooke(1, 3)
    umat = fe.constitution.variation.upJ(nh.P, nh.A)

    FpJ = fe.tools.FpJ((u, p, J))
    f = umat.f(*FpJ)
    A = umat.A(*FpJ)

    return r, u, p, J, f, A


def test_axi():

    r, u, P, A = pre_axi()

    for parallel in [False, True]:

        L = fe.IntegralFormAxisymmetric(P, u, r.dV)
        x = L.integrate(parallel=parallel)

        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        a = fe.IntegralFormAxisymmetric(A, u, r.dV, u)
        y = a.integrate(parallel=parallel)

        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)


def test_linearform():

    r, u, p, P, A = pre()

    for parallel in [False, True]:

        L = fe.IntegralForm(P, u, r.dV, grad_v=True)
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        L = fe.IntegralForm(p.interpolate(), p, r.dV)
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)


def test_bilinearform():

    r, u, p, P, A = pre()

    for parallel in [False, True]:

        a = fe.IntegralForm(A, u, r.dV, u, True, True)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fe.IntegralForm(P, u, r.dV, p, True, False)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)


def test_mixed():

    r, u, p, J, f, A = pre_mixed()
    v = (u, p, J)

    for parallel in [False, True]:

        a = fe.IntegralFormMixed(A, v, r.dV, v)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        K = a.assemble(parallel=parallel).toarray()

        z = r.mesh.ndof + 2 * r.mesh.npoints
        assert K.shape == (z, z)

        L = fe.IntegralFormMixed(f, v, r.dV)
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        b = L.assemble(parallel=parallel).toarray()

        assert b.shape == (z, 1)


if __name__ == "__main__":
    test_linearform()
    test_bilinearform()
    test_axi()
    test_mixed()
