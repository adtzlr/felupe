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


def pre():
    m = fem.Cube(n=3)
    r = fem.RegionHexahedron(m)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)

    v = fem.FieldContainer([u])
    q = fem.FieldContainer([p])

    W = fem.constitution.NeoHooke(1, 3)

    F = v.extract(grad=True, add_identity=True)
    P = W.gradient(F)[:-1]
    A = W.hessian(F)

    return r, v, q, P, A


def pre_broadcast():
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)

    v = fem.FieldContainer([u])
    q = fem.FieldContainer([p])

    W = fem.constitution.LinearElastic(E=1.0, nu=0.3)

    F = v.extract(grad=True, add_identity=True)
    P = W.gradient(F)[:-1]
    A = W.hessian()

    P = [P[0][:, :, 0, 0].reshape(3, 3, 1, 1)]

    return r, v, q, P, A


def pre_axi():
    m = fem.Rectangle(n=3)
    r = fem.RegionQuad(m)

    u = fem.FieldAxisymmetric(r)
    v = fem.FieldContainer([u])

    W = fem.constitution.NeoHooke(1, 3)

    F = v.extract(grad=True, add_identity=True)
    P = W.gradient(F)[:-1]
    A = W.hessian(F)

    return r, v, P, A


def pre_mixed():
    m = fem.mesh.Cube(n=3)
    e = fem.element.Hexahedron()
    q = fem.quadrature.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)
    J = fem.Field(r, values=1)

    f = fem.FieldContainer((u, p, J))

    nh = fem.NeoHooke(1, 3)
    W = fem.ThreeFieldVariation(nh)

    return r, f, W.gradient(f.extract())[:-1], W.hessian(f.extract())


def pre_axi_mixed():
    m = fem.mesh.Rectangle(n=3)
    e = fem.element.Quad()
    q = fem.quadrature.GaussLegendre(1, 2)
    r = fem.Region(m, e, q)

    u = fem.FieldAxisymmetric(r, dim=2)
    p = fem.Field(r)
    J = fem.Field(r, values=1)

    f = fem.FieldContainer((u, p, J))

    nh = fem.NeoHooke(1, 3)
    W = fem.ThreeFieldVariation(nh)

    return r, f, W.gradient(f.extract())[:-1], W.hessian(f.extract())


def test_axi():
    r, u, P, A = pre_axi()

    for parallel in [False, True]:
        L = fem.IntegralForm(P, u, r.dV)
        x = L.integrate(parallel=parallel)

        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        a = fem.IntegralForm(A, u, r.dV, u, grad_v=[True], grad_u=[True])
        y = a.integrate(parallel=parallel)

        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)


def test_linearform():
    r, u, p, P, A = pre()

    for parallel in [False, True]:
        L = fem.IntegralForm(P, u, r.dV, grad_v=[True])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        L = fem.IntegralForm(p.extract(grad=False), p, r.dV, grad_v=[False])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)


def test_linearform_broadcast():
    r, u, p, P, A = pre_broadcast()

    for parallel in [False, True]:
        L = fem.IntegralForm(P, u, r.dV, grad_v=[True])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.ndof, 1)

        L = fem.IntegralForm(p.extract(grad=False), p, r.dV, grad_v=[False])
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)
        b = L.assemble(parallel=parallel).toarray()
        assert b.shape == (r.mesh.npoints, 1)


def test_bilinearform():
    r, u, p, P, A = pre()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, u, r.dV, u)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fem.IntegralForm(P, u, r.dV, p, [True], [False])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)


def test_bilinearform_broadcast():
    r, u, p, P, A = pre_broadcast()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, u, r.dV, u, [True], [True])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.ndof)

        a = fem.IntegralForm(P, u, r.dV, p, [True], [False])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.ndof, r.mesh.npoints)

        q = p.extract(grad=False)
        f = fem.math.dya(q[0], q[0], mode=1)
        a = fem.IntegralForm(f, p, r.dV, p, [False], [False])
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        assert K.shape == (r.mesh.npoints, r.mesh.npoints)
        K = a.assemble(parallel=parallel).toarray()
        assert K.shape == (r.mesh.npoints, r.mesh.npoints)


def test_mixed():
    r, v, f, A = pre_mixed()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, v, r.dV, v)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        K = a.assemble(parallel=parallel).toarray()

        z = r.mesh.ndof + 2 * r.mesh.npoints
        assert K.shape == (z, z)

        L = fem.IntegralForm(f, v, r.dV)
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        b = L.assemble(parallel=parallel).toarray()

        assert b.shape == (z, 1)

    r, v, f, A = pre_axi_mixed()

    for parallel in [False, True]:
        a = fem.IntegralForm(A, v, r.dV, v)
        y = a.integrate(parallel=parallel)
        K = a.assemble(y, parallel=parallel).toarray()
        K = a.assemble(parallel=parallel).toarray()

        z = r.mesh.ndof + 2 * r.mesh.npoints
        assert K.shape == (z, z)

        L = fem.IntegralForm(f, v, r.dV)
        x = L.integrate(parallel=parallel)
        b = L.assemble(x, parallel=parallel).toarray()
        b = L.assemble(parallel=parallel).toarray()

        assert b.shape == (z, 1)


if __name__ == "__main__":
    test_linearform()
    test_linearform_broadcast()
    test_bilinearform()
    test_bilinearform_broadcast()
    test_axi()
    test_mixed()
