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

import felupe as fem
from felupe.math import ddot, dot, dya


def lform(v, w):
    return dot(w, v)


def lform_grad(v, F):
    return ddot(F, v)


def lformu(v, F, p):
    return ddot(F, v)


def lformp(q, F, p):
    return dot(p, q)


def bform(v, u, w):
    return dot(w, v) * dot(w, u)


def bform_grad(v, u, F):
    return ddot(F, v) * ddot(F, u)


def a_uu(v, u, F, p):
    return ddot(u, ddot(dya(F, F), v))


def a_up(v, r, F, p):
    return dot(p, r) * ddot(F, v)


def a_pp(q, r, F, p):
    return dot(p, q) * dot(r, p)


def pre(dim):
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    region0 = fem.RegionConstantHexahedron(mesh.dual(points_per_cell=1))
    u = fem.Field(region, dim=3)
    p = fem.Field(region0)
    up = fem.FieldContainer([u, p])

    return up


def test_form_decorator():
    field = pre(dim=3)
    F, p = field.extract()

    @fem.Form(v=field, u=field, grad_v=(True, False), grad_u=(True, False))
    def a():
        return (a_uu, a_up, a_pp)

    a.assemble(field, field, args=(F, p))

    @fem.Form(v=field, u=field, grad_v=None, grad_u=None)
    def a():
        return (a_uu, a_up, a_pp)

    a.assemble(field, field, args=(F, p))

    @fem.Form(v=field, grad_v=(True, False))
    def L():
        return (lformu, lformp)

    L.assemble(field, args=(F, p), parallel=False)
    L.assemble(field, args=(F, p), parallel=True)
    L.assemble(field, args=(F, p), parallel=False, sym=True)
    L.assemble(field, args=(F, p), parallel=True, sym=True)

    @fem.Form(v=field, grad_v=None)
    def L():
        return (lformu, lformp)

    L.integrate(field, args=(F, p), parallel=False)
    L.integrate(field, args=(F, p), parallel=True)
    L.integrate(field, args=(F, p), parallel=False, sym=True)
    L.integrate(field, args=(F, p), parallel=True, sym=True)

    L.assemble(field, args=(F, p), parallel=False)
    L.assemble(field, args=(F, p), parallel=True)
    L.assemble(field, args=(F, p), parallel=False, sym=True)
    L.assemble(field, args=(F, p), parallel=True, sym=True)


def test_linear_elastic():
    mesh = fem.Cube(n=11)
    region = fem.RegionHexahedron(mesh)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])

    from felupe.math import ddot, sym, trace

    @fem.Form(
        v=field, u=field, grad_v=[True], grad_u=[True], kwargs={"mu": 1.0, "lmbda": 2.0}
    )
    def bilinearform():
        "A container for a bilinear form."

        def linear_elasticity(gradv, gradu, mu, lmbda):
            "Linear elasticity."

            de, e = sym(gradv), sym(gradu)
            return 2 * mu * ddot(de, e) + lmbda * trace(de) * trace(e)

        return [
            linear_elasticity,
        ]

    bilinearform.integrate(v=field, u=field, parallel=False)
    bilinearform.integrate(v=field, u=field, parallel=True)
    bilinearform.integrate(v=field, u=field, parallel=False, sym=True)
    bilinearform.integrate(v=field, u=field, parallel=True, sym=True)

    bilinearform.assemble(v=field, u=field, parallel=False)
    bilinearform.assemble(v=field, u=field, parallel=True)
    bilinearform.assemble(v=field, u=field, parallel=False, sym=True)
    bilinearform.assemble(v=field, u=field, parallel=True, sym=True)


if __name__ == "__main__":
    test_form_decorator()
    test_linear_elastic()
