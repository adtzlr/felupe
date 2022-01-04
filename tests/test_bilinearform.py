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

    m = fe.Cube(n=3)
    r = fe.RegionHexahedron(m)
    u = fe.Field(r, dim=dim)
    return r, u


def pre_mixed(dim):

    mesh = fe.Cube(n=3)
    region = fe.RegionHexahedron(mesh)
    region0 = fe.RegionConstantHexahedron(mesh)
    u = fe.Field(region, dim=3)
    p = fe.Field(region0)
    up = fe.FieldMixed((u, p))

    return up


def test_linearform():

    region, field = pre(dim=3)
    b = fe.Basis(field)

    w = field.extract(grad=False)
    F = field.extract(grad=True, sym=False, add_identity=True)

    r1 = fe.LinearForm(v=b, grad_v=False).assemble(lform, w=w)
    r2 = fe.LinearForm(v=b, grad_v=True).assemble(lform_grad, F=F)

    assert r1.shape == (81, 1)
    assert r2.shape == (81, 1)


def test_bilinearform():

    region, field = pre(dim=3)
    b = fe.Basis(field)

    w = field.extract(grad=False)
    F = field.extract(grad=True, sym=False, add_identity=True)

    K1 = fe.BilinearForm(v=b, grad_v=False, u=b, grad_u=False).assemble(bform, w=w)
    K2 = fe.BilinearForm(v=b, grad_v=True, u=b, grad_u=True).assemble(bform_grad, F=F)

    assert K1.shape == (81, 81)
    assert K2.shape == (81, 81)


def test_linearform_mixed():

    field = pre_mixed(dim=3)
    b = fe.BasisMixed(field)

    F, p = field.extract()
    r = fe.LinearFormMixed(v=b).assemble((lformu, lformp), F=F, p=p)
    r = fe.LinearFormMixed(v=b, grad_v=(True, False)).assemble(
        (lformu, lformp), F=F, p=p
    )
    assert r.shape == (89, 1)


def test_bilinearform_mixed():

    field = pre_mixed(dim=3)
    b = fe.BasisMixed(field)

    F, p = field.extract()
    K = fe.BilinearFormMixed(v=b, u=b).assemble((a_uu, a_up, a_pp), F=F, p=p)
    K = fe.BilinearFormMixed(
        v=b, u=b, grad_v=(True, False), grad_u=(True, False)
    ).assemble((a_uu, a_up, a_pp), F=F, p=p)
    assert K.shape == (89, 89)


if __name__ == "__main__":
    test_linearform()
    test_bilinearform()
    test_linearform_mixed()
    test_bilinearform_mixed()
