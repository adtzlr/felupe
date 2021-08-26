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
import numpy as np
import felupe as fe


def pre():

    m = fe.mesh.Cube(n=3)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)

    u = fe.Field(r, dim=3)
    p = fe.Field(r)
    J = fe.Field(r, values=1)

    return r, (u, p, J)


def test_extract():

    r, fields = pre()
    F, p, J = fe.tools.FpJ(fields)

    assert F.shape == (3, 3, r.quadrature.npoints, r.mesh.ncells)
    assert p.shape == (1, r.quadrature.npoints, r.mesh.ncells)
    assert J.shape == (1, r.quadrature.npoints, r.mesh.ncells)


def test_defgrad():

    r, fields = pre()
    F = fe.tools.defgrad(fields[0])

    assert F.shape == (3, 3, r.quadrature.npoints, r.mesh.ncells)


def test_strain():

    r, fields = pre()
    eps = fe.tools.strain(fields[0])

    assert eps.shape == (3, 3, r.quadrature.npoints, r.mesh.ncells)


def test_strain_voigt():

    m = fe.mesh.Line(n=3)
    e = fe.element.Line()
    q = fe.quadrature.GaussLegendre(1, 1)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=1)
    strain = fe.tools.strain_voigt(u)
    assert strain.shape == (1, r.quadrature.npoints, r.mesh.ncells)

    m = fe.mesh.Rectangle(n=3)
    e = fe.element.Quad()
    q = fe.quadrature.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=2)
    strain = fe.tools.strain_voigt(u)
    assert strain.shape == (3, r.quadrature.npoints, r.mesh.ncells)

    m = fe.mesh.Cube(n=3)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    strain = fe.tools.strain_voigt(u)
    assert strain.shape == (6, r.quadrature.npoints, r.mesh.ncells)


def test_solve_check():

    r, (u, p, J) = pre()

    nh = fe.constitution.models.NeoHooke(1, 3)

    F = fe.tools.defgrad(u)

    b = fe.doftools.symmetry(u)
    dof0, dof1 = fe.doftools.partition(u, b)

    u0ext = fe.doftools.apply(u, b, dof0)

    L = fe.IntegralForm(nh.P(F), u, r.dV, grad_v=True)
    a = fe.IntegralForm(nh.A(F), u, r.dV, u, True, True)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fe.tools.solve(A, b, u, dof0, dof1, u0ext)
    assert dx.shape == u.values.ravel().shape

    fe.tools.check(dx, u, b, dof1, dof0, verbose=0)


def test_solve_mixed_check():

    r, fields = pre()
    u = fields[0]

    F, p, J = fe.tools.FpJ(fields)

    NH = fe.constitution.models.NeoHooke(1, 3)
    nh = fe.constitution.variation.upJ(NH.P, NH.A)

    F = fe.tools.defgrad(u)

    b = fe.doftools.symmetry(u)
    dof0, dof1, unstack = fe.doftools.partition(fields, b)

    u0ext = fe.doftools.apply(u, b, dof0)

    L = fe.IntegralFormMixed(nh.f(F, p, J), fields, r.dV)
    a = fe.IntegralFormMixed(nh.A(F, p, J), fields, r.dV, fields)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fe.tools.solve_mixed(A, b, fields, dof0, dof1, u0ext, unstack)
    assert dx[0].shape == u.values.ravel().shape
    assert dx[1].shape == fields[1].values.ravel().shape
    assert dx[2].shape == fields[2].values.ravel().shape

    fe.tools.check_mixed(dx, fields, b, dof1, dof0, verbose=0)


def test_update():

    r, fields = pre()
    dx = [f.values for f in fields]
    fe.tools.update(fields, dx)


def test_newton():
    def fun(x):
        return (x - 3) ** 2

    def jac(x):
        return np.array([2 * (x - 3)])

    x0 = np.array([3.1])

    res = fe.tools.newtonrhapson(fun, x0, jac)

    assert abs(res.fun) < 1e-6
    assert np.isclose(res.x, 3, rtol=1e-2)


if __name__ == "__main__":
    test_extract()
    test_defgrad()
    test_strain()
    test_strain_voigt()
    test_update()
    test_solve_check()
    test_solve_mixed_check()
    test_newton()
