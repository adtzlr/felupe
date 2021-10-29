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


def pre():

    m = fe.Cube(n=3)
    e = fe.Hexahedron()
    q = fe.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)

    u = fe.Field(r, dim=3)
    p = fe.Field(r)
    J = fe.Field(r, values=1)

    return r, (u, p, J)


def test_solve_check():

    r, (u, p, J) = pre()

    W = fe.constitution.NeoHooke(1, 3)

    F = u.extract()

    bounds = fe.dof.symmetry(u)
    dof0, dof1 = fe.dof.partition(u, bounds)

    u0ext = fe.dof.apply(u, bounds, dof0)

    L = fe.IntegralForm(W.gradient(F), u, r.dV, grad_v=True)
    a = fe.IntegralForm(W.hessian(F), u, r.dV, u, True, True)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fe.tools.solve(A, b, u, dof0, dof1, u0ext)
    assert dx.shape == u.values.ravel().shape

    fe.tools.check(dx, u, b, dof1, dof0, verbose=0)
    fe.tools.check(dx, u, b, dof1, dof0, verbose=1)

    fe.tools.save(r, u)
    fe.tools.save(r, u, r=b)
    fe.tools.save(
        r,
        u,
        r=b,
        F=F,
        gradient=W.gradient(F),
    )

    force = fe.tools.force(u, b, bounds["symx"])
    moment = fe.tools.moment(u, b, bounds["symx"])

    for a in [2, 3, 4, 5]:
        curve = fe.tools.curve(np.arange(a), np.ones(a) * force[0])

    s = fe.math.dot(W.gradient(F), fe.math.inv(fe.math.cof(F)))

    cauchy = fe.tools.project(fe.math.tovoigt(s), region=r, average=True)
    assert cauchy.shape == (r.mesh.npoints, 6)

    cauchy = fe.tools.project(fe.math.tovoigt(s), region=r, average=False)
    assert cauchy.shape == (r.mesh.cells.size, 6)


def test_solve_mixed_check():

    r, fields = pre()
    u = fields[0]

    f = fe.FieldMixed(fields)

    F, p, J = f.extract()

    W = fe.constitution.NeoHooke(1, 3)
    W_mixed = fe.constitution.Mixed(W.gradient, W.hessian)

    F = u.extract()

    bounds = fe.dof.symmetry(u)
    dof0, dof1, unstack = fe.dof.partition(f, bounds)

    u0ext = fe.dof.apply(u, bounds, dof0)

    L = fe.IntegralFormMixed(W_mixed.gradient(F, p, J), f, r.dV)
    a = fe.IntegralFormMixed(W_mixed.hessian(F, p, J), f, r.dV, f)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    dx = fe.tools.solve(A, b, f, dof0, dof1, u0ext, unstack)

    assert dx[0].shape == u.values.ravel().shape
    assert dx[1].shape == fields[1].values.ravel().shape
    assert dx[2].shape == fields[2].values.ravel().shape

    fe.tools.check(dx, f, b, dof1, dof0, verbose=0)
    fe.tools.check(dx, f, b, dof1, dof0, verbose=1)

    fe.tools.save(r, f, unstack=unstack)
    fe.tools.save(r, f, r=b, unstack=unstack)
    fe.tools.save(
        r,
        f,
        r=b,
        unstack=unstack,
        F=F,
        gradient=W_mixed.gradient(F, p, J),
    )

    force = fe.tools.force(u, b, bounds["symx"], unstack=unstack)
    moment = fe.tools.moment(u, b, bounds["symx"], unstack=unstack)

    for a in [2, 3, 4, 5]:
        curve = fe.tools.curve(np.arange(a), np.ones(a) * force[0])


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
    test_solve_check()
    test_solve_mixed_check()
    test_newton()
