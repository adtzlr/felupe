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

import felupe as fem


def test_solve():
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3)
    v = fem.FieldContainer([u])

    W = fem.constitution.NeoHooke(1, 3)

    F = v.extract()
    P = W.gradient(F)[:-1]
    A = W.hessian(F)

    b = fem.dof.symmetry(u)
    dof0, dof1 = fem.dof.partition(v, b)

    ext0 = fem.dof.apply(v, b, dof0)

    L = fem.IntegralForm(P, v, r.dV)
    a = fem.IntegralForm(A, v, r.dV, v)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    system = fem.solve.partition(v, A, dof1, dof0)

    du = fem.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fem.solve.solve(*system, ext0)
    assert np.allclose(du, 0)

    system = fem.solve.partition(v, A, dof1, dof0, b)

    du = fem.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fem.solve.solve(*system, ext0)
    assert np.allclose(du, 0)


if __name__ == "__main__":
    test_solve()
