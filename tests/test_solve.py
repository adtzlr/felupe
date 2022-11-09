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


def test_solve():

    m = fe.Cube(n=3)
    e = fe.Hexahedron()
    q = fe.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    v = fe.FieldContainer([u])

    W = fe.constitution.NeoHooke(1, 3)

    F = v.extract()
    P = W.gradient(F)[:-1]
    A = W.hessian(F)

    b = fe.dof.symmetry(u)
    dof0, dof1 = fe.dof.partition(v, b)

    ext0 = fe.dof.apply(v, b, dof0)

    L = fe.IntegralForm(P, v, r.dV)
    a = fe.IntegralForm(A, v, r.dV, v)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    system = fe.solve.partition(v, A, dof1, dof0)

    du = fe.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fe.solve.solve(*system, ext0)
    assert np.allclose(du, 0)

    system = fe.solve.partition(v, A, dof1, dof0, b)

    du = fe.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fe.solve.solve(*system, ext0)
    assert np.allclose(du, 0)


if __name__ == "__main__":
    test_solve()
