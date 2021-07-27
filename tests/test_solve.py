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

    m = fe.mesh.Cube(n=3)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)

    nh = fe.constitution.models.NeoHooke(1, 3)

    F = fe.tools.defgrad(u)
    P = nh.P(F)
    A = nh.A(F)

    b = fe.doftools.symmetry(u)
    dof0, dof1 = fe.doftools.partition(u, b)

    u0ext = fe.doftools.apply(u, b, dof0)

    L = fe.IntegralForm(P, u, r.dV, grad_v=True)
    a = fe.IntegralForm(A, u, r.dV, u, True, True)

    b = L.assemble().toarray()[:, 0]
    A = a.assemble()

    system = fe.solve.partition(u, A, dof1, dof0)

    du = fe.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fe.solve.solve(*system, u0ext)
    assert np.allclose(du, 0)

    system = fe.solve.partition(u, A, dof1, dof0, b)

    du = fe.solve.solve(*system)
    assert np.allclose(du, 0)

    du = fe.solve.solve(*system, u0ext)
    assert np.allclose(du, 0)


if __name__ == "__main__":
    test_solve()
