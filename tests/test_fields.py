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

    return r, u


def pre_axi():

    m = fe.mesh.Rectangle(n=3)
    e = fe.element.Quad()
    q = fe.quadrature.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)

    u = fe.FieldAxisymmetric(r)

    return r, u


def pre_mixed():

    m = fe.mesh.Cube(n=3)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)

    u = fe.Field(r, dim=3)
    p = fe.Field(r)
    J = fe.Field(r, values=1)

    return r, u, p, J


def test_axi():

    r, u = pre_axi()
    u += u.values


def test_3d():

    r, u = pre()
    u += u.values


def test_3d_mixed():

    r, u, p, J = pre_mixed()

    fe.field.extract((u, p, J), fieldgrad=(True, False, False), add_identity=True)

    J.fill(1.0)
    J.full(1.0)

    u + u.values
    u - u.values
    u * u.values
    J / J.values

    u + u
    u - u
    u * u
    J / J

    J /= J.values
    J /= J

    J *= J.values
    J += J.values
    J -= J.values

    J *= J
    J += J
    J -= J

    dof = [0, 1]
    u[dof]


if __name__ == "__main__":
    test_axi()
    test_3d()
    test_3d_mixed()
