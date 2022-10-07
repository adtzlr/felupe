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


def test_math_field():
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    v = fe.FieldContainer((u, u))

    fe.math.values(v)
    fe.math.defgrad(u)
    fe.math.defgrad(v)

    fe.math.norm([u.values, u.values])
    fe.math.norm(u.values)
    fe.math.interpolate(u)
    fe.math.grad(u)
    fe.math.grad(u, sym=True)
    fe.math.tovoigt(fe.math.strain(u), strain=False)
    fe.math.tovoigt(fe.math.strain(u), strain=True)
    fe.math.strain(u)
    fe.math.extract(u)
    fe.math.extract(u, grad=False)
    fe.math.extract(u, sym=True)
    fe.math.extract(u, grad=True, sym=False, add_identity=False)


def test_math():
    H = (np.random.rand(3, 3, 8, 200) - 0.5) / 10
    F = fe.math.identity(H) + H
    C = fe.math.dot(fe.math.transpose(F), F)
    A = np.random.rand(3, 3, 3, 3, 8, 200)
    B = np.random.rand(3, 3, 3, 8, 200)

    a = np.random.rand(3, 8, 200)

    fe.math.identity(A=None, dim=3, shape=(8, 20))

    fe.math.cross(a, a)
    fe.math.dya(a, a, mode=1)

    with pytest.raises(ValueError):
        fe.math.dya(a, a, mode=3)

    fe.math.sym(H)

    fe.math.dot(C, C)
    fe.math.dot(C, A)
    fe.math.dot(A, C)

    fe.math.transpose(F, mode=1)
    fe.math.transpose(A, mode=2)

    with pytest.raises(ValueError):
        fe.math.transpose(F, mode=3)

    with pytest.raises(TypeError):
        fe.math.dot(C, B)

    with pytest.raises(TypeError):
        fe.math.dot(B, C)

    assert fe.math.dot(C, a).shape == (3, 8, 200)
    assert fe.math.dot(a, C).shape == (3, 8, 200)
    assert fe.math.dot(a, a).shape == (8, 200)

    assert fe.math.dot(a, A).shape == (3, 3, 3, 8, 200)
    assert fe.math.dot(A, a).shape == (3, 3, 3, 8, 200)
    assert fe.math.dot(A, A).shape == (3, 3, 3, 3, 3, 3, 8, 200)

    assert fe.math.ddot(C, C).shape == (8, 200)
    assert fe.math.ddot(C, A).shape == (3, 3, 8, 200)
    assert fe.math.ddot(A, C).shape == (3, 3, 8, 200)

    assert fe.math.ddot(A, A).shape == (3, 3, 3, 3, 8, 200)

    with pytest.raises(TypeError):
        fe.math.ddot(A, B)
        fe.math.ddot(B, B)
        fe.math.ddot(C, B)

    detC = fe.math.det(C)
    fe.math.det(C[:2, :2])
    fe.math.det(C[:1, :1])

    fe.math.inv(C)
    fe.math.inv(C[:2, :2])
    fe.math.inv(C, determinant=detC)
    fe.math.inv(C, full_output=True)
    fe.math.inv(C, sym=True)

    fe.math.dev(C)
    fe.math.cof(C)
    fe.math.dya(C, C)
    fe.math.cdya_ik(F, F)
    fe.math.cdya_il(F, F)
    fe.math.cdya(F, F)

    fe.math.tovoigt(C)
    fe.math.tovoigt(C[:2, :2])
    fe.math.eigvals(C)
    fe.math.eigvals(C[:2, :2])
    fe.math.eigvals(C, shear=True)
    fe.math.eigvals(C[:2, :2], shear=True)
    fe.math.eigvalsh(C)
    fe.math.eigh(C)
    fe.math.eig(C)

    fe.math.majortranspose(A)
    fe.math.trace(C)


if __name__ == "__main__":
    test_math()
    test_math_field()
