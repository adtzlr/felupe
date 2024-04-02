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


def test_math_field():
    m = fem.mesh.Cube()
    e = fem.element.Hexahedron()
    q = fem.quadrature.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3)
    v = fem.FieldContainer((u, u))

    fem.math.values(v)
    fem.math.deformation_gradient(v)

    fem.math.norm([u.values, u.values])
    fem.math.norm(u.values)
    fem.math.interpolate(u)
    fem.math.grad(u)
    fem.math.grad(u, sym=True)
    fem.math.tovoigt(fem.math.strain(v)[:1, :1])
    fem.math.tovoigt(fem.math.strain(v), strain=False)
    fem.math.tovoigt(fem.math.strain(v), strain=True)
    fem.math.strain(v)
    fem.math.extract(u)
    fem.math.extract(u, grad=False)
    fem.math.extract(u, sym=True)
    fem.math.extract(u, grad=True, sym=False, add_identity=False)


def test_math():
    H = (np.random.rand(3, 3, 5, 7) - 0.5) / 10
    F = fem.math.identity(H) + H
    C = fem.math.dot(fem.math.transpose(F), F)
    A = np.random.rand(3, 3, 3, 3, 5, 7)
    B = np.random.rand(3, 3, 3, 5, 7)
    D = np.random.rand(4, 4, 5, 7)

    a = np.random.rand(3, 5, 7)

    fem.math.identity(A=None, dim=3, shape=(8, 20))

    fem.math.cross(a, a)
    fem.math.dya(a, a, mode=1)

    with pytest.raises(ValueError):
        fem.math.dya(a, a, mode=3)

    fem.math.sym(H)

    fem.math.dot(C, C)
    fem.math.dot(C, A, mode=(2, 4))
    fem.math.dot(A, C, mode=(4, 2))

    fem.math.transpose(F, mode=1)
    fem.math.transpose(A, mode=2)

    with pytest.raises(ValueError):
        fem.math.transpose(F, mode=3)

    with pytest.raises(TypeError):
        fem.math.dot(A, B, mode=(4, 3))

    with pytest.raises(TypeError):
        fem.math.dot(B, B, mode=(3, 3))

    fem.math.dot(C, B, mode=(2, 3))
    fem.math.dot(B, C, mode=(3, 2))

    assert fem.math.dot(C, a, mode=(2, 1)).shape == (3, 5, 7)
    assert fem.math.dot(a, C, mode=(1, 2)).shape == (3, 5, 7)
    assert fem.math.dot(a, a, mode=(1, 1)).shape == (5, 7)

    assert fem.math.dot(a, A, mode=(1, 4)).shape == (3, 3, 3, 5, 7)
    assert fem.math.dot(A, a, mode=(4, 1)).shape == (3, 3, 3, 5, 7)
    assert fem.math.dot(A, A, mode=(4, 4)).shape == (3, 3, 3, 3, 3, 3, 5, 7)

    assert fem.math.ddot(C, C, mode=(2, 2)).shape == (5, 7)
    assert fem.math.ddot(C, A, mode=(2, 4)).shape == (3, 3, 5, 7)
    assert fem.math.ddot(A, C, mode=(4, 2)).shape == (3, 3, 5, 7)

    assert fem.math.ddot(A, A, mode=(4, 4)).shape == (3, 3, 3, 3, 5, 7)
    assert fem.math.equivalent_von_mises(C).shape == (5, 7)
    assert fem.math.equivalent_von_mises(C[:2, :2, ...]).shape == (5, 7)

    with pytest.raises(TypeError):
        fem.math.ddot(A, B, mode=(4, 3))

    with pytest.raises(TypeError):
        fem.math.ddot(B, B, mode=(3, 3))

    fem.math.ddot(C, B, mode=(2, 3))
    fem.math.ddot(B, C, mode=(3, 2))

    detC = fem.math.det(C)
    fem.math.det(C[:2, :2])
    fem.math.det(C[:1, :1])

    fem.math.inv(C)
    fem.math.inv(C[:2, :2])
    fem.math.inv(C, determinant=detC)
    fem.math.inv(C, full_output=True)
    fem.math.inv(C, sym=True)
    assert np.allclose(fem.math.inv(C[:1, :1]), 1 / C[:1, :1])

    with pytest.raises(ValueError):
        fem.math.inv(D)
        fem.math.inv(D, determinant=1)

    fem.math.dev(C)
    fem.math.cof(C)
    fem.math.dya(C, C)
    fem.math.cdya_ik(F, F)
    fem.math.cdya_il(F, F)
    fem.math.cdya(F, F)

    fem.math.tovoigt(C)
    fem.math.tovoigt(C[:2, :2])
    with pytest.raises(TypeError):
        fem.math.tovoigt(C[:, :2])

    fem.math.eigvals(C)
    fem.math.eigvals(C[:2, :2])
    fem.math.eigvals(C, shear=True)
    fem.math.eigvals(C[:2, :2], shear=True)
    fem.math.eigvalsh(C)
    fem.math.eigh(C)
    fem.math.eig(C)

    fem.math.majortranspose(A)
    fem.math.trace(C)


def test_math_linsteps():
    steps = fem.math.linsteps([0, 1], num=10)
    assert len(steps) == 11

    steps = fem.math.linsteps([0, 1, 0], num=(10, 100))
    assert len(steps) == 111

    steps = fem.math.linsteps([1], num=0)
    assert len(steps) == 1
    assert steps[-1] == 1

    steps = fem.math.linsteps([1], num=(0, 1))
    assert len(steps) == 1
    assert steps[-1] == 1

    steps = fem.math.linsteps([0, 1, 5], num=(10, 100), axis=1, axes=None)
    assert len(steps) == 111
    assert np.allclose(steps[-1], (0, 5))


if __name__ == "__main__":
    test_math()
    test_math_field()
    test_math_linsteps()
