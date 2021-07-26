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


def test_quad0():
    quad0 = fe.element.ConstantQuad()

    r = [-1, -1]

    h = quad0.basis(r)
    dhdr = quad0.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == 0)

    assert quad0.nbasis, quad0.ndim == dhdr.shape


def test_quad4():
    quad4 = fe.element.Quad()

    r = [-1, -1]

    h = quad4.basis(r)
    dhdr = quad4.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)

    assert quad4.nbasis, quad4.ndim == dhdr.shape


# def test_quad8():
#     quad8 = fe.element.QuadraticQuad()

#     r = [-1,-1]

#     h = quad8.basis(r)
#     dhdr = quad8.basisprime(r)

#     assert h[0] == 1
#     assert np.all(dhdr[0] == __VALUE__)

#     assert quad8.nbasis, quad8.ndim == dhdr.shape


# def test_quad9():
#     quad9 = fe.element.BiQuadraticQuad()

#     r = [-1,-1]

#     h = quad9.basis(r)
#     dhdr = quad9.basisprime(r)

#     assert h[0] == 1
#     assert np.all(dhdr[0] == __VALUE__)

#     assert quad9.nbasis, quad9.ndim == dhdr.shape


def test_hex0():
    hex0 = fe.element.ConstantHexahedron()

    r = [-1, -1, -1]

    h = hex0.basis(r)
    dhdr = hex0.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == 0)

    assert hex0.nbasis, hex0.ndim == dhdr.shape


def test_hex8():
    hex8 = fe.element.Hexahedron()

    r = [-1, -1, -1]

    h = hex8.basis(r)
    dhdr = hex8.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)

    assert hex8.nbasis, hex8.ndim == dhdr.shape


def test_hex20():
    hex20 = fe.element.QuadraticHexahedron()

    r = [-1, -1, -1]

    h = hex20.basis(r)
    dhdr = hex20.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)

    assert hex20.nbasis, hex20.ndim == dhdr.shape


def test_hex27():
    hex27 = fe.element.TriQuadraticHexahedron()

    r = [-1, -1, -1]

    h = hex27.basis(r)
    dhdr = hex27.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)

    assert hex27.nbasis, hex27.ndim == dhdr.shape


def test_tri3():
    tri3 = fe.element.Triangle()

    r = [0, 0]

    h = tri3.basis(r)
    dhdr = tri3.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1)

    assert tri3.nbasis, tri3.ndim == dhdr.shape


def test_tri6():
    tri6 = fe.element.QuadraticTriangle()

    r = [0, 0]

    h = tri6.basis(r)
    dhdr = tri6.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -3)

    assert tri6.nbasis, tri6.ndim == dhdr.shape


def test_tri_mini():
    trim = fe.element.TriangleMINI()

    r = [0, 0]

    h = trim.basis(r)
    dhdr = trim.basisprime(r)

    assert h[0] == 1
    assert h[-1] == 0  # check bubble
    assert np.all(dhdr[0] == -1)

    assert trim.nbasis, trim.ndim == dhdr.shape


def test_tet4():
    tet4 = fe.element.Tetra()

    r = [0, 0, 0]

    h = tet4.basis(r)
    dhdr = tet4.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1)

    assert tet4.nbasis, tet4.ndim == dhdr.shape


def test_tet10():
    tet10 = fe.element.QuadraticTetra()

    r = [0, 0, 0]

    h = tet10.basis(r)
    dhdr = tet10.basisprime(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -3)

    assert tet10.nbasis, tet10.ndim == dhdr.shape


def test_aol():
    aol32 = fe.element.ArbitraryOrderLagrange(order=3, ndim=2)
    aol23 = fe.element.ArbitraryOrderLagrange(order=2, ndim=3)

    r = [-1, -1]

    h = aol32.basis(r)
    dhdr = aol32.basisprime(r)

    assert h[0] == 1
    assert aol32.nbasis, aol32.ndim == dhdr.shape

    r = [-1, -1, -1]

    h = aol23.basis(r)
    dhdr = aol23.basisprime(r)

    assert h[0] == 1
    assert aol23.nbasis, aol23.ndim == dhdr.shape


if __name__ == "__main__":
    test_quad0()
    test_quad4()
    # test_quad8()
    # test_quad9()

    test_hex0()
    test_hex8()
    test_hex20()
    test_hex27()

    test_tri3()
    test_tri6()

    test_tri_mini()

    test_tet4()
    test_tet10()

    test_aol()
