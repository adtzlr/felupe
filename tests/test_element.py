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


def test_line2():
    line2 = fe.element.Line()

    r = [-1]

    h = line2.shape.function(r)
    dhdr = line2.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)

    assert line2.shape.n, line2.dim == dhdr.shape


def test_quad0():
    quad0 = fe.element.ConstantQuad()

    r = [-1, -1]

    h = quad0.shape.function(r)
    dhdr = quad0.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == 0)

    assert quad0.shape.n, quad0.dim == dhdr.shape


def test_quad4():
    quad4 = fe.element.Quad()

    r = [-1, -1]

    h = quad4.shape.function(r)
    dhdr = quad4.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)

    assert quad4.shape.n, quad4.dim == dhdr.shape


def test_hex0():
    hex0 = fe.element.ConstantHexahedron()

    r = [-1, -1, -1]

    h = hex0.shape.function(r)
    dhdr = hex0.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == 0)

    assert hex0.shape.n, hex0.dim == dhdr.shape


def test_hex8():
    hex8 = fe.element.Hexahedron()

    r = [-1, -1, -1]

    h = hex8.shape.function(r)
    dhdr = hex8.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)

    assert hex8.shape.n, hex8.dim == dhdr.shape


def test_hex20():
    hex20 = fe.element.QuadraticHexahedron()

    r = [-1, -1, -1]

    h = hex20.shape.function(r)
    dhdr = hex20.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)

    assert hex20.shape.n, hex20.dim == dhdr.shape


def test_hex27():
    hex27 = fe.element.TriQuadraticHexahedron()

    r = [-1, -1, -1]

    h = hex27.shape.function(r)
    dhdr = hex27.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)

    assert hex27.shape.n, hex27.dim == dhdr.shape


def test_tri3():
    tri3 = fe.element.Triangle()

    r = [0, 0]

    h = tri3.shape.function(r)
    dhdr = tri3.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1)

    assert tri3.shape.n, tri3.dim == dhdr.shape


def test_tri6():
    tri6 = fe.element.QuadraticTriangle()

    r = [0, 0]

    h = tri6.shape.function(r)
    dhdr = tri6.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -3)

    assert tri6.shape.n, tri6.dim == dhdr.shape


def test_tri_mini():
    trim = fe.element.TriangleMINI()

    r = [0, 0]

    h = trim.shape.function(r)
    dhdr = trim.shape.gradient(r)

    assert h[0] == 1
    assert h[-1] == 0  # check bubble
    assert np.all(dhdr[0] == -1)

    assert trim.shape.n, trim.dim == dhdr.shape


def test_tet4():
    tet4 = fe.element.Tetra()

    r = [0, 0, 0]

    h = tet4.shape.function(r)
    dhdr = tet4.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1)

    assert tet4.shape.n, tet4.dim == dhdr.shape


def test_tet10():
    tet10 = fe.element.QuadraticTetra()

    r = [0, 0, 0]

    h = tet10.shape.function(r)
    dhdr = tet10.shape.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -3)

    assert tet10.shape.n, tet10.dim == dhdr.shape


def test_tet_mini():
    tetm = fe.element.TetraMINI()

    r = [0, 0, 0]

    h = tetm.shape.function(r)
    dhdr = tetm.shape.gradient(r)

    assert h[0] == 1
    assert h[-1] == 0  # check bubble
    assert np.all(dhdr[0] == -1)

    assert tetm.shape.n, tetm.dim == dhdr.shape


def test_aol():
    aol32 = fe.element.ArbitraryOrderLagrange(order=3, ndim=2)
    aol23 = fe.element.ArbitraryOrderLagrange(order=2, ndim=3)

    r = [-1, -1]

    h = aol32.shape.function(r)
    dhdr = aol32.shape.gradient(r)

    assert h[0] == 1
    assert aol32.shape.n, aol32.dim == dhdr.shape

    r = [-1, -1, -1]

    h = aol23.shape.function(r)
    dhdr = aol23.shape.gradient(r)

    assert h[0] == 1
    assert aol23.shape.n, aol23.dim == dhdr.shape


if __name__ == "__main__":
    test_line2()

    test_quad0()
    test_quad4()

    test_hex0()
    test_hex8()
    test_hex20()
    test_hex27()

    test_tri3()
    test_tri6()

    test_tri_mini()
    test_tet_mini()

    test_tet4()
    test_tet10()

    test_aol()
