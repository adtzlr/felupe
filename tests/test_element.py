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


def test_vertex1():
    vertex1 = fem.element.Vertex()

    r = [0]

    h = vertex1.function(r)
    dhdr = vertex1.gradient(r)
    d2hdrdr = vertex1.hessian(r)

    assert h[0] == 1.0
    assert np.all(dhdr[0] == 0.0)
    assert np.all(d2hdrdr == 0.0)


def test_line2():
    line2 = fem.element.Line()

    r = [-1]

    h = line2.function(r)
    dhdr = line2.gradient(r)
    d2hdrdr = line2.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)
    assert np.all(d2hdrdr == 0.0)

    line2.plot(off_screen=True)


def test_line_lagrange():
    line6 = fem.element.ArbitraryOrderLagrange(order=5, dim=1)

    r = [-1]

    h = line6.function(r)
    dhdr = line6.gradient(r)

    assert np.isclose(h[0], 1)
    assert np.isclose(dhdr[0, 0], -5.70833333)

    line6.plot(off_screen=True)


def test_quad0():
    quad0 = fem.element.ConstantQuad()

    r = [-1, -1]

    h = quad0.function(r)
    dhdr = quad0.gradient(r)
    d2hdrdr = quad0.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == 0)
    assert np.all(d2hdrdr == 0.0)


def test_quad4():
    quad4 = fem.element.Quad()

    r = [-1, -1]

    h = quad4.function(r)
    dhdr = quad4.gradient(r)
    d2hdrdr = quad4.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)
    assert d2hdrdr.shape == (4, 2, 2)

    quad4.plot(off_screen=True)


def test_quad8():
    quad8 = fem.element.QuadraticQuad()

    r = [-1, -1]

    h = quad8.function(r)
    dhdr = quad8.gradient(r)
    d2hdrdr = quad8.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)
    assert d2hdrdr.shape == (8, 2, 2)


def test_quad9():
    quad9 = fem.element.BiQuadraticQuad()

    r = [-1, -1]

    h = quad9.function(r)
    dhdr = quad9.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)


def test_hex0():
    hex0 = fem.element.ConstantHexahedron()

    r = [-1, -1, -1]

    h = hex0.function(r)
    dhdr = hex0.gradient(r)
    d2hdrdr = hex0.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == 0)
    assert np.all(d2hdrdr == 0)


def test_hex8():
    hex8 = fem.element.Hexahedron()

    r = [-1, -1, -1]

    h = hex8.function(r)
    dhdr = hex8.gradient(r)
    d2hdrdr = hex8.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -0.5)
    assert d2hdrdr.shape == (8, 3, 3)


def test_hex20():
    hex20 = fem.element.QuadraticHexahedron()

    r = [-1, -1, -1]

    h = hex20.function(r)
    dhdr = hex20.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)

    hex20.plot(off_screen=True)


def test_hex27():
    hex27 = fem.element.TriQuadraticHexahedron()

    r = [-1, -1, -1]

    h = hex27.function(r)
    dhdr = hex27.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1.5)

    hex27.plot(off_screen=True)


def test_tri3():
    tri3 = fem.element.Triangle()

    r = [0, 0]

    h = tri3.function(r)
    dhdr = tri3.gradient(r)
    d2hdrdr = tri3.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1)
    assert np.all(d2hdrdr == 0)


def test_tri6():
    tri6 = fem.element.QuadraticTriangle()

    r = [0, 0]

    h = tri6.function(r)
    dhdr = tri6.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -3)


def test_tri_mini():
    trim = fem.element.TriangleMINI()

    r = [0, 0]

    h = trim.function(r)
    dhdr = trim.gradient(r)
    d2hdrdr = trim.hessian(r)

    assert h[0] == 1
    assert h[-1] == 0  # check bubble
    assert np.all(dhdr[0] == -1)
    assert d2hdrdr.shape == (4, 2, 2)


def test_tet4():
    tet4 = fem.element.Tetra()

    r = [0, 0, 0]

    h = tet4.function(r)
    dhdr = tet4.gradient(r)
    d2hdrdr = tet4.hessian(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -1)
    assert np.all(d2hdrdr == 0)


def test_tet10():
    tet10 = fem.element.QuadraticTetra()

    r = [0, 0, 0]

    h = tet10.function(r)
    dhdr = tet10.gradient(r)

    assert h[0] == 1
    assert np.all(dhdr[0] == -3)


def test_tet_mini():
    tetm = fem.element.TetraMINI()

    r = [0, 0, 0]

    h = tetm.function(r)
    dhdr = tetm.gradient(r)
    d2hdrdr = tetm.hessian(r)

    assert h[0] == 1
    assert h[-1] == 0  # check bubble
    assert np.all(dhdr[0] == -1)
    assert d2hdrdr.shape == (5, 3, 3)


def test_aol():
    aol32 = fem.element.ArbitraryOrderLagrange(order=3, dim=2)
    aol23 = fem.element.ArbitraryOrderLagrange(order=2, dim=3)

    r = [-1, -1]

    h = aol32.function(r)
    dhdr = aol32.gradient(r)

    assert h[0] == 1

    r = [-1, -1, -1]

    h = aol23.function(r)
    dhdr = aol23.gradient(r)

    assert h[0] == 1


if __name__ == "__main__":
    test_vertex1()

    test_line2()
    test_line_lagrange()

    test_quad0()
    test_quad4()
    test_quad8()
    test_quad9()

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
