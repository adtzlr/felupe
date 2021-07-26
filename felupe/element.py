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
from scipy.special import factorial
from string import ascii_lowercase as alphabet
from copy import deepcopy as copy


class ArbitraryOrderLagrange:
    "Lagrange quad/hexahdron finite element of arbitrary order."

    def __init__(self, order, ndim, interval=(-1, 1)):
        self.ndim = ndim
        self.order = order
        self.npoints = (order + 1) ** ndim
        self.nbasis = self.npoints
        self.interval = interval

        self._nbasis = order + 1

        # init curve-parameter matrix
        n = self._nbasis
        self._AT = np.linalg.inv(
            np.array([self._polynomial(p, n) for p in self._points(n)])
        ).T

        # indices for outer product in einstein notation
        # idx = ["a", "b", "c", ...][:dim]
        # subscripts = "a,b,c -> abc"
        self._idx = [letter for letter in alphabet][: self.ndim]
        self._subscripts = ",".join(self._idx) + "->" + "".join(self._idx)

    def basis(self, r):
        "Basis function vector at coordinate vector r."
        n = self._nbasis

        # 1d - basis function vectors per axis
        h = [self._AT @ self._polynomial(ra, n) for ra in r]

        return np.einsum(self._subscripts, *h).ravel("F")

    def basisprime(self, r):
        "Basis function derivative vector at coordinate vector r."
        n = self._nbasis

        # 1d - basis function vectors per axis
        h = [self._AT @ self._polynomial(ra, n) for ra in r]

        # shifted 1d - basis function vectors per axis
        k = [self._AT @ np.append(0, self._polynomial(ra, n)[:-1]) for ra in r]

        # init output
        dhdr = np.zeros((n ** self.ndim, self.ndim))

        # loop over columns
        for i in range(self.ndim):
            g = copy(h)
            g[i] = k[i]
            dhdr[:, i] = np.einsum(self._subscripts, *g).ravel("F")

        return dhdr

    def _points(self, n):
        "Equidistant n points in interval [-1, 1]."
        return np.linspace(*self.interval, n)

    def _polynomial(self, r, n):
        "Lagrange-Polynomial of order n evaluated at coordinate r."
        m = np.arange(n)
        return r ** m / factorial(m)


class LineElement:
    def __init__(self):
        self.ndim = 1


class QuadElement:
    def __init__(self):
        self.ndim = 2


class HexahedronElement:
    def __init__(self):
        self.ndim = 3


class TriangleElement:
    def __init__(self):
        self.ndim = 2


class TetraElement:
    def __init__(self):
        self.ndim = 3


class Line(LineElement):
    def __init__(self):
        super().__init__()
        self.npoints = 2
        self.nbasis = 2

    def basis(self, rv):
        "linear line basis functions"
        (r,) = rv
        return np.array([(1 - r), (1 + r)]) * 0.5

    def basisprime(self, rv):
        "linear line derivative of basis functions"
        (r,) = rv
        return np.array([[-1], [1]]) * 0.5


class ConstantQuad(QuadElement):
    def __init__(self):
        super().__init__()
        self.npoints = 4
        self.nbasis = 1

    def basis(self, rst):
        "linear quadrilateral basis functions"
        return np.array([1])

    def basisprime(self, rst):
        "linear quadrilateral derivative of basis functions"
        return np.array([[0, 0, 0]])


class Quad(QuadElement):
    def __init__(self):
        super().__init__()
        self.npoints = 4
        self.nbasis = 4

    def basis(self, rs):
        "linear quadrilateral basis functions"
        r, s = rs
        return (
            np.array(
                [
                    (1 - r) * (1 - s),
                    (1 + r) * (1 - s),
                    (1 + r) * (1 + s),
                    (1 - r) * (1 + s),
                ]
            )
            * 0.25
        )

    def basisprime(self, rs):
        "linear quadrilateral derivative of basis functions"
        r, s = rs
        return (
            np.array(
                [
                    [-(1 - s), -(1 - r)],
                    [(1 - s), -(1 + r)],
                    [(1 + s), (1 + r)],
                    [-(1 + s), (1 - r)],
                ]
            )
            * 0.25
        )


class ConstantHexahedron(HexahedronElement):
    def __init__(self):
        super().__init__()
        self.npoints = 8
        self.nbasis = 1

    def basis(self, rst):
        "constant hexahedron basis functions"
        return np.array([1])

    def basisprime(self, rst):
        "constant hexahedron derivative of basis functions"
        return np.array([[0, 0, 0]])


class Hexahedron(HexahedronElement):
    def __init__(self):
        super().__init__()
        self.npoints = 8
        self.nbasis = 8

    def basis(self, rst):
        "linear hexahedron basis functions"
        r, s, t = rst
        return (
            np.array(
                [
                    (1 - r) * (1 - s) * (1 - t),
                    (1 + r) * (1 - s) * (1 - t),
                    (1 + r) * (1 + s) * (1 - t),
                    (1 - r) * (1 + s) * (1 - t),
                    (1 - r) * (1 - s) * (1 + t),
                    (1 + r) * (1 - s) * (1 + t),
                    (1 + r) * (1 + s) * (1 + t),
                    (1 - r) * (1 + s) * (1 + t),
                ]
            )
            * 0.125
        )

    def basisprime(self, rst):
        "linear hexahedron derivative of basis functions"
        r, s, t = rst
        return (
            np.array(
                [
                    [-(1 - s) * (1 - t), -(1 - r) * (1 - t), -(1 - r) * (1 - s)],
                    [(1 - s) * (1 - t), -(1 + r) * (1 - t), -(1 + r) * (1 - s)],
                    [(1 + s) * (1 - t), (1 + r) * (1 - t), -(1 + r) * (1 + s)],
                    [-(1 + s) * (1 - t), (1 - r) * (1 - t), -(1 - r) * (1 + s)],
                    [-(1 - s) * (1 + t), -(1 - r) * (1 + t), (1 - r) * (1 - s)],
                    [(1 - s) * (1 + t), -(1 + r) * (1 + t), (1 + r) * (1 - s)],
                    [(1 + s) * (1 + t), (1 + r) * (1 + t), (1 + r) * (1 + s)],
                    [-(1 + s) * (1 + t), (1 - r) * (1 + t), (1 - r) * (1 + s)],
                ]
            )
            * 0.125
        )


class QuadraticHexahedron(HexahedronElement):
    def __init__(self):
        super().__init__()
        self.npoints = 20
        self.nbasis = 20

    def basis(self, rst):
        "quadratic serendipity hexahedron basis functions"
        r, s, t = rst
        return np.array(
            [
                -(1 - r) * (1 - s) * (1 - t) * (2 + r + s + t) * 0.125,
                -(1 + r) * (1 - s) * (1 - t) * (2 - r + s + t) * 0.125,
                -(1 + r) * (1 + s) * (1 - t) * (2 - r - s + t) * 0.125,
                -(1 - r) * (1 + s) * (1 - t) * (2 + r - s + t) * 0.125,
                #
                -(1 - r) * (1 - s) * (1 + t) * (2 + r + s - t) * 0.125,
                -(1 + r) * (1 - s) * (1 + t) * (2 - r + s - t) * 0.125,
                -(1 + r) * (1 + s) * (1 + t) * (2 - r - s - t) * 0.125,
                -(1 - r) * (1 + s) * (1 + t) * (2 + r - s - t) * 0.125,
                #
                (1 - r) * (1 + r) * (1 - s) * (1 - t) * 0.25,
                (1 - s) * (1 + s) * (1 + r) * (1 - t) * 0.25,
                (1 - r) * (1 + r) * (1 + s) * (1 - t) * 0.25,
                (1 - s) * (1 + s) * (1 - r) * (1 - t) * 0.25,
                #
                (1 - r) * (1 + r) * (1 - s) * (1 + t) * 0.25,
                (1 - s) * (1 + s) * (1 + r) * (1 + t) * 0.25,
                (1 - r) * (1 + r) * (1 + s) * (1 + t) * 0.25,
                (1 - s) * (1 + s) * (1 - r) * (1 + t) * 0.25,
                #
                (1 - t) * (1 + t) * (1 - r) * (1 - s) * 0.25,
                (1 - t) * (1 + t) * (1 + r) * (1 - s) * 0.25,
                (1 - t) * (1 + t) * (1 + r) * (1 + s) * 0.25,
                (1 - t) * (1 + t) * (1 - r) * (1 + s) * 0.25,
            ]
        )

    def basisprime(self, rst):
        "quadratic serendipity hexahedron derivative of basis functions"
        r, s, t = rst
        return np.array(
            [
                [
                    (1 - s) * (1 - t) * (2 + r + s + t) * 0.125
                    - (1 - r) * (1 - s) * (1 - t) * 0.125,
                    (1 - r) * (1 - t) * (2 + r + s + t) * 0.125
                    - (1 - r) * (1 - s) * (1 - t) * 0.125,
                    (1 - r) * (1 - s) * (2 + r + s + t) * 0.125
                    - (1 - r) * (1 - s) * (1 - t) * 0.125,
                ],
                [
                    -(1 - s) * (1 - t) * (2 - r + s + t) * 0.125
                    + (1 + r) * (1 - s) * (1 - t) * 0.125,
                    (1 + r) * (1 - t) * (2 - r + s + t) * 0.125
                    - (1 + r) * (1 - s) * (1 - t) * 0.125,
                    (1 + r) * (1 - s) * (2 - r + s + t) * 0.125
                    - (1 + r) * (1 - s) * (1 - t) * 0.125,
                ],
                [
                    -(1 + s) * (1 - t) * (2 - r - s + t) * 0.125
                    + (1 + r) * (1 + s) * (1 - t) * 0.125,
                    -(1 + r) * (1 - t) * (2 - r - s + t) * 0.125
                    + (1 + r) * (1 + s) * (1 - t) * 0.125,
                    (1 + r) * (1 + s) * (2 - r - s + t) * 0.125
                    - (1 + r) * (1 + s) * (1 - t) * 0.125,
                ],
                [
                    (1 + s) * (1 - t) * (2 + r - s + t) * 0.125
                    - (1 - r) * (1 + s) * (1 - t) * 0.125,
                    -(1 - r) * (1 - t) * (2 + r - s + t) * 0.125
                    + (1 - r) * (1 + s) * (1 - t) * 0.125,
                    (1 - r) * (1 + s) * (2 + r - s + t) * 0.125
                    - (1 - r) * (1 + s) * (1 - t) * 0.125,
                ],
                #
                [
                    (1 - s) * (1 + t) * (2 + r + s - t) * 0.125
                    - (1 - r) * (1 - s) * (1 + t) * 0.125,
                    (1 - r) * (1 + t) * (2 + r + s - t) * 0.125
                    - (1 - r) * (1 - s) * (1 + t) * 0.125,
                    -(1 - r) * (1 - s) * (2 + r + s - t) * 0.125
                    + (1 - r) * (1 - s) * (1 + t) * 0.125,
                ],
                [
                    -(1 - s) * (1 + t) * (2 - r + s - t) * 0.125
                    + (1 + r) * (1 - s) * (1 + t) * 0.125,
                    (1 + r) * (1 + t) * (2 - r + s - t) * 0.125
                    - (1 + r) * (1 - s) * (1 + t) * 0.125,
                    -(1 + r) * (1 - s) * (2 - r + s - t) * 0.125
                    + (1 + r) * (1 - s) * (1 + t) * 0.125,
                ],
                [
                    -(1 + s) * (1 + t) * (2 - r - s - t) * 0.125
                    + (1 + r) * (1 + s) * (1 + t) * 0.125,
                    -(1 + r) * (1 + t) * (2 - r - s - t) * 0.125
                    + (1 + r) * (1 + s) * (1 + t) * 0.125,
                    -(1 + r) * (1 + s) * (2 - r - s - t) * 0.125
                    + (1 + r) * (1 + s) * (1 + t) * 0.125,
                ],
                [
                    (1 + s) * (1 + t) * (2 + r - s - t) * 0.125
                    - (1 - r) * (1 + s) * (1 + t) * 0.125,
                    -(1 - r) * (1 + t) * (2 + r - s - t) * 0.125
                    + (1 - r) * (1 + s) * (1 + t) * 0.125,
                    -(1 - r) * (1 + s) * (2 + r - s - t) * 0.125
                    + (1 - r) * (1 + s) * (1 + t) * 0.125,
                ],
                #
                [
                    -2 * r * (1 - s) * (1 - t) * 0.25,
                    -(1 - r) * (1 + r) * (1 - t) * 0.25,
                    -(1 - r) * (1 + r) * (1 - s) * 0.25,
                ],
                [
                    (1 - s) * (1 + s) * (1 - t) * 0.25,
                    -2 * s * (1 + r) * (1 - t) * 0.25,
                    -(1 - s) * (1 + s) * (1 + r) * 0.25,
                ],
                [
                    -2 * r * (1 + s) * (1 - t) * 0.25,
                    (1 - r) * (1 + r) * (1 - t) * 0.25,
                    -(1 - r) * (1 + r) * (1 + s) * 0.25,
                ],
                [
                    -(1 - s) * (1 + s) * (1 - t) * 0.25,
                    -2 * s * (1 - r) * (1 - t) * 0.25,
                    -(1 - s) * (1 + s) * (1 - r) * 0.25,
                ],
                #
                [
                    -2 * r * (1 - s) * (1 + t) * 0.25,
                    -(1 - r) * (1 + r) * (1 + t) * 0.25,
                    (1 - r) * (1 + r) * (1 - s) * 0.25,
                ],
                [
                    (1 - s) * (1 + s) * (1 + t) * 0.25,
                    -2 * s * (1 + r) * (1 + t) * 0.25,
                    (1 - s) * (1 + s) * (1 + r) * 0.25,
                ],
                [
                    -2 * r * (1 + s) * (1 + t) * 0.25,
                    (1 - r) * (1 + r) * (1 + t) * 0.25,
                    (1 - r) * (1 + r) * (1 + s) * 0.25,
                ],
                [
                    -(1 - s) * (1 + s) * (1 + t) * 0.25,
                    -2 * s * (1 - r) * (1 + t) * 0.25,
                    (1 - s) * (1 + s) * (1 - r) * 0.25,
                ],
                #
                [
                    -(1 - t) * (1 + t) * (1 - s) * 0.25,
                    -(1 - t) * (1 + t) * (1 - r) * 0.25,
                    -2 * t * (1 - r) * (1 - s) * 0.25,
                ],
                [
                    (1 - t) * (1 + t) * (1 - s) * 0.25,
                    -(1 - t) * (1 + t) * (1 + r) * 0.25,
                    -2 * t * (1 + r) * (1 - s) * 0.25,
                ],
                [
                    (1 - t) * (1 + t) * (1 + s) * 0.25,
                    (1 - t) * (1 + t) * (1 + r) * 0.25,
                    -2 * t * (1 + r) * (1 + s) * 0.25,
                ],
                [
                    -(1 - t) * (1 + t) * (1 + s) * 0.25,
                    (1 - t) * (1 + t) * (1 - r) * 0.25,
                    -2 * t * (1 - r) * (1 + s) * 0.25,
                ],
            ]
        )


class TriQuadraticHexahedron(Hexahedron):
    def __init__(self):
        super().__init__()
        self.npoints = 27
        self.nbasis = 27
        self.lagrange = ArbitraryOrderLagrange(order=2, ndim=3)
        self.vertices = np.array([0, 2, 8, 6, 18, 20, 26, 24])
        self.edges = np.array([1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15])
        self.faces = np.array([12, 14, 10, 16, 4, 22])
        self.volume = np.array([13])
        self.permute = np.concatenate(
            (self.vertices, self.edges, self.faces, self.volume)
        )

    def basis(self, rst):
        return self.lagrange.basis(rst)[self.permute]

    def basisprime(self, rst):
        return self.lagrange.basisprime(rst)[self.permute, :]


class Triangle(TriangleElement):
    def __init__(self):
        super().__init__()
        self.npoints = 3
        self.nbasis = 3

    def basis(self, rs):
        "linear triangle basis functions"
        r, s = rs
        return np.array([1 - r - s, r, s])

    def basisprime(self, rs):
        "linear triangle derivative of basis functions"
        r, s = rs
        return np.array([[-1, -1], [1, 0], [0, 1]], dtype=float)


class TriangleMINI(TriangleElement):
    def __init__(self):
        super().__init__()
        self.npoints = 4
        self.nbasis = 3

    def basis(self, rs):
        "linear triangle basis functions"
        r, s = rs
        return np.array([1 - r - s, r, s, 27 * r * s * (1 - r - s)])

    def basisprime(self, rs):
        "linear triangle derivative of basis functions"
        r, s = rs
        return np.array(
            [
                [-1, -1],
                [1, 0],
                [0, 1],
                [27 * (s * (1 - r - s) - r * s), 27 * (r * (1 - r - s) - r * s)],
            ],
            dtype=float,
        )


class Tetra(TetraElement):
    def __init__(self):
        super().__init__()
        self.npoints = 4
        self.nbasis = 4

    def basis(self, rst):
        "linear tetrahedral basis functions"
        r, s, t = rst
        return np.array([1 - r - s - t, r, s, t])

    def basisprime(self, rst):
        "linear tetrahedral derivative of basis functions"
        r, s, t = rst
        return np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


class TetraMINI(TetraElement):
    def __init__(self):
        super().__init__()
        self.npoints = 5
        self.nbasis = 4

    def basis(self, rst):
        "linear tetrahedral basis functions"
        r, s, t = rst
        return np.array([1 - r - s - t, r, s, t, 256 * r * s * t * (1 - r - s - t)])

    def basisprime(self, rst):
        "linear tetrahedral derivative of basis functions"
        r, s, t = rst
        return np.array(
            [
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [
                    256 * (s * t * (1 - r - s - t) - r * s * t),
                    256 * (r * t * (1 - r - s - t) - r * s * t),
                    256 * (r * s * (1 - r - s - t) - r * s * t),
                ],
            ],
            dtype=float,
        )


class QuadraticTriangle(TriangleElement):
    def __init__(self):
        super().__init__()
        self.npoints = 6
        self.nbasis = 6

    def basis(self, rs):
        "linear triangle basis functions"
        r, s = rs
        h = np.array(
            [1 - r - s, r, s, 4 * r * (1 - r - s), 4 * r * s, 4 * s * (1 - r - s)]
        )
        h[0] += -h[3] / 2 - h[5] / 2
        h[1] += -h[3] / 2 - h[4] / 2
        h[2] += -h[4] / 2 - h[5] / 2
        return h

    def basisprime(self, rs):
        "linear triangle derivative of basis functions"
        r, s = rs

        t1 = 1 - r - s
        t2 = r
        t3 = s

        dhdr_a = np.array([[-1, -1], [1, 0], [0, 1]], dtype=float)
        dhdr_b = np.array(
            [
                [4 * (t1 - t2), -4 * t2],
                [4 * t3, 4 * t2],
                [-4 * t3, 4 * (t1 - t2)],
            ]
        )
        dhdr = np.vstack((dhdr_a, dhdr_b))
        dhdr[0] += -dhdr[3] / 2 - dhdr[5] / 2
        dhdr[1] += -dhdr[3] / 2 - dhdr[4] / 2
        dhdr[2] += -dhdr[4] / 2 - dhdr[5] / 2
        return dhdr


class QuadraticTetra(TetraElement):
    def __init__(self):
        super().__init__()
        self.npoints = 10
        self.nbasis = 10

    def basis(self, rst):
        "linear tetrahedral basis functions"
        r, s, t = rst

        t1 = 1 - r - s - t
        t2 = r
        t3 = s
        t4 = t

        h = np.array(
            [
                t1 * (2 * t1 - 1),
                t2 * (2 * t2 - 1),
                t3 * (2 * t3 - 1),
                t4 * (2 * t4 - 1),
                4 * t1 * t2,
                4 * t2 * t3,
                4 * t3 * t1,
                4 * t1 * t4,
                4 * t2 * t4,
                4 * t3 * t4,
            ]
        )

        return h

    def basisprime(self, rst):
        "linear tetrahedral derivative of basis functions"
        r, s, t = rst

        t1 = 1 - r - s - t
        t2 = r
        t3 = s
        t4 = t

        dhdt = np.array(
            [
                [4 * t1 - 1, 0, 0, 0],
                [0, 4 * t2 - 1, 0, 0],
                [0, 0, 4 * t3 - 1, 0],
                [0, 0, 0, 4 * t4 - 1],
                [4 * t2, 4 * t1, 0, 0],
                [0, 4 * t3, 4 * t2, 0],
                [4 * t3, 0, 4 * t1, 0],
                [4 * t4, 0, 0, 4 * t1],
                [0, 4 * t4, 0, 4 * t2],
                [0, 0, 4 * t4, 4 * t3],
            ]
        )

        dtdr = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        return np.dot(dhdt, dtdr)
