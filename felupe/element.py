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
    def __init__(self, order, ndim):
        self.ndim = ndim
        self.order = order
        self.nnodes = (order + 1) ** ndim
        self.nbasis = self.nnodes

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
        i = np.arange(n)
        return 2 * i / (n - 1) - 1

    def _polynomial(self, r, n):
        "Lagrange-Polynomial of order n evaluated at coordinate r."
        m = np.arange(n)
        return r ** m / factorial(m)


class Line:
    def __init__(self):
        self.ndim = 1


class Quad:
    def __init__(self):
        self.ndim = 2


class Hex:
    def __init__(self):
        self.ndim = 3


class Line1(Line):
    def __init__(self):
        super().__init__()
        self.nnodes = 2
        self.nbasis = 2

    def basis(self, rv):
        "linear line basis functions"
        (r,) = rv
        return np.array([(1 - r), (1 + r)]) * 0.5

    def basisprime(self, rv):
        "linear line derivative of basis functions"
        (r,) = rv
        return np.array([[-1], [1]]) * 0.5


class Quad0(Quad):
    def __init__(self):
        super().__init__()
        self.nnodes = 4
        self.nbasis = 1

    def basis(self, rst):
        "linear quadrilateral basis functions"
        return np.array([1])

    def basisprime(self, rst):
        "linear quadrilateral derivative of basis functions"
        return np.array([[0, 0, 0]])


class Quad1(Quad):
    def __init__(self):
        super().__init__()
        self.nnodes = 4
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


class Hex0(Hex):
    def __init__(self):
        super().__init__()
        self.nnodes = 8
        self.nbasis = 1

    def basis(self, rst):
        "constant hexahedron basis functions"
        return np.array([1])

    def basisprime(self, rst):
        "constant hexahedron derivative of basis functions"
        return np.array([[0, 0, 0]])


class Hex1(Hex):
    def __init__(self):
        super().__init__()
        self.nnodes = 8
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


class Hex2s(Hex):
    def __init__(self):
        super().__init__()
        self.nnodes = 20
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
        return (
            np.array(
                [
                    [
                        (1 - s) * (1 - t) * (2 + r + s + t) * 0.125
                        - (1 - r) * (1 - s) * (1 - t) * (2 + s + t) * 0.125,
                        (1 - r) * (1 - t) * (2 + r + s + t) * 0.125
                        - (1 - r) * (1 - s) * (1 - t) * (2 + r + t) * 0.125,
                        (1 - r) * (1 - s) * (2 + r + s + t) * 0.125
                        - (1 - r) * (1 - s) * (1 - t) * (2 + r + s) * 0.125,
                    ],
                    [
                        -(1 - s) * (1 - t) * (2 - r + s + t) * 0.125
                        + (1 + r) * (1 - s) * (1 - t) * (2 + s + t) * 0.125,
                        (1 + r) * (1 - t) * (2 - r + s + t) * 0.125
                        - (1 + r) * (1 - s) * (1 - t) * (2 - r + t) * 0.125,
                        (1 + r) * (1 - s) * (2 - r + s + t) * 0.125
                        - (1 + r) * (1 - s) * (1 - t) * (2 - r + s) * 0.125,
                    ],
                    [
                        -(1 + s) * (1 - t) * (2 - r - s + t) * 0.125
                        + (1 + r) * (1 + s) * (1 - t) * (2 - s + t) * 0.125,
                        -(1 + r) * (1 - t) * (2 - r - s + t) * 0.125
                        + (1 + r) * (1 + s) * (1 - t) * (2 - r + t) * 0.125,
                        (1 + r) * (1 + s) * (2 - r - s + t) * 0.125
                        - (1 + r) * (1 + s) * (1 - t) * (2 - r - s) * 0.125,
                    ],
                    [
                        (1 + s) * (1 - t) * (2 + r - s + t) * 0.125
                        - (1 - r) * (1 + s) * (1 - t) * (2 - s + t) * 0.125,
                        -(1 - r) * (1 - t) * (2 + r - s + t) * 0.125
                        + (1 - r) * (1 + s) * (1 - t) * (2 + r + t) * 0.125,
                        (1 - r) * (1 + s) * (2 + r - s + t) * 0.125
                        - (1 - r) * (1 + s) * (1 - t) * (2 + r - s) * 0.125,
                    ],
                    #
                    [
                        (1 - s) * (1 + t) * (2 + r + s - t) * 0.125
                        - (1 - r) * (1 - s) * (1 + t) * (2 + s - t) * 0.125,
                        (1 - r) * (1 + t) * (2 + r + s - t) * 0.125
                        - (1 - r) * (1 - s) * (1 + t) * (2 + r - t) * 0.125,
                        -(1 - r) * (1 - s) * (2 + r + s - t) * 0.125
                        + (1 - r) * (1 - s) * (1 + t) * (2 + r + s) * 0.125,
                    ],
                    [
                        -(1 - s) * (1 + t) * (2 - r + s - t) * 0.125
                        + (1 + r) * (1 - s) * (1 + t) * (2 + s - t) * 0.125,
                        (1 + r) * (1 + t) * (2 - r + s - t) * 0.125
                        - (1 + r) * (1 - s) * (1 + t) * (2 - r - t) * 0.125,
                        -(1 + r) * (1 - s) * (2 - r + s - t) * 0.125
                        + (1 + r) * (1 - s) * (1 + t) * (2 - r + s) * 0.125,
                    ],
                    [
                        -(1 + s) * (1 + t) * (2 - r - s - t) * 0.125
                        + (1 + r) * (1 + s) * (1 + t) * (2 - s - t) * 0.125,
                        -(1 + r) * (1 + t) * (2 - r - s - t) * 0.125
                        + (1 + r) * (1 + s) * (1 + t) * (2 - r - t) * 0.125,
                        -(1 + r) * (1 + s) * (2 - r - s - t) * 0.125
                        + (1 + r) * (1 + s) * (1 + t) * (2 - r - s) * 0.125,
                    ],
                    [
                        (1 + s) * (1 + t) * (2 + r - s - t) * 0.125
                        - (1 - r) * (1 + s) * (1 + t) * (2 - s - t) * 0.125,
                        -(1 - r) * (1 + t) * (2 + r - s - t) * 0.125
                        + (1 - r) * (1 + s) * (1 + t) * (2 + r - t) * 0.125,
                        -(1 - r) * (1 + s) * (2 + r - s - t) * 0.125
                        + (1 - r) * (1 + s) * (1 + t) * (2 + r - s) * 0.125,
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
            * 0.125
        )
