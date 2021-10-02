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

from ._base import HexahedronElement
from ._lagrange import ArbitraryOrderLagrange


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