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
