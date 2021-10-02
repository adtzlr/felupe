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

from ._base import TetraElement


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
    def __init__(self, bubble_multiplier=1.0):
        super().__init__()
        self.npoints = 5
        self.nbasis = 5
        self.bubble_multiplier = bubble_multiplier

    def basis(self, rst):
        "linear tetrahedral basis functions"
        r, s, t = rst
        a = self.bubble_multiplier
        return np.array([1 - r - s - t, r, s, t, a * r * s * t * (1 - r - s - t)])

    def basisprime(self, rst):
        "linear tetrahedral derivative of basis functions"
        r, s, t = rst
        a = self.bubble_multiplier
        return np.array(
            [
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [
                    a * (s * t * (1 - r - s - t) - r * s * t),
                    a * (r * t * (1 - r - s - t) - r * s * t),
                    a * (r * s * (1 - r - s - t) - r * s * t),
                ],
            ],
            dtype=float,
        )


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
