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

from ._base import Element


class Tetra(Element):
    def __init__(self):
        super().__init__(shape=(4, 3))
        self.points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )

    def function(self, rst):
        "linear tetrahedral shape functions"
        r, s, t = rst
        return np.array([1 - r - s - t, r, s, t])

    def gradient(self, rst):
        "linear tetrahedral gradient of shape functions"
        r, s, t = rst
        return np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


class TetraMINI(Element):
    def __init__(self, bubble_multiplier=1.0):
        super().__init__(shape=(5, 3))
        self.points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]],
            dtype=float,
        )
        self.bubble_multiplier = bubble_multiplier

    def function(self, rst):
        "linear bubble-enriched tetrahedral basis functions"
        r, s, t = rst
        a = self.bubble_multiplier
        return np.array([1 - r - s - t, r, s, t, a * r * s * t * (1 - r - s - t)])

    def gradient(self, rst):
        "linear bubble-enriched tetrahedral derivative of basis functions"
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


class QuadraticTetra(Element):
    def __init__(self):
        super().__init__(shape=(10, 3))
        self.points = np.zeros(self.shape)
        self.points[:4] = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        self.points[4] = np.mean(self.points[[0, 1]], axis=0)
        self.points[5] = np.mean(self.points[[1, 2]], axis=0)
        self.points[6] = np.mean(self.points[[2, 0]], axis=0)
        self.points[7] = np.mean(self.points[[0, 3]], axis=0)
        self.points[8] = np.mean(self.points[[1, 3]], axis=0)
        self.points[9] = np.mean(self.points[[2, 3]], axis=0)

    def function(self, rst):
        "quadratic tetrahedral shape functions"
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

    def gradient(self, rst):
        "quadratic tetrahedral gradient of shape functions"
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
