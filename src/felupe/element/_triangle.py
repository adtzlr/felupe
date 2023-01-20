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


class Triangle(Element):
    def __init__(self):
        super().__init__(shape=(3, 2))
        self.points = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)

    def function(self, rs):
        "linear triangle shape functions"
        r, s = rs
        return np.array([1 - r - s, r, s])

    def gradient(self, rs):
        "linear triangle gradient of shape functions"
        r, s = rs
        return np.array([[-1, -1], [1, 0], [0, 1]], dtype=float)


class TriangleMINI(Element):
    def __init__(self, bubble_multiplier=1.0):
        super().__init__(shape=(4, 2))
        self.points = np.array([[0, 0], [1, 0], [0, 1], [1 / 3, 1 / 3]], dtype=float)
        self.bubble_multiplier = bubble_multiplier

    def function(self, rs):
        "linear bubble-enriched triangle shape functions"
        r, s = rs
        a = self.bubble_multiplier
        return np.array([1 - r - s, r, s, a * r * s * (1 - r - s)])

    def gradient(self, rs):
        "linear bubble-enriched triangle gradient of shape functions"
        r, s = rs
        a = self.bubble_multiplier
        return np.array(
            [
                [-1, -1],
                [1, 0],
                [0, 1],
                [a * (s * (1 - r - s) - r * s), a * (r * (1 - r - s) - r * s)],
            ],
            dtype=float,
        )


class QuadraticTriangle(Element):
    def __init__(self):
        super().__init__(shape=(6, 2))
        self.points = np.zeros(self.shape)
        self.points[:3] = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        self.points[3] = np.mean(self.points[[0, 1]], axis=0)
        self.points[4] = np.mean(self.points[[1, 2]], axis=0)
        self.points[5] = np.mean(self.points[[2, 0]], axis=0)

    def function(self, rs):
        "quadratic triangle shape functions"
        r, s = rs
        h = np.array(
            [1 - r - s, r, s, 4 * r * (1 - r - s), 4 * r * s, 4 * s * (1 - r - s)]
        )
        h[0] += -h[3] / 2 - h[5] / 2
        h[1] += -h[3] / 2 - h[4] / 2
        h[2] += -h[4] / 2 - h[5] / 2

        return h

    def gradient(self, rs):
        "quadratic triangle gradient of shape functions"
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
