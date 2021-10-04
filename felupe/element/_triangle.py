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

from ._base import TriangleElement


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
    def __init__(self, bubble_multiplier=1.0):
        super().__init__()
        self.npoints = 4
        self.nbasis = 4
        self.bubble_multiplier = bubble_multiplier

    def basis(self, rs):
        "linear triangle basis functions"
        r, s = rs
        a = self.bubble_multiplier
        return np.array([1 - r - s, r, s, a * r * s * (1 - r - s)])

    def basisprime(self, rs):
        "linear triangle derivative of basis functions"
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
            [[4 * (t1 - t2), -4 * t2], [4 * t3, 4 * t2], [-4 * t3, 4 * (t1 - t2)],]
        )
        dhdr = np.vstack((dhdr_a, dhdr_b))
        dhdr[0] += -dhdr[3] / 2 - dhdr[5] / 2
        dhdr[1] += -dhdr[3] / 2 - dhdr[4] / 2
        dhdr[2] += -dhdr[4] / 2 - dhdr[5] / 2
        return dhdr
