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

from types import SimpleNamespace

import numpy as np

from . import Scheme


class Tetrahedron(Scheme):
    "An integration scheme for Tetrahedrons."

    def __init__(self, order: int):
        "Integration scheme for Tetrahedrons of `order` 1, 2 or 3."

        scheme = SimpleNamespace()
        volume = 1 / 6

        if order == 1:
            scheme.points = np.ones((1, 4)) / 4
            scheme.weights = np.ones(1)

        elif order == 2:
            a = 0.58541020
            b = 0.13819660
            scheme.points = np.array(
                [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]]
            )
            scheme.weights = np.ones(4) / 4

        elif order == 3:
            a = 1 / 6
            b = 1 / 2
            c = 1 / 4
            scheme.points = np.array(
                [[c, c, c, c], [b, a, a, a], [a, b, a, a], [a, a, b, a], [a, a, a, b]]
            )
            scheme.weights = np.array([-4 / 5, 9 / 20, 9 / 20, 9 / 20, 9 / 20])

        else:
            raise NotImplementedError("order must be either 1, 2 or 3.")

        tetra = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        points = np.dot(tetra.T, scheme.points.T).T
        super().__init__(points, scheme.weights * volume)
