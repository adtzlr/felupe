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
import quadpy

from types import SimpleNamespace


class Scheme:
    def __init__(self, points, weights):
        self.points = points
        self.weights = weights
        self.npoints, self.dim = self.points.shape


class GaussLegendre(Scheme):
    def __init__(self, order: int, dim: int):
        # integration point weights and coordinates

        if dim == 3:
            scheme = quadpy.c3.product(quadpy.c1.gauss_legendre(order + 1))
        elif dim == 2:
            scheme = quadpy.c2.product(quadpy.c1.gauss_legendre(order + 1))
        elif dim == 1:
            scheme = quadpy.c1.gauss_legendre(order + 1)
            scheme.points = scheme.points.reshape(1, -1)
        else:
            raise ValueError("Wrong dimension.")

        if dim > 1:
            weights = scheme.weights * 2 ** dim
        else:
            weights = scheme.weights

        super().__init__(scheme.points.T, weights)


class Triangle(Scheme):
    def __init__(self, order: int):
        scheme = SimpleNamespace()
        area = 1 / 2

        if order == 1:
            scheme.points = np.ones((1, 3)) / 3
            scheme.weights = np.ones(1)

        elif order == 2:
            a = 2 / 3
            b = 1 / 6
            scheme.points = np.array([[a, b, b], [b, a, b], [b, b, a]])
            scheme.weights = np.ones(3) / 3

        elif order == 3:
            a = 0.6
            b = 0.2
            c = 1 / 3
            scheme.points = np.array([[c, c, c], [b, a, a], [a, b, a], [a, a, b]])
            scheme.weights = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])

        else:
            raise NotImplementedError("order must be either 1, 2 or 3.")

        triangle = np.array([[0, 0], [1, 0], [0, 1]])
        points = np.dot(triangle.T, scheme.points.T).T

        super().__init__(points, scheme.weights * area)


class Tetrahedron(Scheme):
    def __init__(self, order: int):
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
