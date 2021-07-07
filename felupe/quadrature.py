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

import quadpy


class GaussLegendre:
    def __init__(self, order, dim):
        # integration point weights and coordinates
        self.dim = dim
        self.order = order
        self.npoints = (order + 1) ** dim

        if dim == 3:
            scheme = quadpy.c3.product(quadpy.c1.gauss_legendre(order + 1))
        elif dim == 2:
            scheme = quadpy.c2.product(quadpy.c1.gauss_legendre(order + 1))
        elif dim == 1:
            scheme = quadpy.c1.gauss_legendre(order + 1)
        else:
            raise ValueError("Wrong dimension.")

        self.points = scheme.points.T
        self.weights = scheme.weights * 2 ** dim


class Constant(GaussLegendre):
    def __init__(self, dim):
        super().__init__(order=0, dim=dim)


class Linear(GaussLegendre):
    def __init__(self, dim):
        super().__init__(order=1, dim=dim)


class Quadratic(GaussLegendre):
    def __init__(self, dim):
        super().__init__(order=2, dim=dim)


class Scheme:
    def __init__(self, points, weights):
        self.points = points
        self.weights = weights
        self.npoints, self.dim = self.points.shape


class QuadPyScheme(Scheme):
    def __init__(self, scheme):
        super().__init__(scheme.points.T, scheme.weights)
