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
from string import ascii_lowercase

from . import Scheme


class GaussLegendre(Scheme):
    "A n-dimensional Gauss-Legendre quadrature rule."

    def __init__(self, order: int, dim: int, permute: bool = True):
        """Arbitrary `order` Gauss-Legendre quadrature rule of `dim` 1, 2 or 3
        on the interval [-1, 1] as approximation of

            ∫ f(x) dx ≈ ∑ f(x_a) w_a                                    (1)

        with `a` quadrature points `x_a` and corresponding weights `w_a`.

        """
        # integration point weights and coordinates

        if dim not in [1, 2, 3]:
            raise ValueError("Wrong dimension.")

        x, w = np.polynomial.legendre.leggauss(1 + order)

        points = (
            np.stack(np.meshgrid(*([x] * dim), indexing="ij"))[::-1].reshape(dim, -1).T
        )

        idx = list(ascii_lowercase)[:dim]
        weights = np.einsum(", ".join(idx), *([w] * dim)).ravel()

        if permute and order == 1 and dim == 2:
            points = points[[0, 1, 3, 2]]
            weights = weights[[0, 1, 3, 2]]

        if permute and order == 1 and dim == 3:
            points = points[[0, 1, 3, 2, 4, 5, 7, 6]]
            weights = weights[[0, 1, 3, 2, 4, 5, 7, 6]]

        if permute and order == 2 and dim == 3:
            vertices = np.array([0, 2, 8, 6, 18, 20, 26, 24])
            edges = np.array([1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15])
            faces = np.array([12, 14, 10, 16, 4, 22])
            volume = np.array([13])

            permute = np.concatenate((vertices, edges, faces, volume))

            points = points[permute]
            weights = weights[permute]

        super().__init__(points, weights)

    def inv(self):
        "Return the inverse quadrature scheme."

        points = self.points.copy()
        points[self.points != 0] = 1 / points[self.points != 0]

        return Scheme(points, self.weights)


class GaussLegendreBoundary(GaussLegendre):
    "A n-dimensional Gauss-Legendre quadrature rule on boundaries."

    def __init__(self, order: int, dim: int, permute: bool = True):
        """Arbitrary `order` Gauss-Legendre quadrature rule of `dim` 1, 2 or 3
        on the interval [-1, 1] as approximation of

            ∫ f(x) dx ≈ ∑ f(x_a) w_a                                    (1)

        with `a` quadrature points `x_a` and corresponding weights `w_a`.

        """

        super().__init__(order=order, dim=dim - 1, permute=permute)

        # reset dimension
        self.dim = dim

        if self.dim == 2 or self.dim == 3:
            # quadrature points projected onto first edge of a quad
            #                          or onto first face of a hexahedron
            self.points = np.hstack((self.points, -np.ones((len(self.points), 1))))

        else:
            raise ValueError("Wrong dimension.")
