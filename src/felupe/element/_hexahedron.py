# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from ._base import Element
from ._lagrange import ArbitraryOrderLagrange


class ConstantHexahedron(Element):
    r"""A 3D hexahedron (brick) element formulation with constant shape functions.

    Notes
    -----
    The hexahedron element is defined by eight points (0-7) where (0,1,2,3) forms the
    base and (4,5,6,7) the opposite quad. [1]_

    The shape function :math:`h` in terms of the coordinates :math:`(r,s,t)` is constant
    and hence, its gradient is zero.

    .. math::

       h(r,s,t) = 1

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """
    def __init__(self):
        self.points = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        super().__init__(shape=(1, 3))

    def function(self, rst):
        "Return the shape functions at given coordinates (r, s, t)."
        return np.array([1])

    def gradient(self, rst):
        "Return the gradient of shape functions at given coordinates (r, s, t)."
        return np.array([[0, 0, 0]])


class Hexahedron(Element):
    r"""A 3D hexahedron (brick) element formulation with linear shape functions.

    Notes
    -----
    The hexahedron element is defined by eight points (0-7) where (0,1,2,3) forms the
    base and (4,5,6,7) the opposite quad. [1]_

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates 
    :math:`(r,s,t)`.

    .. math::

       \boldsymbol{h}(r,s,t) = \begin{bmatrix}
               (1-r) (1-s) (1-t) \\
               (1+r) (1-s) (1-t) \\
               (1+r) (1+s) (1-t) \\
               (1-r) (1+s) (1-t) \\
               (1-r) (1-s) (1+t) \\
               (1+r) (1-s) (1+t) \\
               (1+r) (1+s) (1+t) \\
               (1-r) (1+s) (1+t)
           \end{bmatrix}

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self):
        self.points = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        super().__init__(shape=(8, 3))

    def function(self, rst):
        "Return the shape functions at given coordinates (r, s, t)."
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

    def gradient(self, rst):
        "Return the gradient of shape functions at given coordinates (r, s, t)."
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


class QuadraticHexahedron(Element):
    def __init__(self):
        self.points = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                #
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
                #
                [0, -1, -1],
                [1, 0, -1],
                [0, 1, -1],
                [-1, 0, -1],
                #
                [0, -1, 1],
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                #
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
            ],
            dtype=float,
        )
        super().__init__(shape=(20, 3))

    def function(self, rst):
        "quadratic serendipity hexahedron shape functions"
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

    def gradient(self, rst):
        "quadratic serendipity hexahedron gradient of shape functions"
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


class TriQuadraticHexahedron(Element):
    def __init__(self):
        super().__init__(shape=(27, 3))

        self.points = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                #
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
                #
                [0, -1, -1],
                [1, 0, -1],
                [0, 1, -1],
                [-1, 0, -1],
                #
                [0, -1, 1],
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                #
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
                #
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
                #
                [0, 0, 0],
            ],
            dtype=float,
        )

        self._lagrange = ArbitraryOrderLagrange(order=2, dim=3)

        self._vertices = np.array([0, 2, 8, 6, 18, 20, 26, 24])
        self._edges = np.array([1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15])
        self._faces = np.array([12, 14, 10, 16, 4, 22])
        self._volume = np.array([13])

        self._permute = np.concatenate(
            (self._vertices, self._edges, self._faces, self._volume)
        )

    def function(self, rst):
        "quadratic hexahedron shape functions"

        return self._lagrange.function(rst)[self._permute]

    def gradient(self, rst):
        "quadratic hexahedron gradient of shape functions"

        return self._lagrange.gradient(rst)[self._permute, :]
