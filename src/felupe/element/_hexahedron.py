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
    base and (4,5,6,7) the opposite quad. [1]

    The shape function :math:`h` in terms of the coordinates :math:`(r,s,t)` is constant
    and hence, its gradient is zero.

    .. math::

       h(r,s,t) = 1

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.ConstantHexahedron()
       >>> element.plot().show()

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
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "hexahedron"
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
    base and (4,5,6,7) the opposite quad. [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s,t)`.

    .. math::

       \boldsymbol{h}(r,s,t) = \frac{1}{8} \begin{bmatrix}
               (1-r) (1-s) (1-t) \\
               (1+r) (1-s) (1-t) \\
               (1+r) (1+s) (1-t) \\
               (1-r) (1+s) (1-t) \\
               (1-r) (1-s) (1+t) \\
               (1+r) (1-s) (1+t) \\
               (1+r) (1+s) (1+t) \\
               (1-r) (1+s) (1+t)
           \end{bmatrix}

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.Hexahedron()
       >>> element.plot().show()

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
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "hexahedron"
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
    r"""A 3D hexahedron (brick) element formulation with quadratic (serendipity) shape
    functions.

    Notes
    -----
    The hexahedron element is defined by twenty points with eight corner points (0-7)
    where (0,1,2,3) forms the base and (4,5,6,7) the opposite quad; followed by 12 mid-
    edge points. The mid-edge points correspond to the edges defined by the lines
    between the points (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4),
    (1,5), (2,6), (3,7). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s,t)`..

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticHexahedron()
       >>> element.plot().show()

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
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "hexahedron20"
        super().__init__(shape=(20, 3))

    def function(self, rst):
        "Return the shape functions at given coordinates (r, s, t)."
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
        "Return the gradient of shape functions at given coordinates (r, s, t)."
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
    r"""A 3D hexahedron (brick) element formulation with tri-quadratic shape functions.

    Notes
    -----
    The hexahedron element is defined by 27 points. This includes 8 corner points, 12
    mid-edge points, 6 mid-face points and one mid-volume point. The ordering of the 27
    points defining the element is point ids (0-7,8-19, 20-25, 26) where point ids 0-7
    are the eight corner vertices of the cube; followed by twelve midedge points (8-19);
    followed by 6 mid-face points (20-25) and the last point (26) is the mid-volume
    point. Note that these mid-edge points correspond to the edges defined by (0,1),
    (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7). The
    mid-surface points lie on the faces defined by (first edge point id's, than mid-edge
    point id's): (0,1,5,4;8,17,12,16), (1,2,6,5;9,18,13,17), (2,3,7,6,10,19,14,18),
    (3,0,4,7;11,16,15,19), (0,1,2,3;8,9,10,11), (4,5,6,7;12,13,14,15). The last point
    lies in the center (0,1,2,3,4,5,6,7). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s,t)`.

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.TriQuadraticHexahedron()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

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
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "hexahedron27"

        self._lagrange = ArbitraryOrderLagrange(order=2, dim=3, permute=False)

        self._vertices = np.array([0, 2, 8, 6, 18, 20, 26, 24])
        self._edges = np.array([1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15])
        self._faces = np.array([12, 14, 10, 16, 4, 22])
        self._volume = np.array([13])

        self._permute = np.concatenate(
            (self._vertices, self._edges, self._faces, self._volume)
        )

    def function(self, rst):
        "Return the shape functions at given coordinates (r, s, t)."

        return self._lagrange.function(rst)[self._permute]

    def gradient(self, rst):
        "Return the gradient of shape functions at given coordinates (r, s, t)."

        return self._lagrange.gradient(rst)[self._permute, :]
