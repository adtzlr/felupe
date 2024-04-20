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


class Triangle(Element):
    r"""A 2D triangle element formulation with linear shape functions.

    Notes
    -----
    The triangle element is defined by three points (0-2). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)` [2]_.

    .. math::

       \boldsymbol{h}(r,s) = \begin{bmatrix}
               1-r-s \\
               r \\
               s
           \end{bmatrix}

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.Triangle()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    .. [2] K. J. Bathe, Finite element procedures, 2nd ed. K. J. Bathe, Watertown, MA,
       2014.
    """

    def __init__(self):
        super().__init__(shape=(3, 2))
        self.points = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "triangle"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        r, s = rs
        return np.array([1 - r - s, r, s])

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."
        r, s = rs
        return np.array([[-1, -1], [1, 0], [0, 1]], dtype=float)


class TriangleMINI(Element):
    r"""A 2D triangle element formulation with bubble-enriched linear shape functions.

    Notes
    -----
    The MINI triangle element is defined by four points (0-3). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)`.

    .. math::

       \boldsymbol{h}(r,s) = \begin{bmatrix}
               1-r-s \\
               r \\
               s \\
               r s (1-r-s)
           \end{bmatrix}

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.TriangleMINI()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self, bubble_multiplier=1.0):
        super().__init__(shape=(4, 2))
        self.points = np.array([[0, 0], [1, 0], [0, 1], [1 / 3, 1 / 3]], dtype=float)
        self.cells = np.arange(len(self.points) - 1).reshape(1, -1)
        self.cell_type = "triangle"
        self.bubble_multiplier = bubble_multiplier

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        r, s = rs
        a = self.bubble_multiplier
        return np.array([1 - r - s, r, s, a * r * s * (1 - r - s)])

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."
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
    r"""A 2D triangle element formulation with quadratic shape functions.

    Notes
    -----
    The quadratic triangle element is defined by six points (0-5). The element includes
    three mid-edge points besides the three triangle vertices. The ordering of the three
    points defining the element is point ids (0-2,3-5) where id #3 is the mid-edge point
    between points (0,1); id #4 is the mid-edge point between points (1,2); and id #5 is
    the mid-edge point between points (2,0). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)` [2]_.

    .. math::

       \boldsymbol{h}(r,s) = \begin{bmatrix}
               1-r-s \\
               r \\
               s \\
               4 r (1-r-s) \\
               4 r s \\
               4 s (1-r-s)
           \end{bmatrix}

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticTriangle()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    .. [2] K. J. Bathe, Finite element procedures, 2nd ed. K. J. Bathe, Watertown, MA,
       2014.
    """

    def __init__(self):
        super().__init__(shape=(6, 2))
        self.points = np.zeros(self.shape)
        self.points[:3] = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        self.points[3] = np.mean(self.points[[0, 1]], axis=0)
        self.points[4] = np.mean(self.points[[1, 2]], axis=0)
        self.points[5] = np.mean(self.points[[2, 0]], axis=0)

        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "triangle6"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        r, s = rs
        h = np.array(
            [1 - r - s, r, s, 4 * r * (1 - r - s), 4 * r * s, 4 * s * (1 - r - s)]
        )
        h[0] += -h[3] / 2 - h[5] / 2
        h[1] += -h[3] / 2 - h[4] / 2
        h[2] += -h[4] / 2 - h[5] / 2

        return h

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."
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
