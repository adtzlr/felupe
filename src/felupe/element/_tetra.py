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


class Tetra(Element):
    r"""A 3D tetrahedron element formulation with linear shape functions.

    Notes
    -----
    The tetrahedron element is defined by four points (0-3). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s,t)`.

    .. math::

       \boldsymbol{h}(r,s,t) = \begin{bmatrix}
               1-r-s-t \\
               r \\
               s \\
               t
           \end{bmatrix}

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.Tetra()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self):
        super().__init__(shape=(4, 3))
        self.points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "tetra"

    def function(self, rst):
        "Return the shape functions at given coordinates (r, s, t)."
        r, s, t = rst
        return np.array([1 - r - s - t, r, s, t])

    def gradient(self, rst):
        "Return the gradient of shape functions at given coordinates (r, s, t)."
        r, s, t = rst
        return np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


class TetraMINI(Element):
    r"""A 3D tetrahedron element formulation with bubble-enriched linear shape
    functions.

    Notes
    -----
    The MINI tetrahedron element is defined by five points (0-4). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s,t)`.

    .. math::

       \boldsymbol{h}(r,s,t) = \begin{bmatrix}
               1-r-s-t \\
               r \\
               s \\
               t \\
               r s t (1-r-s-t)
           \end{bmatrix}

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.TetraMINI()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self, bubble_multiplier=1.0):
        super().__init__(shape=(5, 3))
        self.points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]],
            dtype=float,
        )
        self.cells = np.arange(len(self.points) - 1).reshape(1, -1)
        self.cell_type = "tetra"
        self.bubble_multiplier = bubble_multiplier

    def function(self, rst):
        "Return the shape functions at given coordinates (r, s, t)."
        r, s, t = rst
        a = self.bubble_multiplier
        return np.array([1 - r - s - t, r, s, t, a * r * s * t * (1 - r - s - t)])

    def gradient(self, rst):
        "Return the gradient of shape functions at given coordinates (r, s, t)."
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
    r"""A 3D tetrahedron element formulation with quadratic shape functions.

    Notes
    -----
    The quadratic tetrahedron element is defined by ten points (0-9). The element
    includes a mid-edge point on each of the edges of the tetrahedron. The ordering of
    the ten points defining the cell is point ids (0-3,4-9) where ids 0-3 are the four
    tetra vertices; and point ids 4-9 are the mid-edge points between (0,1), (1,2),
    (2,0), (0,3), (1,3), and (2,3). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s,t)`.

    .. math::

       \boldsymbol{h}(r,s,t) = \begin{bmatrix}
               t_1 (2 t_1 - 1) \\
               t_2 (2 t_2 - 1) \\
               t_3 (2 t_3 - 1) \\
               t_4 (2 t_4 - 1) \\
               4 t_1 t_2 \\
               4 t_2 t_3 \\
               4 t_3 t_1 \\
               4 t_1 t_4 \\
               4 t_2 t_4 \\
               4 t_3 t_4
           \end{bmatrix}

    with

    .. math::

       t_1 &= 1 - r - s - t

       t_2 &= r

       t_3 &= s

       t_4 &= t

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticTetra()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

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

        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "tetra10"

    def function(self, rst):
        "Return the shape functions at given coordinates (r, s, t)."
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
        "Return the gradient of shape functions at given coordinates (r, s, t)."
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
