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


class ConstantQuad(Element):
    r"""A 2D quadrilateral element formulation with constant shape functions.

    Notes
    -----
    The quadrilateral element is defined by four points (0-3). [1]

    The shape function :math:`h` is given in terms of the coordinates :math:`(r,s)`.

    .. math::

       h(r,s) = 1

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.ConstantQuad()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self):
        super().__init__(shape=(1, 2))
        self.points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "quad"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        return np.array([1])

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."
        return np.array([[0, 0]])


class Quad(Element):
    r"""A 2D quadrilateral element formulation with linear shape functions.

    Notes
    -----
    The quadrilateral element is defined by four points (0-3). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)`.

    .. math::

       \boldsymbol{h}(r,s) = \frac{1}{4} \begin{bmatrix}
               (1-r) (1-s) \\
               (1+r) (1-s) \\
               (1+r) (1+s) \\
               (1-r) (1+s)
           \end{bmatrix}

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.Quad()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self):
        super().__init__(shape=(4, 2))
        self.points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "quad"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        r, s = rs
        return (
            np.array(
                [
                    (1 - r) * (1 - s),
                    (1 + r) * (1 - s),
                    (1 + r) * (1 + s),
                    (1 - r) * (1 + s),
                ]
            )
            * 0.25
        )

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."

        r, s = rs
        return (
            np.array(
                [
                    [-(1 - s), -(1 - r)],
                    [(1 - s), -(1 + r)],
                    [(1 + s), (1 + r)],
                    [-(1 + s), (1 - r)],
                ]
            )
            * 0.25
        )


class QuadraticQuad(Element):
    r"""A 2D quadrilateral element formulation with quadratic (serendipity) shape
    functions.

    Notes
    -----
    The quadratic (serendipity) quadrilateral element is defined by eight points (0-7).
    [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)`.

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticQuad()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self):
        super().__init__(shape=(8, 2))
        self.points = np.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
                [0, -1],
                [1, 0],
                [0, 1],
                [-1, 0],
            ],
            dtype=float,
        )
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "quad8"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        r, s = rs
        ra, sa = self.points.T

        h = (1 + ra * r) * (1 + sa * s) * (ra * r + sa * s - 1) / 4
        h[ra == 0] = (1 - r**2) * (1 + sa[ra == 0] * s) / 2
        h[sa == 0] = (1 + ra[sa == 0] * r) * (1 - s**2) / 2

        return h

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."

        r, s = rs
        ra, sa = self.points.T

        dhdr = (
            ra * (1 + sa * s) * (ra * r + sa * s - 1) / 4
            + (1 + ra * r) * (1 + sa * s) * ra / 4
        )

        dhdr[ra == 0] = -2 * r * (1 + sa[ra == 0] * s) / 2
        dhdr[sa == 0] = ra[sa == 0] * (1 - s**2) / 2

        dhds = (1 + ra * r) * sa * (ra * r + sa * s - 1) / 4 + (1 + ra * r) * (
            1 + sa * s
        ) * sa / 4

        dhds[ra == 0] = (1 - r**2) * sa[ra == 0] / 2
        dhds[sa == 0] = (1 + ra[sa == 0] * r) * -2 * s / 2

        return np.vstack([dhdr, dhds]).T


class BiQuadraticQuad(Element):
    r"""A 2D quadrilateral element formulation with bi-quadratic shape functions.

    Notes
    -----
    The bi-quadratic quadrilateral element is defined by nine points (0-8). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)`.

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.BiQuadraticQuad()
       >>> element.plot().show()

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self):
        super().__init__(shape=(9, 2))

        self._lagrange = ArbitraryOrderLagrange(order=2, dim=2, permute=False)

        self._vertices = np.array([0, 2, 8, 6])
        self._edges = np.array([1, 5, 7, 3])
        self._faces = np.array([4])
        self._volume = np.array([], dtype=int)

        self._permute = np.concatenate(
            (self._vertices, self._edges, self._faces, self._volume)
        )

        self.points = self._lagrange.points[self._permute]
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "quad9"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."

        return self._lagrange.function(rs)[self._permute]

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."

        return self._lagrange.gradient(rs)[self._permute, :]
