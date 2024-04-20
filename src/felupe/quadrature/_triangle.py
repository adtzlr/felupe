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

from types import SimpleNamespace

import numpy as np

from ._scheme import Scheme


class Triangle(Scheme):
    r"""A quadrature scheme suitable for Triangles of order 1, 2 or 3 on the interval
    :math:`[0, 1]`.

    Parameters
    ----------
    order : int
        The order of the quadrature scheme.

    Notes
    -----

    The approximation is given by

    ..  math::

        \int_A f(x) dA \approx \sum f(x_q) w_q

    with quadrature points :math:`x_q` and corresponding weights :math:`w_q` [1]_ [2]_.

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.Triangle()
       >>> quadrature = fem.TriangleQuadrature(order=1)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()


    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticTriangle()
       >>> quadrature = fem.TriangleQuadrature(order=2)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticTriangle()
       >>> quadrature = fem.TriangleQuadrature(order=5)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    References
    ----------
    .. [1] K. J. Bathe, Finite element procedures, 2nd ed. K. J. Bathe, Watertown, MA,
       2014.
    .. [2] O. C. Zienkiewicz, R. L. Taylor and J. Z. Zhu, The Finite Element Method: Its
       Basis and Fundamentals, 7th ed., Elsevier, 2013.
    """

    def __init__(self, order: int):
        scheme = SimpleNamespace()

        if order == 1:
            scheme.points = np.ones((1, 2)) / 3
            scheme.weights = np.ones(1) / 2

        elif order == 2:
            a = 1 / 6
            b = 2 / 3
            scheme.points = np.array([[a, a], [b, a], [a, b]])
            scheme.weights = np.ones(3) / 6

        elif order == 3:
            a = 3 / 5
            b = 1 / 5

            c = 1 / 3
            d = 25 / 96
            scheme.points = np.array([[c, c], [a, b], [b, a], [b, b]])
            scheme.weights = np.array([-9 / 32, d, d, d])

        elif order == 5:
            a = 0.1012865073235
            b = 0.7974269853531
            c = 0.4701420641051
            d = 0.0597158717898
            e = 0.3333333333333
            f = 0.1259391805448
            g = 0.1323941527885
            h = 0.225
            scheme.points = np.array(
                [[a, a], [b, a], [a, b], [c, d], [c, c], [d, c], [e, e]]
            )
            scheme.weights = np.array([f, f, f, g, g, g, h]) / 2

        else:
            raise NotImplementedError("order must be 1, 2, 3 or 5.")

        super().__init__(scheme.points, scheme.weights)
