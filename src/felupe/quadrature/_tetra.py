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


class Tetrahedron(Scheme):
    r"""A quadrature scheme suitable for Tetrahedrons of order 1, 2 or 3 on the interval
    :math:`[0, 1]`.

    Parameters
    ----------
    order : int
        The order of the quadrature scheme.

    Notes
    -----

    The approximation is given by

    ..  math::

        \int_V f(x) dV \approx \sum f(x_q) w_q

    with quadrature points :math:`x_q` and corresponding weights :math:`w_q` [1]_.

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.Tetra()
       >>> quadrature = fem.TetrahedronQuadrature(order=1)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()


    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticTetra()
       >>> quadrature = fem.TetrahedronQuadrature(order=2)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticTetra()
       >>> quadrature = fem.TetrahedronQuadrature(order=5)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    References
    ----------
    .. [1] O. C. Zienkiewicz, R. L. Taylor and J. Z. Zhu, The Finite Element Method: Its
       Basis and Fundamentals, 7th ed., Elsevier, 2013.
    """

    def __init__(self, order: int):
        scheme = SimpleNamespace()

        if order == 1:
            scheme.points = np.ones((1, 3)) / 4
            scheme.weights = np.ones(1) / 6

        elif order == 2:
            a = 0.13819660
            b = 0.58541020
            scheme.points = np.array([[a, a, a], [b, a, a], [a, b, a], [a, a, b]])
            scheme.weights = np.ones(4) / 24

        elif order == 3:
            a = 1 / 6
            b = 1 / 2
            c = 1 / 4
            d = 9 / 120
            scheme.points = np.array(
                [[a, a, a], [b, a, a], [a, b, a], [b, b, a], [c, c, c]]
            )
            scheme.weights = np.array([d, d, d, d, -4 / 30])

        elif order == 5:
            a = 1 / 2
            b = 0.6984197043243866
            c = 0.1005267652252045
            d = 0.0568813795204234
            e = 0.31437287349319221
            f = 0.019047619047619
            g = 0.0885898247429807
            h = 0.1328387466855907
            scheme.points = np.array(
                [
                    [0, a, a, a, 0, 0, b, c, c, c, d, e, e, e],
                    [a, 0, a, 0, a, 0, c, c, c, b, e, e, e, d],
                    [a, a, 0, 0, 0, a, c, c, b, c, e, e, d, e],
                ]
            ).T
            scheme.weights = np.array([f, f, f, f, f, f, g, g, g, g, h, h, h, h]) / 6

        else:
            raise NotImplementedError("order must be either 1, 2, 3 or 5.")

        super().__init__(scheme.points, scheme.weights)
