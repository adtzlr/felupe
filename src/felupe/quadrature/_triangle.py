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

        \int_V f(x) dV \approx \sum f(x_q) w_q

    with quadrature points :math:`x_q` and corresponding weights :math:`w_q`.

    """

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
