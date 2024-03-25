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


class Line(Element):
    r"""A 1D line element formulation with linear shape functions.

    Notes
    -----
    The linear line element is defined by two points (0-1). [1]

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r)`.

    .. math::

       \boldsymbol{h}(r) = \frac{1}{2} \begin{bmatrix}
               (1-r) \\
               (1+r)
           \end{bmatrix}

    References
    ----------
    .. [1] W. Schroeder, K. Martin and B. Lorensen. The Visualization
       Toolkit, 4th ed. Kitware, 2006. ISBN: 978-1-930934-19-1.
    """

    def __init__(self):
        super().__init__(shape=(2, 1))
        self.points = np.array([-1, 1], dtype=float).reshape(-1, 1)
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "line"

    def function(self, rv):
        "Return the shape functions at given coordinates (r,)."
        (r,) = rv
        return np.array([(1 - r), (1 + r)]) * 0.5

    def gradient(self, rv):
        "Return the gradient of shape functions at given coordinates (r,)."
        (r,) = rv
        return np.array([[-1], [1]]) * 0.5
