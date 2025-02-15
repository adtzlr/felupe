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


class Vertex(Element):
    r"""A vertex element formulation with constant shape functions.

    Notes
    -----
    The vertex element is defined by one point.
    """

    def __init__(self):
        self.points = np.array([[0.0]])
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "vertex"

    def function(self, r):
        "Return the shape functions at given coordinate (r)."
        return np.ones(1)

    def gradient(self, r):
        "Return the gradient of shape functions at given coordinate (r)."
        return np.zeros((1, 1))

    def hessian(self, rs):
        "Return the hessian of shape functions at given coordinate (r)."
        return np.zeros((1, 1, 1))
