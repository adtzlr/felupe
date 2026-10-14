# SPDX-License-Identifier: GPL-3.0-or-later

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
