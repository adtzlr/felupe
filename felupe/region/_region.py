# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from ..math import det, inv


class Region:
    """A numeric region."""

    def __init__(self, mesh, element, quadrature, grad=True):
        """A numeric region as a combination of a `mesh`,
        an `element` and a numeric integration scheme (`quadrature`).
        The gradients of the element shape functions are evaluated at
        all integration points of each cell in the region if the
        optional argument `grad` is True (default is True).
        """

        self.mesh = mesh
        self.element = element
        self.quadrature = quadrature

        # element shape function "a" evaluated at quadrature point "p"
        #
        # h_ap
        self.h = np.array([self.element.function(p) for p in self.quadrature.points]).T

        # partial derivative of element shape function "a"
        # w.r.t. natural element coordinate "J" evaluated at quadrature point "p"
        #
        # dhdr_aJp
        self.dhdr = np.array(
            [self.element.gradient(p) for p in self.quadrature.points]
        ).transpose(1, 2, 0)

        if grad:

            # geometric gradient as partial derivative of undeformed coordinate "I"
            # w.r.t. natural element coordinate "J" evaluated at quadrature point "p"
            # for every cell "c" (geometric gradient or
            # **Jacobian** transformation between "X" and "r")
            #
            # dXdr_IJpe
            self.dXdr = np.einsum(
                "caI,aJp->IJpc", self.mesh.points[self.mesh.cells], self.dhdr
            )

            # inverse of dXdr
            self.drdX = inv(self.dXdr)

            # Determinant of geometric gradient evaluated at quadrature point "p"
            # for every cell "c" multiplied by corresponding quadrature weight
            # according to integration point "p", denoted as
            # "differential volume element"
            #
            # dV_pc = det(dXdr)_pc * w_p
            self.dV = det(self.dXdr) * self.quadrature.weights.reshape(-1, 1)

            # Partial derivative of element shape function "a"
            # w.r.t. undeformed coordinate "J" evaluated at quadrature point "p"
            # for every cell "c"
            #
            # dhdX_aJpc
            self.dhdX = np.einsum("aIp,IJpc->aJpc", self.dhdr, self.drdX)
