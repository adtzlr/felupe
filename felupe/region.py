# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:29:15 2021

@author: adutz
"""
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

from .math import det, inv


class Region:
    """A numeric region."""

    def __init__(self, mesh, element, quadrature):
        """A numeric region, created by a combination of a mesh,
        an element and a numeric integration scheme (quadrature).
        """
        self.mesh = mesh
        self.element = element
        self.quadrature = quadrature

        # array with degrees of freedom
        # h_bp
        # ----
        # basis function "b" evaluated at quadrature point "p"
        self.h = np.array([self.element.basis(p) for p in self.quadrature.points]).T

        # dhdr_bJp
        # --------
        # partial derivative of basis function "b"
        # w.r.t. natural coordinate "J" evaluated at quadrature point "p"
        self.dhdr = np.array(
            [self.element.basisprime(p) for p in self.quadrature.points]
        ).transpose(1, 2, 0)

        if self.element.nbasis > 1 and self.mesh.ndim == self.element.ndim:

            # dXdr_IJpe and its inverse drdX_IJpe
            # -----------------------------------
            # geometric gradient as partial derivative of undeformed coordinate "I"
            # w.r.t. natural coordinate "J" evaluated at quadrature point "p"
            # for every cell "c"
            dXdr = np.einsum(
                "cbI,bJp->IJpc", self.mesh.points[self.mesh.cells], self.dhdr
            )
            drdX = inv(dXdr)

            # dV_pe = det(dXdr)_pc * w_p
            # determinant of geometric gradient evaluated at quadrature point "p"
            # for every cell "c" multiplied by corresponding quadrature weight
            # denoted as "differential volume element"
            self.dV = det(dXdr) * self.quadrature.weights.reshape(-1, 1)

            # dhdX_bJpc
            # ---------
            # partial derivative of basis function "b"
            # w.r.t. undeformed coordinate "J" evaluated at quadrature point "p"
            # for every cell "c"
            self.dhdX = np.einsum("bIp,IJpc->bJpc", self.dhdr, drdX)

    def volume(self, detF=1):
        "Calculate cell volume for cell 'c'."
        return np.einsum("pc->c", detF * self.dV)
