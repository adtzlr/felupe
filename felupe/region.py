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
    def __init__(self, mesh, element, quadrature):
        self.mesh = mesh
        self.element = element
        self.quadrature = quadrature
        self.connectivity = mesh.connectivity
        self.nodes = mesh.nodes
        self.nnodes = mesh.nnodes
        self.nelements = mesh.nelements

        # array with degrees of freedom
        # h_ap
        # ----
        # basis function "a" evaluated at quadrature point "p"
        self.h = np.array([self.element.basis(p) for p in self.quadrature.points]).T

        # dhdr_aJp
        # --------
        # partial derivative of basis function "a"
        # w.r.t. natural coordinate "J" evaluated at quadrature point "p"
        self.dhdr = np.array(
            [self.element.basisprime(p) for p in self.quadrature.points]
        ).transpose(1, 2, 0)

        if self.element.nbasis > 1:

            # dXdr_IJpe and its inverse drdX_IJpe
            # -----------------------------------
            # geometric gradient as partial derivative of undeformed coordinate "I"
            # w.r.t. natural coordinate "J" evaluated at quadrature point "p"
            # for every element "e"
            dXdr = np.einsum(
                "eaI,aJp->IJpe", self.mesh.nodes[self.connectivity], self.dhdr
            )
            drdX = inv(dXdr)

            # dV_pe = det(dXdr)_pe * w_p
            # determinant of geometric gradient evaluated at quadrature point "p"
            # for every element "e" multiplied by corresponding quadrature weight
            # denoted as "differential volume element"
            self.dV = det(dXdr) * self.quadrature.weights.reshape(-1, 1)

            # dhdX_aJpe
            # ---------
            # partial derivative of basis function "a"
            # w.r.t. undeformed coordinate "J" evaluated at quadrature point "p"
            # for every element "e"
            self.dhdX = np.einsum("aIp,IJpe->aJpe", self.dhdr, drdX)

    def volume(self, detF=1):
        "Calculate element volume for element 'e'."
        return np.einsum("pe->e", detF * self.dV)


# TODO
# class AxiRegion(Region):

#     def __init__(self, mesh, element, quadrature):
#         super.__init__(mesh, element, quadrature)

#         self.R = np.einsum("ea,ap->pe", self.nodes[:,[1,]][self.connectivity], self.h)
#         zeros = np.zeros(self.dhdX.shape[-2:])
#         h = np.tile(self.h.reshape(*self.h.shape,-1), self.nelements)
#         self.dhdX = np.block([[self.dhdX, zeros],
#                               [zeros.T, h/self.R]])
