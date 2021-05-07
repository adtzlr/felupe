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
from scipy.sparse import csr_matrix as sparsematrix
from scipy.sparse import bmat, vstack
import meshio

from copy import deepcopy as copy

from .math import det, inv, interpolate, dot, transpose, eigvals


class Region:
    def __init__(self, mesh, element, quadrature):
        self.mesh = copy(mesh)
        self.element = element
        self.quadrature = quadrature

        if element.nbasis > 1:
            self.connectivity = self.mesh.connectivity[:, : element.nbasis]
            self.mesh.update(self.connectivity)
            self.nodes = self.mesh.nodes
            self.nnodes = self.mesh.nnodes
        else:
            self.nodes = np.stack(
                [
                    np.mean(self.mesh.nodes[conn], axis=0)
                    for conn in self.mesh.connectivity
                ]
            )
            self.nnodes = self.nodes.shape[0]
            self.connectivity = np.arange(self.mesh.nelements).reshape(-1, 1)
            self.mesh.nodes = self.nodes
            self.mesh.update(self.connectivity)

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
