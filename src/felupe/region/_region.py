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
    r"""
    A numeric region as a combination of a mesh, an element and a numeric
    integration scheme (quadrature). The gradients of the element shape
    functions are evaluated at all integration points of each cell in the
    region if the optional gradient argument is True.

    .. math::

       \frac{\partial X^I}{\partial r^J} &= X_a^I \frac{\partial h_a}{\partial r^J}

       \frac{\partial h_a}{\partial X^J} &= \frac{\partial h_a}{\partial r^I} \frac{\partial r^I}{\partial X^J}

       dV &= \det\left(\frac{\partial X^I}{\partial r^J}\right) w


    Parameters
    ----------
    mesh : Mesh
        A mesh with points and cells.
    element : Element
        The finite element formulation to be applied on the cells.
    quadrature: Quadrature
        An element-compatible numeric integration scheme with points and weights.
    grad : bool, optional
        A flag to invoke gradient evaluation (default is True).

    Attributes
    ----------
    mesh : Mesh
        A mesh with points and cells.
    element : Finite element
        The finite element formulation to be applied on the cells.
    quadrature: Quadrature scheme
        An element-compatible numeric integration scheme with points and weights.
    h : ndarray
        Element shape function array ``h_ap`` of shape function ``a`` evaluated at quadrature point ``p``.
    dhdr : ndarray
        Partial derivative of element shape function array ``dhdr_aJp`` with shape function ``a`` w.r.t. natural element coordinate ``J`` evaluated at quadrature point ``p`` for every cell ``c`` (geometric gradient or **Jacobian** transformation between ``X`` and ``r``).
    dXdr : ndarray
        Geometric gradient ``dXdr_IJpc`` as partial derivative of undeformed coordinate ``I`` w.r.t. natural element coordinate ``J`` evaluated at quadrature point ``p`` for every cell ``c`` (geometric gradient or **Jacobian** transformation between ``X`` and ``r``).
    drdX : ndarray
        Inverse of dXdr.
    dV : ndarray
        Numeric *Differential volume element* as product of determinant of geometric gradient  ``dV_pc = det(dXdr)_pc w_p`` and quadrature weight ``w_p``, evaluated at quadrature point ``p`` for every cell ``c``.
    dhdX : ndarray
        Partial derivative of element shape functions ``dhdX_aJpc`` of shape function ``a`` w.r.t. undeformed coordinate ``J`` evaluated at quadrature point ``p`` for every cell ``c``.
    """

    def __init__(self, mesh, element, quadrature, grad=True):

        self.mesh = mesh
        self.element = element
        self.quadrature = quadrature

        # element shape function
        self.element.h = np.array(
            [self.element.function(p) for p in self.quadrature.points]
        ).T
        self.h = np.tile(np.expand_dims(self.element.h, -1), self.mesh.ncells)

        # partial derivative of element shape function
        self.element.dhdr = np.array(
            [self.element.gradient(p) for p in self.quadrature.points]
        ).transpose(1, 2, 0)
        self.dhdr = np.tile(np.expand_dims(self.element.dhdr, -1), self.mesh.ncells)

        if grad:

            # geometric gradient
            self.dXdr = np.einsum(
                "caI,aJpc->IJpc", self.mesh.points[self.mesh.cells], self.dhdr
            )

            # inverse of dXdr
            self.drdX = inv(self.dXdr)

            # numeric **differential volume element**
            self.dV = det(self.dXdr) * self.quadrature.weights.reshape(-1, 1)

            # Partial derivative of element shape function
            # w.r.t. undeformed coordinates
            self.dhdX = np.einsum("aIpc,IJpc->aJpc", self.dhdr, self.drdX)
