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

import warnings

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

       \frac{\partial h_a}{\partial X^J} &= \frac{\partial h_a}{\partial r^I}
       \frac{\partial r^I}{\partial X^J}

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
        Element shape function array ``h_aq`` of shape function ``a`` evaluated at
        quadrature point ``q``.
    dhdr : ndarray
        Partial derivative of element shape function array ``dhdr_aJq`` with shape
        function ``a`` w.r.t. natural element coordinate ``J`` evaluated at quadrature
        point ``q`` for every cell ``c`` (geometric gradient or **Jacobian**
        transformation between ``X`` and ``r``).
    dXdr : ndarray
        Geometric gradient ``dXdr_IJqc`` as partial derivative of undeformed coordinate
        ``I`` w.r.t. natural element coordinate ``J`` evaluated at quadrature point
        ``q`` for every cell ``c`` (geometric gradient or **Jacobian** transformation
        between ``X`` and ``r``).
    drdX : ndarray
        Inverse of dXdr.
    dV : ndarray
        Numeric *Differential volume element* as product of determinant of geometric
        gradient  ``dV_qc = det(dXdr)_qc w_q`` and quadrature weight ``w_q``, evaluated
        at quadrature point ``q`` for every cell ``c``.
    dhdX : ndarray
        Partial derivative of element shape functions ``dhdX_aJqc`` of shape function
        ``a`` w.r.t. undeformed coordinate ``J`` evaluated at quadrature point ``q`` for
        every cell ``c``.

    Examples
    --------
    >>> from felupe import Cube, Hexahedron, GaussLegendre, Region

    >>> mesh = Cube(n=3)
    >>> element = Hexahedron()
    >>> quadrature = GaussLegendre(order=1, dim=3)

    >>> region = Region(mesh, element, quadrature)
    >>> region
    <felupe Region object>
      Element formulation: Hexahedron
      Quadrature rule: GaussLegendre
      Gradient evaluated: True

    Cell-volumes may be obtained by a sum of the differential volumes located at the
    quadrature points.

    >>> region.dV.sum(axis=0)
    array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

    The partial derivative of the first element shape function w.r.t. the undeformed
    coordinates evaluated at the second integration point of the last element of the
    region:

    >>> region.dhdX[0, :, 1, -1]
    array([-1.24401694, -0.33333333, -0.33333333])

    """

    def __init__(self, mesh, element, quadrature, grad=True):
        self.mesh = mesh
        self.element = element
        self.quadrature = quadrature

        # element shape function
        self.element.h = np.array(
            [self.element.function(q) for q in self.quadrature.points]
        ).T
        self.h = np.tile(np.expand_dims(self.element.h, -1), self.mesh.ncells)

        # partial derivative of element shape function
        self.element.dhdr = np.array(
            [self.element.gradient(q) for q in self.quadrature.points]
        ).transpose(1, 2, 0)
        self.dhdr = np.tile(np.expand_dims(self.element.dhdr, -1), self.mesh.ncells)

        if grad:
            # geometric gradient
            self.dXdr = np.einsum(
                "caI,aJqc->IJqc", self.mesh.points[self.mesh.cells], self.dhdr
            )

            # inverse of dXdr
            self.drdX = inv(self.dXdr)

            # numeric **differential volume element**
            self.dV = det(self.dXdr) * self.quadrature.weights.reshape(-1, 1)

            # check for negative **differential volume elements**
            if np.any(self.dV < 0):
                cells_negative_volume = np.where(np.any(self.dV < 0, axis=0))[0]
                message_negative_volumes = "".join(
                    [
                        f"Negative volumes for cells \n {cells_negative_volume}\n",
                        "Try ``mesh.flip(np.any(region.dV < 0, axis=0))`` ",
                        "and re-create the region.",
                    ]
                )
                warnings.warn(message_negative_volumes)

            # Partial derivative of element shape function
            # w.r.t. undeformed coordinates
            self.dhdX = np.einsum("aIqc,IJqc->aJqc", self.dhdr, self.drdX)

    def __repr__(self):
        header = "<felupe Region object>"
        element = f"  Element formulation: {type(self.element).__name__}"
        quadrature = f"  Quadrature rule: {type(self.quadrature).__name__}"
        grad = f"  Gradient evaluated: {hasattr(self, 'dV')}"

        return "\n".join([header, element, quadrature, grad])
