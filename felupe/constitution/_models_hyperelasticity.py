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

from ..math import (
    dot,
    ddot,
    transpose,
    inv,
    dya,
    cdya_ik,
    cdya_il,
    det,
    identity,
    trace,
)


class NeoHooke:
    r"""Nearly-incompressible isotropic hyperelastic Neo-Hooke material
    formulation. The strain energy density function of the Neo-Hookean
    material formulation is a linear function of the trace of the
    isochoric part of the right Cauchy-Green deformation tensor.

    In a nearly-incompressible constitutive framework the strain energy
    density is an additive composition of an isochoric and a volumetric
    part. While the isochoric part is defined on the distortional part of
    the deformation gradient, the volumetric part of the strain
    energy function is defined on the determinant of the deformation
    gradient.

    .. math::

        \psi &= \hat{\psi}(\hat{\boldsymbol{C}}) + U(J)

        \hat\psi(\hat{\boldsymbol{C}}) &= \frac{\mu}{2} (\text{tr}(\hat{\boldsymbol{C}}) - 3)

    with

    .. math::

       J &= \text{det}(\boldsymbol{F})

       \hat{\boldsymbol{F}} &= J^{-1/3} \boldsymbol{F}

       \hat{\boldsymbol{C}} &= \hat{\boldsymbol{F}}^T \hat{\boldsymbol{F}}

    The volumetric part of the strain energy density function is a function
    the volume ratio.

    .. math::

       U(J) = \frac{K}{2} (J - 1)^2

    The first Piola-Kirchhoff stress tensor is evaluated as the gradient
    of the strain energy density function. The hessian of the strain
    energy density function enables the corresponding elasticity tensor.

    .. math::

       \boldsymbol{P} &= \frac{\partial \psi}{\partial \boldsymbol{F}}

       \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}

    A chain rule application leads to the following expression for the stress tensor.
    It is formulated as a sum of the **physical**-deviatoric (not the mathematical deviator!) and the physical-hydrostatic stress tensors.

    .. math::

       \boldsymbol{P} &= \boldsymbol{P}' + \boldsymbol{P}_U

       \boldsymbol{P}' &= \frac{\partial \hat{\psi}}{\partial \hat{\boldsymbol{F}}} : \frac{\partial \hat{\boldsymbol{F}}}{\partial \boldsymbol{F}} = \bar{\boldsymbol{P}} - \frac{1}{3} (\bar{\boldsymbol{P}} : \boldsymbol{F}) \boldsymbol{F}^{-T}

       \boldsymbol{P}_U &= \frac{\partial U(J)}{\partial J} \frac{\partial J}{\partial \boldsymbol{F}} = U'(J) J \boldsymbol{F}^{-T}

    with

    .. math::

       \frac{\partial \hat{\boldsymbol{F}}}{\partial \boldsymbol{F}} &= J^{-1/3} \left( \boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I} - \frac{1}{3} \boldsymbol{F} \otimes \boldsymbol{F}^{-T} \right)

       \frac{\partial J}{\partial \boldsymbol{F}} &= J \boldsymbol{F}^{-T}

       \bar{\boldsymbol{P}} &= J^{-1/3} \frac{\partial \hat{\psi}}{\partial \hat{\boldsymbol{F}}}

    With the above partial derivatives the first Piola-Kirchhoff stress
    tensor of the Neo-Hookean material model takes the following form.

    .. math::

       \boldsymbol{P} = \mu J^{-2/3} \left( \boldsymbol{F} - \frac{1}{3} (\boldsymbol{F} : \boldsymbol{F}) \boldsymbol{F}^{-T} \right) + K (J - 1) J \boldsymbol{F}^{-T}

    Again, a chain rule application leads to an expression for the elasticity tensor.

    .. math::

       \mathbb{A} &= \mathbb{A}' + \mathbb{A}_{U}

       \mathbb{A}' &= \bar{\mathbb{A}} - \frac{1}{3} \left( (\bar{\mathbb{A}} : \boldsymbol{F}) \otimes \boldsymbol{F}^{-T} + \boldsymbol{F}^{-T} \otimes (\boldsymbol{F} : \bar{\mathbb{A}}) \right ) + \frac{1}{9} (\boldsymbol{F} : \bar{\mathbb{A}} : \boldsymbol{F}) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T}

       \mathbb{A}_{U} &= (U''(J) J + U'(J)) J \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - U'(J) J \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T}

    with

    .. math::

       \bar{\mathbb{A}} = J^{-1/3} \frac{\partial^2 \hat\psi}{\partial \hat{\boldsymbol{F}}\ \partial \hat{\boldsymbol{F}}} J^{-1/3}

    With the above partial derivatives the (physical-deviatoric and
    -hydrostatic) parts of the elasticity tensor associated
    to the first Piola-Kirchhoff stress tensor of the Neo-Hookean
    material model takes the following form.

    .. math::

       \mathbb{A} &= \mathbb{A}' + \mathbb{A}_{U}

       \mathbb{A}' &= J^{-2/3} \left(\boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I} - \frac{1}{3} \left( \boldsymbol{F} \otimes \boldsymbol{F}^{-T} + \boldsymbol{F}^{-T} \otimes \boldsymbol{F} \right ) + \frac{1}{9} (\boldsymbol{F} : \boldsymbol{F}) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} \right)

       \mathbb{A}_{U} &= K J \left( (2J - 1) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} - (J - 1) \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T} \right)


    Arguments
    ---------
    mu : float
        Shear modulus
    bulk : float
        Bulk modulus

    """

    def __init__(self, mu=None, bulk=None, parallel=False):

        self.parallel = parallel

        self.mu = mu
        self.bulk = bulk

        # aliases for function, gradient and hessian
        self.energy = self.function
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def function(self, x, mu=None, bulk=None):
        """Strain energy density function per unit undeformed volume of the
        Neo-Hookean material formulation.

        Arguments
        ---------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
        """

        F = x[0]

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        C = dot(transpose(F), F, parallel=self.parallel)

        W = mu / 2 * (J ** (-2 / 3) * trace(C) - 3)

        if bulk is not None:
            W += bulk * (J - 1) ** 2 / 2

        return [W]

    def gradient(self, x, mu=None, bulk=None):
        """Gradient of the strain energy density function per unit
        undeformed volume of the Neo-Hookean material formulation.

        Arguments
        ---------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
        """

        F, statevars = x[0], x[-1]

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))

        # "physical"-deviatoric (not math-deviatoric!) part of P
        P = mu * (F - ddot(F, F, parallel=self.parallel) / 3 * iFT) * J ** (-2 / 3)

        if bulk is not None:
            # "physical"-volumetric (not math-volumetric!) part of P
            P += bulk * (J - 1) * J * iFT

        return [P, statevars]

    def hessian(self, x, mu=None, bulk=None):
        """Hessian of the strain energy density function per unit
        undeformed volume of the Neo-Hookean material formulation.

        Arguments
        ---------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
        """

        F = x[0]

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))
        eye = identity(F)

        # "physical"-deviatoric (not math-deviatoric!) part of A4
        A4 = (
            mu
            * (
                cdya_ik(eye, eye, parallel=self.parallel)
                - 2 / 3 * dya(F, iFT, parallel=self.parallel)
                - 2 / 3 * dya(iFT, F, parallel=self.parallel)
                + 2
                / 9
                * ddot(F, F, parallel=self.parallel)
                * dya(iFT, iFT, parallel=self.parallel)
                + 1
                / 3
                * ddot(F, F, parallel=self.parallel)
                * cdya_il(iFT, iFT, parallel=self.parallel)
            )
            * J ** (-2 / 3)
        )

        if bulk is not None:

            p = bulk * (J - 1)
            q = p + bulk * J

            # "physical"-volumetric (not math-volumetric!) part of A4
            A4 += J * (
                q * dya(iFT, iFT, parallel=self.parallel)
                - p * cdya_il(iFT, iFT, parallel=self.parallel)
            )

        return [A4]
