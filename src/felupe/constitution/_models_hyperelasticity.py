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

from ..math import (
    cdya_ik,
    cdya_il,
    ddot,
    det,
    dot,
    dya,
    identity,
    inv,
    trace,
    transpose,
)
from ._base import ConstitutiveMaterial


class NeoHooke(ConstitutiveMaterial):
    r"""Nearly-incompressible isotropic hyperelastic Neo-Hookean material
    formulation. The strain energy density function of the Neo-Hookean
    material formulation is a linear function of the trace of the
    isochoric part of the right Cauchy-Green deformation tensor.

    Parameters
    ----------
    mu : float or None, optional
        Shear modulus
    bulk : float or None, optional
        Bulk modulus

    Notes
    -----

    ..  note::
        At least one of the two material parameters must not be None.

    In a nearly-incompressible constitutive framework the strain energy
    density is an additive composition of an isochoric and a volumetric
    part. While the isochoric part is defined on the distortional part of
    the deformation gradient, the volumetric part of the strain
    energy function is defined on the determinant of the deformation
    gradient.

    .. math::

        \psi &= \hat{\psi}(\hat{\boldsymbol{C}}) + U(J)

        \hat\psi(\hat{\boldsymbol{C}}) &= \frac{\mu}{2}
        (\text{tr}(\hat{\boldsymbol{C}}) - 3)

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

       \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial
       \boldsymbol{F}}

    A chain rule application leads to the following expression for the stress tensor.
    It is formulated as a sum of the **physical**-deviatoric (not the mathematical
    deviator!) and the physical-hydrostatic stress tensors.

    .. math::

       \boldsymbol{P} &= \boldsymbol{P}' + \boldsymbol{P}_U

       \boldsymbol{P}' &= \frac{\partial \hat{\psi}}{\partial \hat{\boldsymbol{F}}} :
       \frac{\partial \hat{\boldsymbol{F}}}{\partial \boldsymbol{F}} =
       \bar{\boldsymbol{P}} - \frac{1}{3} (\bar{\boldsymbol{P}} : \boldsymbol{F})
       \boldsymbol{F}^{-T}

       \boldsymbol{P}_U &= \frac{\partial U(J)}{\partial J} \frac{\partial J}{\partial
       \boldsymbol{F}} = U'(J) J \boldsymbol{F}^{-T}

    with

    .. math::

       \frac{\partial \hat{\boldsymbol{F}}}{\partial \boldsymbol{F}} &= J^{-1/3} \left(
       \boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I} - \frac{1}{3} \boldsymbol{F}
       \otimes \boldsymbol{F}^{-T} \right)

       \frac{\partial J}{\partial \boldsymbol{F}} &= J \boldsymbol{F}^{-T}

       \bar{\boldsymbol{P}} &= J^{-1/3} \frac{\partial \hat{\psi}}{\partial
       \hat{\boldsymbol{F}}}

    With the above partial derivatives the first Piola-Kirchhoff stress
    tensor of the Neo-Hookean material model takes the following form.

    .. math::

       \boldsymbol{P} = \mu J^{-2/3} \left( \boldsymbol{F} - \frac{1}{3} (
       \boldsymbol{F} : \boldsymbol{F}) \boldsymbol{F}^{-T} \right) + K (J - 1) J
       \boldsymbol{F}^{-T}

    Again, a chain rule application leads to an expression for the elasticity tensor.

    .. math::

       \mathbb{A} &= \mathbb{A}' + \mathbb{A}_{U}

       \mathbb{A}' &= \bar{\mathbb{A}} - \frac{1}{3} \left( (\bar{\mathbb{A}} :
       \boldsymbol{F}) \otimes \boldsymbol{F}^{-T} + \boldsymbol{F}^{-T} \otimes
       (\boldsymbol{F} : \bar{\mathbb{A}}) \right ) + \frac{1}{9} (\boldsymbol{F} :
       \bar{\mathbb{A}} : \boldsymbol{F}) \boldsymbol{F}^{-T} \otimes
       \boldsymbol{F}^{-T}

       \mathbb{A}_{U} &= (U''(J) J + U'(J)) J \boldsymbol{F}^{-T} \otimes
       \boldsymbol{F}^{-T} - U'(J) J \boldsymbol{F}^{-T} \overset{il}{\otimes}
       \boldsymbol{F}^{-T}

    with

    .. math::

       \bar{\mathbb{A}} = J^{-1/3} \frac{\partial^2 \hat\psi}{\partial
       \hat{\boldsymbol{F}}\ \partial \hat{\boldsymbol{F}}} J^{-1/3}

    With the above partial derivatives the (physical-deviatoric and
    -hydrostatic) parts of the elasticity tensor associated
    to the first Piola-Kirchhoff stress tensor of the Neo-Hookean
    material model takes the following form.

    .. math::

       \mathbb{A} &= \mathbb{A}' + \mathbb{A}_{U}

       \mathbb{A}' &= J^{-2/3} \left(\boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I}
       - \frac{1}{3} \left( \boldsymbol{F} \otimes \boldsymbol{F}^{-T} +
       \boldsymbol{F}^{-T} \otimes \boldsymbol{F} \right ) + \frac{1}{9} (\boldsymbol{F}
       : \boldsymbol{F}) \boldsymbol{F}^{-T} \otimes \boldsymbol{F}^{-T} \right)

       \mathbb{A}_{U} &= K J \left( (2J - 1) \boldsymbol{F}^{-T} \otimes
       \boldsymbol{F}^{-T} - (J - 1) \boldsymbol{F}^{-T} \overset{il}{\otimes}
       \boldsymbol{F}^{-T} \right)

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> ax = umat.plot()

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    """

    def __init__(self, mu=None, bulk=None, parallel=False):
        self.parallel = parallel

        self.mu = mu
        self.bulk = bulk

        self.kwargs = {}

        if self.mu is not None:
            self.kwargs["mu"] = self.mu

        if self.bulk is not None:
            self.kwargs["bulk"] = self.bulk

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

        Parameters
        ----------
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
        W = np.zeros_like(J)

        if mu is not None:
            W += mu / 2 * (J ** (-2 / 3) * trace(C) - 3)

        if bulk is not None:
            W += bulk * (J - 1) ** 2 / 2

        return [W]

    def gradient(self, x, mu=None, bulk=None, out=None):
        """Gradient of the strain energy density function per unit
        undeformed volume of the Neo-Hookean material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
        """

        F, statevars = x[0], x[-1]

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))

        P = out
        if P is None:
            P = np.zeros_like(F)

        if mu is not None:
            # "physical"-deviatoric (not math-deviatoric!) part of P
            trC = ddot(F, F, parallel=self.parallel)
            trC_3 = np.divide(trC, 3, out=trC)
            np.multiply(trC_3, iFT, out=P)
            np.add(F, -P, out=P)

            Jm23 = np.power(J, -2 / 3, out=trC)
            np.multiply(P, Jm23, out=P)
            np.multiply(P, mu, out=P)

        if bulk is not None:
            # "physical"-volumetric (not math-volumetric!) part of P
            JiFT = np.multiply(J, iFT, out=iFT)
            J_1 = np.add(J, -1, out=J)
            dUdJ = np.multiply(bulk, J_1, out=J_1)
            dUdF = np.multiply(dUdJ, JiFT, out=JiFT)
            np.add(P, dUdF, out=P)

        return [P, statevars]

    def hessian(self, x, mu=None, bulk=None, out=None):
        """Hessian of the strain energy density function per unit
        undeformed volume of the Neo-Hookean material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        bulk : float, optional
            Bulk modulus (default is None)
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
        """

        F = x[0]

        if mu is None:
            mu = self.mu

        if bulk is None:
            bulk = self.bulk

        J = det(F)
        iFT = transpose(inv(F, J))

        A4 = out
        if A4 is None:
            A4 = np.zeros((*F.shape[:2], *F.shape[:2], *F.shape[-2:]))
        else:
            np.multiply(A4, 0, out=A4)

        trC = None
        A4b = None
        A4c = None

        if mu is not None:
            # "physical"-deviatoric (not math-deviatoric!) part of A4
            eye = identity(F)
            trC = ddot(F, F, parallel=self.parallel, out=trC)
            np.add(A4, cdya_ik(eye, eye), out=A4)
            A4b = dya(F, iFT, parallel=self.parallel, out=A4b)
            np.multiply(-2 / 3, A4b, out=A4b)
            np.add(A4, A4b, out=A4)
            np.add(A4, np.transpose(A4b, [2, 3, 0, 1, 4, 5]), out=A4)
            A4b = dya(iFT, iFT, out=A4b)
            trC_3 = np.divide(trC, 3, out=trC)
            A4c = np.multiply(A4b, trC_3, out=A4c)
            np.add(A4, np.transpose(A4c, [0, 3, 2, 1, 4, 5]), out=A4)
            np.multiply(A4c, 2 / 3, out=A4c)
            np.add(A4, A4c, out=A4)
            np.multiply(mu, A4, out=A4)
            Jm23 = np.power(J, -2 / 3, out=trC)
            np.multiply(Jm23, A4, out=A4)

        if bulk is not None:
            # "physical"-volumetric (not math-volumetric!) part of A4
            if A4b is None:
                A4b = dya(iFT, iFT, out=A4b)

            J_1 = np.add(J, -1, out=trC)
            p = np.multiply(bulk, J_1, out=J_1)
            pJ = np.multiply(p, J, out=p)
            J2 = np.power(J, 2, out=J)
            bulk_J2 = np.multiply(J2, bulk, out=J2)
            qJ = np.add(pJ, bulk_J2, out=bulk_J2)
            A4c = np.multiply(qJ, A4b, out=A4c)
            np.add(A4, A4c, out=A4)
            np.multiply(-pJ, np.transpose(A4b, [0, 3, 2, 1, 4, 5]), out=A4c)
            np.add(A4, A4c, out=A4)

        return [A4]


class NeoHookeCompressible(ConstitutiveMaterial):
    r"""Compressible isotropic hyperelastic Neo-Hookean material formulation. The strain
    energy density function of the Neo-Hookean material formulation is a linear function
    of the trace of the right Cauchy-Green deformation tensor.

    Parameters
    ----------
    mu : float
        Shear modulus (second Lamé constant)
    lmbda : float
        First Lamé constant

    Notes
    -----

    .. math::

        \psi &= \psi(\boldsymbol{C})

        \psi(\boldsymbol{C}) &= \frac{\mu}{2} \text{tr}(\boldsymbol{C})
            - \mu \ln(J) + \frac{\lambda}{2} \ln(J)^2

    with

    .. math::

       J = \text{det}(\boldsymbol{F})

    The first Piola-Kirchhoff stress tensor is evaluated as the gradient
    of the strain energy density function.

    .. math::

       \boldsymbol{P} &= \frac{\partial \psi}{\partial \boldsymbol{F}}

       \boldsymbol{P} &= \mu \left( \boldsymbol{F} - \boldsymbol{F}^{-T} \right)
           + \lambda \ln(J) \boldsymbol{F}^{-T}

    The hessian of the strain energy density function enables the corresponding
    elasticity tensor.

    .. math::

       \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial
       \boldsymbol{F}}

       \mathbb{A} &= \mu \boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I}
           + \left(\mu - \lambda \ln(J) \right)
               \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T}
           + \lambda \boldsymbol{F}^{-T} {\otimes} \boldsymbol{F}^{-T}

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.NeoHookeCompressible(mu=1.0, lmbda=2.0)
        >>> ax = umat.plot()

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    """

    def __init__(self, mu=None, lmbda=None, parallel=False):
        self.parallel = parallel

        self.mu = mu
        self.lmbda = lmbda

        self.kwargs = {"mu": self.mu}
        if self.lmbda is not None:
            self.kwargs["lmbda"] = self.lmbda

        # aliases for function, gradient and hessian
        self.energy = self.function
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def function(self, x, mu=None, lmbda=None):
        """Strain energy density function per unit undeformed volume of the Neo-Hookean
        material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        lmbda : float, optional
            First Lamé constant (default is None)
        """

        F = x[0]

        if mu is None:
            mu = self.mu

        if lmbda is None:
            lmbda = self.lmbda

        lnJ = np.log(det(F))
        C = dot(transpose(F), F, parallel=self.parallel)

        W = mu * (trace(C) / 2 - lnJ)

        if lmbda is not None:
            W += lmbda * lnJ**2 / 2

        return [W]

    def gradient(self, x, mu=None, lmbda=None, out=None):
        """Gradient of the strain energy density function per unit undeformed volume of
        the Neo-Hookean material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        lmbda : float, optional
            First Lamé constant (default is None)
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
        """

        F, statevars = x[0], x[-1]

        if mu is None:
            mu = self.mu

        if lmbda is None:
            lmbda = self.lmbda

        J = det(F)
        iFT = transpose(inv(F, J))
        lnJ = np.log(J, out=J)

        P = np.multiply(mu, F, out=out)

        if lmbda is None:
            Pb = np.multiply(iFT, -mu, out=iFT)
        else:
            lmbda_lnJ = np.multiply(lmbda, lnJ, out=lnJ)
            Pb = np.multiply(iFT, -mu + lmbda_lnJ, out=iFT)

        np.add(P, Pb, out=P)

        return [P, statevars]

    def hessian(self, x, mu=None, lmbda=None, out=None):
        """Hessian of the strain energy density function per unit undeformed volume of
        the Neo-Hookean material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        mu : float, optional
            Shear modulus (default is None)
        lmbda : float, optional
            First Lamé constant (default is None)
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
        """

        F = x[0]

        if mu is None:
            mu = self.mu

        if lmbda is None:
            lmbda = self.lmbda

        J = det(F)
        iFT = transpose(inv(F, J))
        lnJ = np.log(J, out=J)
        eye = identity(F)

        iFTiFT = cdya_il(iFT, iFT, out=out)
        A4a = cdya_ik(eye, eye)
        np.multiply(mu, A4a, out=A4a)

        if lmbda is not None:
            lmbda_lnJ = np.multiply(lmbda, lnJ, out=lnJ)
            A4b = np.multiply(mu - lmbda_lnJ, iFTiFT, out=iFTiFT)
            iFT_iFT = dya(iFT, iFT)
            A4c = np.multiply(lmbda, iFT_iFT, out=iFT_iFT)
            np.add(A4a, A4b, out=A4b)
            A4 = np.add(A4b, A4c, out=A4c)
        else:
            A4b = np.multiply(mu, iFTiFT, out=iFTiFT)
            A4 = np.add(A4a, A4b, out=A4b)

        return [A4]


class Volumetric(NeoHooke):
    "Neo-Hookean material formulation with deactivated shear modulus."

    def __init__(self, bulk, parallel=False):
        super().__init__(mu=None, bulk=bulk, parallel=parallel)
