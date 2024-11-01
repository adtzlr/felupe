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

from ...math import cdya_ik, cdya_il, det, dot, dya, identity, inv, trace, transpose
from .._base import ConstitutiveMaterial


class NeoHookeCompressible(ConstitutiveMaterial):
    r"""Compressible isotropic hyperelastic Neo-Hookean material formulation. The strain
    energy density function of the Neo-Hookean material formulation is a linear function
    of the trace of the right Cauchy-Green deformation tensor.

    Parameters
    ----------
    mu : float or None, optional
        Shear modulus (second Lamé constant). Default is None.
    lmbda : float or None, optional
        First Lamé constant (default is None)

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

    def function(self, x):
        """Strain energy density function per unit undeformed volume of the Neo-Hookean
        material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        """

        F = x[0]

        mu = self.mu
        lmbda = self.lmbda

        lnJ = np.log(det(F))
        C = dot(transpose(F), F, parallel=self.parallel)

        W = mu * (trace(C) / 2 - lnJ)

        if lmbda is not None:
            W += lmbda * lnJ**2 / 2

        return [W]

    def gradient(self, x, out=None):
        """Gradient of the strain energy density function per unit undeformed volume of
        the Neo-Hookean material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
        """

        F, statevars = x[0], x[-1]

        mu = self.mu
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

    def hessian(self, x, out=None):
        """Hessian of the strain energy density function per unit undeformed volume of
        the Neo-Hookean material formulation.

        Parameters
        ----------
        x : list of ndarray
            List with the Deformation gradient ``F`` (3x3) as first item
        out : ndarray or None, optional
            A location into which the result is stored (default is None).
        """

        F = x[0]

        mu = self.mu
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
