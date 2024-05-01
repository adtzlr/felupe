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

from ..math import dya
from ._base import ConstitutiveMaterial


class OgdenRoxburgh(ConstitutiveMaterial):
    r"""`Ogden-Roxburgh <https://doi.org/10.1098%2Frspa.1999.0431>`_ Pseudo-Elastic
    material formulation for an isotropic treatment of the load-history dependent
    Mullins-softening of rubber-like materials.

    Parameters
    ----------
    material : NeoHooke, Hyperelastic, Material or MaterialAD
        An isotropic hyperelastic (user) material definition.
    r : float
        Reciprocal value of the maximum relative amount of softening. i.e. ``r=3`` means
        the shear modulus of the base material scales down from :math:`1` (no softening)
        to :math:`1 - 1/3 = 2/3` (maximum softening).
    m : float
        The initial Mullins softening modulus.
    beta : float
        Maximum deformation-dependent part of the Mullins softening modulus.

    Notes
    -----
    ..  note::
        This implementation uses the hyperbolic tangent instead of the Gauss error
        function.

    ..  math::

        \eta(\psi, \psi_\text{max}) &= 1 - \frac{1}{r} \tanh \left(
            \frac{\psi_\text{max} - \psi}{m + \beta~\psi_\text{max}}
        \right)

        \boldsymbol{P} &= \eta \frac{\partial \psi}{\partial \boldsymbol{F}}

        \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F} \partial
        \boldsymbol{F}} + \frac{\partial \eta}{\partial \psi} \frac{\partial \psi}
        {\partial \boldsymbol{F}} \otimes \frac{\partial \psi}{\partial \boldsymbol{F}}

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> neo_hooke = fem.NeoHooke(mu=1.0)
        >>> umat = fem.OgdenRoxburgh(material=neo_hooke, r=3.0, m=1.0, beta=0.0)
        >>>
        >>> ax = umat.plot(
        ...     ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
        ...     ps=None,
        ...     bx=None,
        ...     incompressible=True,
        ... )

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

    def __init__(self, material, r, m, beta):
        # isotropic hyperelastic material formulation
        self.material = self.fun = material

        # ogden-roxburgh material parameters
        self.r = r
        self.m = m
        self.beta = beta

        self.kwargs = {
            "r": self.r,
            "m": self.m,
            "beta": self.beta,
            **self.material.kwargs,
        }

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(1)]

    def gradient(self, x):
        # unpack variables into deformation gradient and state variables
        F, statevars = x[0], x[-1]

        # material parameters alias
        r, m, beta = self.r, self.m, self.beta

        # isotropic material formulation: evaluate
        # * the strain energy function and
        # * the first Piola-Kirchhoff stress tensor
        W = self.material.function([F, statevars])[0]
        P = self.material.gradient([F, statevars])[0]

        # get the maximum load-history strain energy function
        Wmax = np.maximum(W, statevars[0])
        z = (Wmax - W) / (m + beta * Wmax)

        # softening function
        eta = 1 - np.tanh(z) / r

        # update the state variables
        statevars_new = statevars.copy()
        statevars_new[0] = Wmax

        return [eta * P, statevars_new]

    def hessian(self, x):
        # unpack variables into deformation gradient and state variables
        F, statevars = x[0], x[-1]

        # material parameters alias
        r, m, beta = self.r, self.m, self.beta

        # isotropic material formulation: evaluate
        # * the strain energy function and
        # * the first Piola-Kirchhoff stress tensor as well as
        # * the according fourth-order elasticity tensor
        W = self.material.function([F, statevars])[0]
        P = self.material.gradient([F, statevars])[0]
        A = self.material.hessian([F, statevars])[0]

        # get the maximum load-history strain energy function
        Wmax = np.maximum(W, statevars[0])
        z = (Wmax - W) / (m + beta * Wmax)

        # softening function
        eta = 1 - np.tanh(z) / r

        # derivative of softening function
        detadz = (-1 / r) * 1 / np.cosh(z) ** 2
        dzdW = -1 / (m + beta * Wmax)
        detadW = detadz * dzdW

        # set non-softened derivative to zero
        detadW[np.isclose(eta, 1)] = 0

        return [eta * A + detadW * dya(P, P)]
