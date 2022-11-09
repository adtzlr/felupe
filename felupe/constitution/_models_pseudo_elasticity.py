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

from ..math import dya


class OgdenRoxburgh:
    r"""A Pseudo-Elastic material formulation for an isotropic treatment of the
    load-history dependent Mullins-softening of rubber-like materials.

    ..  math::

        \eta(W, W_{max}) &= 1 - \frac{1}{r} erf\left( \frac{W_{max} - W}{m + \beta~W_{max}} \right)

        \boldsymbol{P} &= \eta \frac{\partial \psi}{\partial \boldsymbol{F}}

        \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F} \partial \boldsymbol{F}} + \frac{\partial \eta}{\partial \psi} \frac{\partial \psi}{\partial \boldsymbol{F}} \otimes \frac{\partial \psi}{\partial \boldsymbol{F}}

    """

    def __init__(self, material, r, m, beta):

        # isotropic hyperelastic material formulation
        self.material = material

        # ogden-roxburgh material parameters
        self.r = r
        self.m = m
        self.beta = beta

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
