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

from ._models_hyperelasticity import NeoHooke
from ._models_linear_elasticity import lame_converter


class LinearElasticLargeStrain:
    r"""Linear-elastic material formulation suitable for large-rotation analyses based
    on the nearly-incompressible Neo-Hookean material formulation.

    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    See Also
    --------
    NeoHooke: Nearly-incompressible isotropic hyperelastic Neo-Hooke material
        formulation.

    """

    def __init__(self, E=None, nu=None, parallel=False):
        self.E = E
        self.nu = nu

        # aliases for gradient and hessian
        self.energy = self.function
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

        mu = None
        bulk = None

        if self.E is not None and self.nu is not None:
            gamma, mu = lame_converter(E, nu)
            bulk = gamma + 2 * mu / 3

        self.material = NeoHooke(mu=mu, bulk=bulk, parallel=parallel)

    def function(self, x, E=None, nu=None):
        """Evaluate the strain energy (as a function of the deformation gradient).

        Arguments
        ---------
        x : list of ndarray
            List with Deformation gradient ``F`` (3x3) as first item
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        gamma, mu = lame_converter(E, nu)
        bulk = gamma + 2 * mu / 3

        return self.material.function(x, mu=mu, bulk=bulk)

    def gradient(self, x, E=None, nu=None):
        """Evaluate the stress tensor (as a function of the deformation gradient).

        Arguments
        ---------
        x : list of ndarray
            List with Deformation gradient ``F`` (3x3) as first item
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            Stress tensor (3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        gamma, mu = lame_converter(E, nu)
        bulk = gamma + 2 * mu / 3

        return self.material.gradient(x, mu=mu, bulk=bulk)

    def hessian(self, x, E=None, nu=None):
        """Evaluate the elasticity tensor (as a function of the deformation gradient).

        Arguments
        ---------
        x : list of ndarray
            List with Deformation gradient ``F`` (3x3) as first item.
        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            elasticity tensor (3x3x3x3)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        gamma, mu = lame_converter(E, nu)
        bulk = gamma + 2 * mu / 3

        return self.material.hessian(x, mu=mu, bulk=bulk)
