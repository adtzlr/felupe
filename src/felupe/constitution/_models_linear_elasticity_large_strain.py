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

from ._base import ConstitutiveMaterial
from ._models_hyperelasticity import NeoHookeCompressible
from ._models_linear_elasticity import lame_converter


class LinearElasticLargeStrain(ConstitutiveMaterial):
    r"""Linear-elastic material formulation suitable for large-rotation analyses based
    on the compressible Neo-Hookean material formulation.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    See Also
    --------
    NeoHookeCompressible: Compressible isotropic hyperelastic Neo-Hooke material
        formulation.

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.LinearElasticLargeStrain(E=1.0, nu=0.3)
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

    def __init__(self, E=None, nu=None, parallel=False):
        self.E = E
        self.nu = nu

        self.kwargs = {"E": self.E, "nu": self.nu}

        # aliases for gradient and hessian
        self.energy = self.function
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

        mu = None
        lmbda = None

        if self.E is not None and self.nu is not None:
            lmbda, mu = lame_converter(E, nu)

        self.material = NeoHookeCompressible(mu=mu, lmbda=lmbda, parallel=parallel)

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

        lmbda, mu = lame_converter(E, nu)

        return self.material.function(x, mu=mu, lmbda=lmbda)

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

        lmbda, mu = lame_converter(E, nu)

        return self.material.gradient(x, mu=mu, lmbda=lmbda)

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

        lmbda, mu = lame_converter(E, nu)

        return self.material.hessian(x, mu=mu, lmbda=lmbda)
