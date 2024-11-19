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

from ...math import ddot, identity, transpose
from .._base import ConstitutiveMaterial


class LinearElasticOrthotropic(ConstitutiveMaterial):
    r"""Orthotropic linear-elastic material formulation.

    Parameters
    ----------
    E : float
        Young's modulus (E1, E2, E3).
    nu : float
        Poisson ratio (nu12, nu23, n31).
    G : float
        Shear modulus (G12, G23, G31).

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.LinearElasticOrthotropic(
        ...     E=[1, 1, 1], nu=[0.3, 0.3, 0.3], G=[0.4, 0.4, 0.4]
        ... )
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

    def __init__(self, E, nu, G):
        self.E = E
        self.nu = nu
        self.G = G

        self.kwargs = {"E": self.E, "nu": self.nu, "G": self.G}

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def gradient(self, x):
        r"""Evaluate the stress tensor (as a function of the deformation gradient).

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`\boldsymbol{F}` (3x3) as first item.

        Returns
        -------
        ndarray of shape (3, 3, ...)
            Stress tensor

        """

        F, statevars = x[0], x[-1]

        # convert the deformation gradient to strain
        H = F - identity(F)
        strain = (H + transpose(H)) / 2

        # init stress
        elast = self.hessian(x=x)[0]

        return [ddot(elast, strain, mode=(4, 2)), statevars]

    def hessian(self, x=None, shape=(1, 1), dtype=None):
        r"""Evaluate the elasticity tensor. The Deformation gradient is only
        used for the shape of the trailing axes.

        Parameters
        ----------
        x : list of ndarray, optional
            List with Deformation gradient :math:`\boldsymbol{F}` (3x3) as first item
            (default is None).
        shape : tuple of int, optional
            Tuple with shape of the trailing axes (default is (1, 1)).

        Returns
        -------
        ndarray of shape (3, 3, 3, 3, ...)
            elasticity tensor

        """

        E1, E2, E3 = self.E
        nu12, nu23, nu31 = self.nu
        G12, G23, G31 = self.G

        if x is not None:
            dtype = x[0].dtype

        elast = np.zeros((3, 3, 3, 3, *shape), dtype=dtype)

        nu21 = nu12 * E2 / E1
        nu32 = nu23 * E3 / E2
        nu13 = nu31 * E1 / E3

        J = 1 / (1 - nu12 * nu21 - nu23 * nu32 - nu31 * nu13 - 2 * nu21 * nu32 * nu13)

        elast[0, 0, 0, 0] = E1 * (1 - nu23 * nu32) * J
        elast[1, 1, 1, 1] = E2 * (1 - nu13 * nu31) * J
        elast[2, 2, 2, 2] = E3 * (1 - nu12 * nu21) * J

        elast[0, 0, 1, 1] = elast[1, 1, 0, 0] = E1 * (nu21 + nu31 * nu23) * J
        elast[0, 0, 2, 2] = elast[2, 2, 0, 0] = E1 * (nu31 + nu21 * nu32) * J
        elast[1, 1, 2, 2] = elast[2, 2, 1, 1] = E2 * (nu32 + nu12 * nu31) * J

        elast[
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ] = G12

        elast[
            [0, 2, 0, 2],
            [2, 0, 2, 0],
            [0, 0, 2, 2],
            [2, 2, 0, 0],
        ] = G31

        elast[
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 1, 2, 2],
            [2, 2, 1, 1],
        ] = G23

        return [elast]
