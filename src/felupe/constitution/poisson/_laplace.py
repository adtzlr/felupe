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

from ...math import cdya_ik, ddot, identity
from .._base import ConstitutiveMaterial


class Laplace(ConstitutiveMaterial):
    r"""Laplace equation as hessian of one half of the second main invariant of the
    field gradient.

    Parameters
    ----------
    multiplier : float, optional
        A multiplier which scales the potential (default is 1.0).

    Notes
    -----
    The potential is given by the second main invariant of the field gradient w.r.t.
    the undeformed coordinates.

    ..  math::

        \psi = \frac{1}{2} \left( \boldsymbol{H} : \boldsymbol{H} \right)

    with the field gradient w.r.t. the undeformed coordinates.

    ..  math::

        \boldsymbol{H} = \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}}

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Laplace()
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

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.kwargs = {"multiplier": self.multiplier}

        # aliases for gradient and hessian
        self.stress = self.gradient
        self.elasticity = self.hessian

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(0)]

    def function(self, x):
        r"""Evaluate the potential per unit undeformed volume.

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`\boldsymbol{F}` as first item.

        Returns
        -------
        ndarray of shape (...)
            potential

        """

        F = x[0]
        H = F - identity(F)

        return [self.multiplier * ddot(H, H) / 2]

    def gradient(self, x):
        r"""Evaluate the stress tensor.

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`\boldsymbol{F}` as first item.

        Returns
        -------
        ndarray of shape (n, m, ...)
            gradient of the potential w.r.t. the undeformed coordinates

        """

        F, statevars = x[0], x[-1]
        H = F - identity(F)

        return [self.multiplier * H, statevars]

    def hessian(self, x):
        r"""Evaluate the elasticity tensor.

        Parameters
        ----------
        x : list of ndarray
            List with Deformation gradient :math:`\boldsymbol{F}` as first item.

        Returns
        -------
        ndarray of shape (n, m, n, m, ...)
            hessian of the potential w.r.t. the undeformed coordinates

        """

        n, m = x[0].shape[:2]
        ntrax = len(x[0].shape) - 2
        ones = np.ones(ntrax, dtype=int)

        return [
            self.multiplier * cdya_ik(np.eye(n), np.eye(m)).reshape(n, m, n, m, *ones)
        ]
