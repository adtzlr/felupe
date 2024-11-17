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

from tensortrax.math import sqrt, trace
from tensortrax.math.linalg import det


def blatz_ko(C, mu):
    r"""Strain energy function of the Blatz-Ko isotropic hyperelastic
    `foam <https://doi.org/10.1122/1.548937>`_ material formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : float
        The shear modulus.

    Notes
    -----
    The Poisson ratio of the Blatz-Ko model formulation is :math:`\nu = 0.25`. The
    strain energy function is given in Eq. :eq:`psi-blatz-ko`

    ..  math::
        :label: psi-blatz-ko

        \psi = \frac{\mu}{2} \left(\frac{I_2}{I_3} + 2 \sqrt{I_3} - 5 \right)

    The shear modulus :math:`\mu` is related to young's modulus as denoted in Eq.
    :eq:`shear-modulus-blatz-ko`.

    ..  math::
        :label: shear-modulus-blatz-ko

        \mu = \frac{2 E}{5}

    Examples
    --------
    First, choose the desired automatic differentiation backend

    ..  pyvista-plot::
        :context:

        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    and create the hyperelastic material.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = mat.Hyperelastic(mat.models.hyperelastic.blatz_ko, mu=1.0)
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

    References
    ----------
    .. [1] P. J. Blatz and W. L. Ko, "Application of Finite Elastic Theory to the
       Deformation of Rubbery Materials", Transactions of the Society of Rheology, vol.
       6, no. 1. Society of Rheology, pp. 223–252, Mar. 01, 1962. doi: 10.1122/1.548937.
    """

    I1 = trace(C)
    I2 = (I1**2 - trace(C @ C)) / 2
    I3 = det(C)

    return mu * (I2 / I3 + 2 * sqrt(I3) - 5)
