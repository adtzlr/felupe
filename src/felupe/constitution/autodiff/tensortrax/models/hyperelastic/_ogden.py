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

from tensortrax.math import sum as tsum
from tensortrax.math.linalg import det, eigvalsh


def ogden(C, mu, alpha):
    r"""Strain energy function of the isotropic hyperelastic
    `Ogden <https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model)>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : list of float
        List of moduli.
    alpha : list of float
        List of stretch exponents.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-ogden`

    ..  math::
        :label: psi-ogden

        \psi = \sum_i \frac{2 \mu_i}{\alpha^2_i} \left(
            \lambda_1^{\alpha_i} + \lambda_2^{\alpha_i} + \lambda_3^{\alpha_i} - 3
        \right)

    The sum of the moduli :math:`\mu_i` is equal to the initial shear modulus
    :math:`\mu`, see Eq. :eq:`shear-modulus-ogden`.

    ..  math::
        :label: shear-modulus-ogden

        \mu = \sum_i \mu_i

    Examples
    --------

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.ogden, mu=[1, 0.2], alpha=[1.7, -1.5])
        >>> ax = umat.plot(incompressible=True)

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

    wC = det(C) ** (-1 / 3) * eigvalsh(C)
    return tsum([2 * m / a**2 * (tsum(wC ** (a / 2)) - 3) for m, a in zip(mu, alpha)])
