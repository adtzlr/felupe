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


def storakers(C, mu, alpha, beta):
    r"""Strain energy function of Storåkers' isotropic hyperelastic
    `Foam <https://doi.org/10.1016/0022-5096(86)90033-5>`_ material formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : list of float
        List of moduli.
    alpha : list of float
        List of stretch exponents.
    beta : list of float
        List of coefficients for the degree of compressibility.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-foam`

    ..  math::
        :label: psi-ogden

        \psi = \sum_i \frac{2 \mu_i}{\alpha^2_i} \left[
            \hat{\lambda}_1^{\alpha_i} +
            \hat{\lambda}_2^{\alpha_i} +
            \hat{\lambda}_3^{\alpha_i} - 3
            + \frac{1}{\beta_i} \left( J^{-\alpha \beta} - 1 \right)
        \right]

    The sum of the moduli :math:`\mu_i` is equal to the initial shear modulus
    :math:`\mu`, see Eq. :eq:`shear-modulus-foam`,

    ..  math::
        :label: shear-modulus-ogden

        \mu = \sum_i \mu_i

    and the initial bulk modulus is given in Eq. :eq:`bulk-modulus-foam`.

    ..  math::
        :label: bulk-modulus-ogden

        K = \sum_i 2 \mu_i \left( \frac{1}{3} + \beta_i \right)

    Examples
    --------
    First, import the automatic differentiation backend

    ..  pyvista-plot::
        :context:

        >>> import felupe.constitution.tensortrax as mat

    and create the hyperelastic material [2]_.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.storakers,
        ...     mu=[104.869e-3],
        ...     alpha=[7.10874],
        ...     beta=[0.106469],
        ... )
        >>> ax = umat.plot(
        ...     ux=fem.math.linsteps([1, 0.25], 15),
        ...     ps=fem.math.linsteps([1, 0.25], 15),
        ...     bx=fem.math.linsteps([1, 0.55], 9),
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

    References
    ----------
    .. [1] B. Storåkers, "On material representation and constitutive branching in
       finite compressible elasticity", Journal of the Mechanics and Physics of Solids,
       vol. 34, no. 2. Elsevier BV, pp. 125–145, Jan. 1986. doi:
       10.1016/0022-5096(86)90033-5.

    .. [2] A. Kossa and S. Berezvai, "Novel strategy for the hyperelastic parameter
       fitting procedure of polymer foam materials", Polymer Testing, vol. 53. Elsevier
       BV, pp. 149–155, Aug. 2016. doi: 10.1016/j.polymertesting.2016.05.014.
    """

    λ2 = eigvalsh(C)
    
    return tsum(
        [
            2 * μ / α**2 * (tsum(λ2 ** (α / 2)) - 3 + (det(C) ** (-α * β / 2) - 1) / β)
            for μ, α, β in zip(mu, alpha, beta)
        ]
    )
