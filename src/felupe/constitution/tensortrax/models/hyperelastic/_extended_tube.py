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

from tensortrax.math import log
from tensortrax.math import sum as tsum
from tensortrax.math import trace
from tensortrax.math.linalg import det, eigvalsh


def extended_tube(C, Gc, delta, Ge, beta):
    r"""Strain energy function of the isotropic hyperelastic
    `Extended Tube <https://www.doi.org/10.5254/1.3538822>`_ [1]_ material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    Gc : float
        Cross-link contribution to the initial shear modulus.
    delta : float
         Finite extension parameter of the polymer strands.
    Ge : float
        Constraint contribution to the initial shear modulus.
    beta : float
        Global rearrangements of cross-links upon deformation (release of topological
        constraints).

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-et`

    ..  math::
        :label: psi-et

        \psi = \frac{G_c}{2} \left[ \frac{\left( 1 - \delta^2 \right)
            \left( \hat{I}_1 - 3 \right)}{1 - \delta^2 \left( \hat{I}_1 - 3 \right)} +
            \ln \left( 1 - \delta^2 \left( \hat{I}_1 - 3 \right) \right) \right] +
            \frac{2 G_e}{\beta^2} \left( \hat{\lambda}_1^{-\beta} +
            \hat{\lambda}_2^{-\beta} + \hat{\lambda}_3^{-\beta} - 3 \right)

    with the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor as given in Eq. :eq:`invariants-et`

    ..  math::
        :label: invariants-et

        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

    and the principal stretches, obtained from the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`stretches-et`.

    ..  math::
        :label: stretches-et

        \lambda^2_\alpha &= \text{eigvals}\left( \boldsymbol{C} \right)

        \hat{\lambda}_\alpha &= J^{-1/3} \lambda_\alpha

    The initial shear modulus results from the sum of the cross-link and the constraint
    contributions to the total initial shear modulus as denoted in Eq.
    :eq:`shear-modulus-et`.

    ..  math::
        :label: shear-modulus-et

        \mu = G_e + G_c

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
        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.extended_tube,
        ...     Gc=0.1867,
        ...     Ge=0.2169,
        ...     beta=0.2,
        ...     delta=0.09693,
        ... )
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

    References
    ----------
    ..  [1] M. Kaliske and G. Heinrich, "An Extended Tube-Model for Rubber Elasticity:
        Statistical-Mechanical Theory and Finite Element Implementation", Rubber
        Chemistry and Technology, vol. 72, no. 4. Rubber Division, ACS, pp. 602–632,
        Sep. 01, 1999. doi:
        `10.5254/1.3538822 <https://www.doi.org/10.5254/1.3538822>`_.

    """
    J3 = det(C) ** (-1 / 3)
    D = J3 * trace(C)
    wC = J3 * eigvalsh(C)
    g = (1 - delta**2) * (D - 3) / (1 - delta**2 * (D - 3))
    Wc = Gc / 2 * (g + log(1 - delta**2 * (D - 3)))
    We = 2 * Ge / beta**2 * tsum(wC ** (-beta / 2) - 1)
    return Wc + We
