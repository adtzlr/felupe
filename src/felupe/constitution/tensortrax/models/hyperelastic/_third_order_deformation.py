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

from tensortrax.math import trace
from tensortrax.math.linalg import det


def third_order_deformation(C, C10, C01, C11, C20, C30):
    r"""Strain energy function of the isotropic hyperelastic
    `Third-Order-Deformation <https://doi.org/10.1002/app.1975.070190723>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    C10 : float
        Material parameter associated to the linear term of the first invariant.
    C01 : float
        Material parameter associated to the linear term of the second invariant.
    C11 : float
        Material parameter associated to the mixed term of the first and second
        invariant.
    C20 : float
        Material parameter associated to the quadratic term of the first invariant.
    C30 : float
        Material parameter associated to the cubic term of the first invariant.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-tod`

    ..  math::
        :label: psi-tod

        \psi &= C_{10} \left(\hat{I}_1 - 3 \right) + C_{01} \left(\hat{I}_2 - 3 \right)
             + C_{11} \left(\hat{I}_1 - 3 \right) \left(\hat{I}_2 - 3 \right)

            &+ C_{20} \left(\hat{I}_1 - 3 \right)^2
             + C_{30} \left(\hat{I}_1 - 3 \right)^3

    with the first and second main invariant of the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`invariants-tod`.

    ..  math::
        :label: invariants-tod

        \hat{I}_1 &= J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

        \hat{I}_2 &= J^{-4/3} \frac{1}{2} \left(
            \text{tr}\left(\boldsymbol{C}\right)^2 -
            \text{tr}\left(\boldsymbol{C}^2\right)
        \right)

    The doubled sum of the material parameters :math:`C_{10}` and :math:`C_{01}` is
    equal to the initial shear modulus :math:`\mu` as denoted in Eq.
    :eq:`shear-modulus-tod`.

    ..  math::
        :label: shear-modulus-tod

        \mu = 2 \left( C_{10} + C_{01} \right)

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

        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.third_order_deformation,
        ...     C10=0.5,
        ...     C01=0.1,
        ...     C11=0.01,
        ...     C20=-0.1,
        ...     C30=0.02,
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

    """
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (I1**2 - J3**2 * trace(C @ C)) / 2
    return (
        C10 * (I1 - 3)
        + C01 * (I2 - 3)
        + C11 * (I1 - 3) * (I2 - 3)
        + C20 * (I1 - 3) ** 2
        + C30 * (I1 - 3) ** 3
    )
