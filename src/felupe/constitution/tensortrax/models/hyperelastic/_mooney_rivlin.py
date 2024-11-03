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


def mooney_rivlin(C, C10, C01):
    r"""Strain energy function of the isotropic hyperelastic
    `Mooney-Rivlin <https://en.wikipedia.org/wiki/Mooney-Rivlin_solid>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    C10 : float
        First material parameter associated to the first invariant.
    C01 : float
        Second material parameter associated to the second invariant.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-mr`

    ..  math::
        :label: psi-mr

        \psi = C_{10} \left(\hat{I}_1 - 3 \right) + C_{01} \left(\hat{I}_2 - 3 \right)

    with the first and second main invariant of the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`invariants-mr`.

    ..  math::
        :label: invariants-mr

        \hat{I}_1 &= J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

        \hat{I}_2 &= J^{-4/3} \frac{1}{2} \left(
            \text{tr}\left(\boldsymbol{C}\right)^2 -
            \text{tr}\left(\boldsymbol{C}^2\right)
        \right)

    The doubled sum of both material parameters is equal to the shear modulus
    :math:`\mu` as denoted in Eq. :eq:`shear-modulus-mr`.

    ..  math::
        :label: shear-modulus-mr

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
        ...     mat.models.hyperelastic.mooney_rivlin, C10=0.3, C01=0.8
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
    return C10 * (I1 - 3) + C01 * (I2 - 3)
