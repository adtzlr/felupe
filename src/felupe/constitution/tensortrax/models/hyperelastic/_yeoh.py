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


def yeoh(C, C10, C20, C30):
    r"""Strain energy function of the isotropic hyperelastic
    `Yeoh <https://en.wikipedia.org/wiki/Yeoh_(hyperelastic_model)>`_
    material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    C10 : float
        Material parameter associated to the linear term of the first invariant.
    C20 : float
        Material parameter associated to the quadratic term of the first invariant.
    C30 : float
        Material parameter associated to the cubic term of the first invariant.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-yeoh`

    ..  math::
        :label: psi-yeoh

        \psi = C_{10} \left(\hat{I}_1 - 3 \right) + C_{20} \left(\hat{I}_1 - 3 \right)^2
             + C_{30} \left(\hat{I}_1 - 3 \right)^3

    with the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`invariants-yeoh`.

    ..  math::
        :label: invariants-yeoh

        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

    The :math:`C_{10}` material parameter is equal to half the initial shear modulus
    :math:`\mu` as denoted in Eq. :eq:`shear-modulus-yeoh`.

    ..  math::
        :label: shear-modulus-yeoh

        \mu = 2 C_{10}

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
        ...     mat.models.hyperelastic.yeoh, C10=0.5, C20=-0.1, C30=0.02
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

    I1 = det(C) ** (-1 / 3) * trace(C)
    return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3
