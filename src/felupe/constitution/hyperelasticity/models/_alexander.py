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
from tensortrax.math import log, trace
from tensortrax.math.linalg import det


def alexander(C, C1, C2, C3, gamma):
    r"""Strain energy function of the isotropic hyperelastic
    `Alexander <https://doi.org/10.1016/0020-7225(68)90006-2>`_ material
    formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    C1 : float
        First material parameter associated to the first invariant.
    C2 : float
        Second material parameter associated to the second invariant.
    C3 : float
        Third material parameter associated to the second invariant.
    gamma : float
        Dimensionless material parameter associated to the second invariant.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-alexander`

    ..  math::
        :label: psi-alexander

        \psi = C_1 \left(\hat{I}_1 - 3 \right)
             + C_2 \ln{\frac{\hat{I}_2 - 3 + \gamma}{\gamma}}
             + C_3 \left(\hat{I}_2 - 3 \right)

    with the first and second main invariant of the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`invariants-alexander`.

    ..  math::
        :label: invariants-alexander

        \hat{I}_1 &= J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

        \hat{I}_2 &= J^{-4/3} \frac{1}{2} \left(
            \text{tr}\left(\boldsymbol{C}\right)^2 -
            \text{tr}\left(\boldsymbol{C}^2\right)
        \right)

    The initial shear modulus :math:`\mu` is given in Eq.
    :eq:`shear-modulus-alexander`.

    ..  math::
        :label: shear-modulus-alexander

        \mu = 2 \left( C_1 + \frac{C_2}{\gamma} + C_3 \right)

    Examples
    --------

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...    fem.alexander, C1=0.117, C2=0.137, C3=0.00690, gamma=0.735
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
    .. [1] H. Alexander, "A constitutive relation for rubber-like materials",
       International Journal of Engineering Science, vol. 6, no. 9. Elsevier BV, pp.
       549–563, Sep. 1968. doi:
       `10.1016/0020-7225(68)90006-2 <https://doi.org/10.1016/0020-7225(68)90006-2>`_.
    """
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (I1**2 - J3**2 * trace(C @ C)) / 2

    return C1 * (I1 - 3) + C2 * log(((I2 - 3) + gamma) / gamma) + C3 * (I2 - 3)
