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


def saint_venant_kirchhoff(C, mu, lmbda):
    r"""Strain energy function of the isotropic hyperelastic
    `Saint-Venant Kirchhoff <https://en.wikipedia.org/wiki/Hyperelastic_material#Saint_Venant-Kirchhoff_model>`_
    material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : float
        Second Lamé constant (shear modulus).
    lmbda : float
        First Lamé constant (shear modulus).

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-svk`

    ..  math::
        :label: psi-svk

        \psi = \mu I_2 + \lambda \frac{I_1^2}{2}

    with the first and second invariant of the Green-Lagrange strain tensor
    :math:`\boldsymbol{E} = \frac{1}{2} (\boldsymbol{C} - \boldsymbol{1})`, see Eq.
    :eq:`invariants-svk`.

    ..  math::
        :label: invariants-svk

        I_1 &= \text{tr}\left( \boldsymbol{E} \right)

        I_2 &= \boldsymbol{E} : \boldsymbol{E}

    Examples
    --------
    ..  warning::
        The Saint-Venant Kirchhoff material formulation is unstable for large strains.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.saint_venant_kirchhoff, mu=1.0, lmbda=20.0)
        >>> ax = umat.plot(incompressible=False)

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
    I1 = trace(C) / 2 - 3 / 2
    I2 = trace(C @ C) / 4 - trace(C) / 2 + 3 / 4
    return mu * I2 + lmbda * I1**2 / 2
