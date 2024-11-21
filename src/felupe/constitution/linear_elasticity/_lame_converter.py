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
from numpy.linalg import inv


def lame_converter(E, nu):
    r"""Convert the pair of given material parameters Young's modulus :math:`E` and
    Poisson ratio :math:`\nu` to first and second Lamé - constants :math:`\lambda` and
    :math:`\mu`.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    Returns
    -------
    lmbda : float
        First Lamé - constant.
    mu : float
        Second Lamé - constant (shear modulus).

    Notes
    -----

    ..  math::

        \lambda &= \frac{E \nu}{(1 + \nu) (1 - 2 \nu)}

        \mu &= \frac{E}{2 (1 + \nu)}
    """

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    return lmbda, mu


def lame_converter_orthotropic(E, nu, G):
    r"""Convert elastic orthotropic material parameters to Lamé constants.

    Parameters
    ----------
    E : list of float
        List of the three elastic moduli :math:`E_1, E_2, E_3`.
    nu : list of float
        List of three poisson ratios :math:`\nu_{12}, \nu_{23}, \nu_{31}`.
    G : list of float
        List of three shear moduli :math:`G_{12}, G_{23}, G_{31}`.

    Returns
    -------
    lmbda : list of float
        List of six (upper triangle) first Lamé parameters :math:`\lambda_{11},
        \lambda_{12}, \lambda_{13}, \lambda_{22}, \lambda_{23}, \lambda_{33}`.
    mu : list of float
        List of the three second Lamé parameters :math:`\mu_1,\mu_2, \mu_3`.

    Notes
    -----
    The orthotropic material parameters are converted to orthotropic Lamé
    constants.

    The compliance matrix as the inverse of the stiffness matrix with the
    parameters :math:`E_i`, :math:`\nu_{ij}` and :math:`G_{ij}` is given in
    Eq. :eq:`ortho-matrix-inv`.

    ..  math::
        :label: ortho-matrix-inv

        \boldsymbol{C}^{-1} = \begin{bmatrix}
            \frac{1}{E_1} & -\frac{\nu_{21}}{E_2} & -\frac{\nu_{31}}{E_3}
                & 0 & 0 & 0 \\
            -\frac{\nu_{12}}{E_1} & \frac{1}{E_2} & -\frac{\nu_{32}}{E_3}
                & 0 & 0 & 0 \\
            -\frac{\nu_{13}}{E_1} & -\frac{\nu_{23}}{E_2} & \frac{1}{E_3}
                & 0 & 0 & 0 \\
            0 & 0 & 0 & \frac{1}{G_{12}} & 0 & 0 \\
            0 & 0 & 0 & 0 & \frac{1}{G_{23}} & 0 \\  
            0 & 0 & 0 & 0 & 0 & \frac{1}{G_{31}}
        \end{bmatrix}

    The stiffness matrix with the Lamé constants is denoted in
    Eq. :eq:`ortho-matrix`.

    ..  math::
        :label: ortho-matrix-inv

        \boldsymbol{C} = \begin{bmatrix}
            \lambda_{11} + 2 \mu_1 & \lambda_{12} & \lambda_{13} & 0 & 0 & 0 \\
            \lambda_{11} & \lambda_{12} + 2 \mu_2 & \lambda_{13} & 0 & 0 & 0 \\
            \lambda_{11} & \lambda_{12} & \lambda_{13} + 2 \mu_3 & 0 & 0 & 0 \\
            0 & 0 & 0 & \frac{\mu_1 + \mu_2}{2} & 0 & 0 \\
            0 & 0 & 0 & 0 & \frac{\mu_2 + \mu_3}{2} & 0 \\  
            0 & 0 & 0 & 0 & 0 & \frac{\mu_3 + \mu_1}{2}
        \end{bmatrix}

    Eq. :eq:`ortho-matrix-inv` is evaluated and inverted numerically to extract
    the Lamé constants.

    See Also
    --------
    felupe.LinearElasticOrthotropic : Orthotropic linear-elastic
        material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.saint_venant_kirchhoff_orthotropic :
        Strain energy function of the orthotropic hyperelastic Saint-Venant
        Kirchhoff material formulation.
    """

    # unpack orthotropic elastic material parameters
    E1, E2, E3 = E
    ν12, ν23, ν31 = nu
    G12, G23, G31 = G

    # orthotropic symmetry
    ν21 = ν12 * E2 / E1
    ν32 = ν23 * E3 / E2
    ν13 = ν31 * E1 / E3

    C = inv(
        [
            [1 / E1, -ν21 / E2, -ν31 / E3, 0, 0, 0],
            [-ν12 / E1, 1 / E2, -ν32 / E3, 0, 0, 0],
            [-ν13 / E1, -ν23 / E2, 1 / E3, 0, 0, 0],
            [0, 0, 0, 1 / G12, 0, 0],
            [0, 0, 0, 0, 1 / G23, 0],
            [0, 0, 0, 0, 0, 1 / G31],
        ]
    )

    # take the components from this matrix
    #
    # [λ11 + 2 * μ1, λ12, λ13, 0, 0, 0]
    # [λ12, λ22 + 2 * μ2, λ23, 0, 0, 0]
    # [λ13, λ23, λ33 + 2 * μ3, 0, 0, 0]
    # [0, 0, 0, (μ1 + μ2) / 2, 0, 0]
    # [0, 0, 0, 0, (μ2 + μ3) / 2, 0]
    # [0, 0, 0, 0, 0, (μ1 + μ3) / 2]

    λ12 = C[0, 1]
    λ23 = C[1, 2]
    λ13 = C[0, 2]

    μ1 = C[3, 3] - C[4, 4] + C[5, 5]
    μ2 = C[3, 3] + C[4, 4] - C[5, 5]
    μ3 = -C[3, 3] + C[4, 4] + C[5, 5]

    λ11 = C[0, 0] - 2 * μ1
    λ22 = C[1, 1] - 2 * μ2
    λ33 = C[2, 2] - 2 * μ3

    lmbda = [λ11, λ12, λ13, λ22, λ23, λ33]
    mu = [μ1, μ2, μ3]

    return lmbda, mu
