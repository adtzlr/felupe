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

from functools import wraps

import numpy as np
from tensortrax.math import log, sqrt
from tensortrax.math import sum as sum1
from tensortrax.math import trace
from tensortrax.math._linalg import det, eigvalsh, inv
from tensortrax.math._special import from_triu_1d, triu_1d


def isochoric_volumetric_split(fun):
    """Apply the material formulation only on the isochoric part of the
    multiplicative split of the deformation gradient."""

    @wraps(fun)
    def apply_iso(C, *args, **kwargs):
        return fun(det(C) ** (-1 / 3) * C, *args, **kwargs)

    return apply_iso


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
    ..  math::

        \psi = \mu I_2 + \lambda \frac{I_1^2}{2}

    With the first and second invariant of the Green-Lagrange strain tensor
    :math:`\boldsymbol{E} = \frac{1}{2} (\boldsymbol{C} - \boldsymbol{1})`.

    ..  math::

        \hat{I}_1 &= \text{tr}\left( \boldsymbol{E} \right)

        \hat{I}_2 &= \boldsymbol{E} : \boldsymbol{E}

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.saint_venant_kirchhoff, mu=1.0, lambda=20.0)
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_saint_venant_kirchhoff.png
        :width: 400px

    """
    I1 = trace(C) / 2 - 3 / 2
    I2 = trace(C @ C) / 4 - trace(C) / 2 + 3 / 4
    return mu * I2 + lmbda * I1**2 / 2


def neo_hooke(C, mu):
    r"""Strain energy function of the isotropic hyperelastic
    `Neo-Hookean <https://en.wikipedia.org/wiki/Neo-Hookean_solid>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : float
        Shear modulus.

    Notes
    -----
    ..  math::

        \psi = \frac{\mu}{2} \left(\text{tr}\left(\hat{\boldsymbol{C}}\right) - 3\right)

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.neo_hooke, mu=1.0)
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_neo_hooke.png
        :width: 400px

    """
    return mu / 2 * (det(C) ** (-1 / 3) * trace(C) - 3)


def mooney_rivlin(C, C10, C01):
    r"""Strain energy function of the isotropic hyperelastic
    `Mooney-Rivlin <https://en.wikipedia.org/wiki/Mooney-Rivlin_solid>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    C10 : float
        First material parameter associated to the first invariant.
    C01 : float
        Second material parameter associated to the second invariant.

    Notes
    -----
    ..  math::

        \psi = C_{10} \left(\hat{I}_1 - 3 \right) + C_{01} \left(\hat{I}_2 - 3 \right)

    With the first and second main invariant of the distortional part of the right
    Cauchy-Green deformation tensor.

    ..  math::

        \hat{I}_1 &= J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

        \hat{I}_2 &= J^{-4/3} \frac{1}{2} \left(
            \text{tr}\left(\boldsymbol{C}\right)^2 -
            \text{tr}\left(\boldsymbol{C}^2\right)
        \right)

    The doubled sum of both material parameters is equal to the shear modulus
    :math:`\mu`.

    ..  math::

        \mu = 2 \left( C_{10} + C_{01} \right)

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.mooney_rivlin, C10=0.3, C01=0.8)
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_mooney_rivlin.png
        :width: 400px

    """
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (I1**2 - J3**2 * trace(C @ C)) / 2
    return C10 * (I1 - 3) + C01 * (I2 - 3)


def yeoh(C, C10, C20, C30):
    r"""Strain energy function of the isotropic hyperelastic
    `Yeoh <https://en.wikipedia.org/wiki/Yeoh_(hyperelastic_model)>`_
    material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    C10 : float
        Material parameter associated to the linear term of the first invariant.
    C20 : float
        Material parameter associated to the quadratic term of the first invariant.
    C30 : float
        Material parameter associated to the cubic term of the first invariant.

    Notes
    -----
    ..  math::

        \psi = C_{10} \left(\hat{I}_1 - 3 \right) + C_{20} \left(\hat{I}_1 - 3 \right)^2
             + C_{30} \left(\hat{I}_1 - 3 \right)^3

    With the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor.

    ..  math::

        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

    The :math:`C_{10}` material parameter is equal to half the initial shear modulus
    :math:`\mu`.

    ..  math::

        \mu = 2 C_{10}

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.yeoh, C10=0.5, C20=-0.1, C30=0.02)
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_yeoh.png
        :width: 400px

    """

    I1 = det(C) ** (-1 / 3) * trace(C)
    return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3


def third_order_deformation(C, C10, C01, C11, C20, C30):
    r"""Strain energy function of the isotropic hyperelastic
    `Third-Order-Deformation <https://onlinelibrary.wiley.com/doi/abs/10.1002/app.1975.070190723>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
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
    ..  math::

        \psi &= C_{10} \left(\hat{I}_1 - 3 \right) + C_{01} \left(\hat{I}_2 - 3 \right)
             + C_{11} \left(\hat{I}_1 - 3 \right) \left(\hat{I}_2 - 3 \right)

            &+ C_{20} \left(\hat{I}_1 - 3 \right)^2
             + C_{30} \left(\hat{I}_1 - 3 \right)^3

    With the first and second main invariant of the distortional part of the right
    Cauchy-Green deformation tensor.

    ..  math::

        \hat{I}_1 &= J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

        \hat{I}_2 &= J^{-4/3} \frac{1}{2} \left(
            \text{tr}\left(\boldsymbol{C}\right)^2 -
            \text{tr}\left(\boldsymbol{C}^2\right)
        \right)

    The doubled sum of the material parameters :math:`C_{10}` and :math:`C_{01}` is
    equal to the initial shear modulus :math:`\mu`.

    ..  math::

        \mu = 2 \left( C_{10} + C_{01} \right)

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(
    >>>     fem.third_order_deformation, C10=0.5, C01=0.1, C11=0.01, C20=-0.1, C30=0.02
    >>> )
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_third_order_deformation.png
        :width: 400px

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
    ..  math::

        \psi = \sum_i \frac{2 \mu_i}{\alpha^2_i} \left(
            \lambda_1^{\alpha_i} + \lambda_2^{\alpha_i} + \lambda_3^{\alpha_i} - 3
        \right)

    The sum of the moduli :math:`\mu_i` is equal to the initial shear modulus
    :math:`\mu`.

    ..  math::

        \mu = \sum_i \mu_i

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.ogden, mu=[1, 0.2], alpha=[1.7, -1.5])
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_ogden.png
        :width: 400px

    """

    wC = det(C) ** (-1 / 3) * eigvalsh(C)
    return sum1([2 * m / a**2 * (sum1(wC ** (a / 2)) - 3) for m, a in zip(mu, alpha)])


def arruda_boyce(C, C1, limit):
    r"""Strain energy function of the isotropic hyperelastic
    `Arruda-Boyce <https://en.wikipedia.org/wiki/Arruda-Boyce_model>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    C1 : list of float
        Initial shear modulus.
    limit : list of float
        Limiting stretch at which the polymer chain network becomes locked 
        :math:`\lambda_m`.

    Notes
    -----
    ..  math::

        \psi = C_1 \sum_{i=1}^5 \alpha_i \beta^{i-1} \left( \hat{I}_1^i - 3^i \right)

    With the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor
    
    ..  math::
    
        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)
    
    and :math_`\alpha_i` and :math`\beta`.
    
    ..  math::
        
        \boldsymbol{\alpha} &= \begin{bmatrix} 
            \frac{1}{2} \\ 
            \frac{1}{20} \\
            \frac{11}{1050} \\
            \frac{19}{7000} \\
            \frac{519}{673750}
        \end{bmatrix}
        
        \beta &= \frac{1}{\lambda_m^2}
    
    The initial shear modulus is a function of both material parameters.
    
    ..  math::
        
        \mu = C_1 \left( 
            1 + \frac{3}{5 \lambda_m^2} + \frac{99}{175 \lambda_m^4} 
              + \frac{513}{875 \lambda_m^6} + \frac{42039}{67375 \lambda_m^8} 
        \right)

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.arruda_boyce, C1=1.0, limit=3.2)
    >>> ax = umat.plot(incompressible=True)
    
    ..  image:: images/umat_arruda_boyce.png
        :width: 400px

    """
    I1 = det(C) ** (-1 / 3) * trace(C)

    alphas = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    beta = 1 / limit**2

    out = []
    for j, alpha in enumerate(alphas):
        i = j + 1
        out.append(alpha * beta ** (i - 1) * (I1**i - 3**i))

    return C1 * sum1(out)


def extended_tube(C, Gc, delta, Ge, beta):
    r"""Strain energy function of the isotropic hyperelastic
    `Extended Tube <https://www.doi.org/10.5254/1.3538822>`_ [1] material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
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
    ..  math::

        \psi = \frac{G_c}{2} \left[ \frac{\left( 1 - \delta^2 \right)
            \left( \hat{I}_1 - 3 \right)}{1 - \delta^2 \left( \hat{I}_1 - 3 \right)} +
            \ln \left( 1 - \delta^2 \left( \hat{I}_1 - 3 \right) \right) \right] +
            \frac{2 G_e}{\beta^2} \left( \hat{\lambda}_1^{-\beta} +
            \hat{\lambda}_2^{-\beta} + \hat{\lambda}_3^{-\beta} - 3 \right)

    With the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor

    ..  math::

        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

    and the principal stretches, obtained from the distortional part of the right
    Cauchy-Green deformation tensor

    ..  math::

        \hat{\lambda}_\alpha = J^{-1/3} \lambda_\alpha

    The initial shear modulus results from the sum of the cross-link and the constraint
    contributions to the total initial shear modulus.

    ..  math::

        \mu = G_e + G_c

    Examples
    --------

    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(
    >>>     fem.extended_tube, Gc=0.1867, Ge=0.2169, beta=0.2, delta=0.09693
    >>> )
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_extended_tube.png
        :width: 400px

    References
    ----------
    [1] M. Kaliske and G. Heinrich, "An Extended Tube-Model for Rubber Elasticity:
    Statistical-Mechanical Theory and Finite Element Implementation", Rubber Chemistry
    and Technology, vol. 72, no. 4. Rubber Division, ACS, pp. 602–632, Sep. 01, 1999.
    doi: 10.5254/1.3538822.

    """
    J3 = det(C) ** (-1 / 3)
    D = J3 * trace(C)
    wC = J3 * eigvalsh(C)
    g = (1 - delta**2) * (D - 3) / (1 - delta**2 * (D - 3))
    Wc = Gc / 2 * (g + log(1 - delta**2 * (D - 3)))
    We = 2 * Ge / beta**2 * sum1(wC ** (-beta / 2) - 1)
    return Wc + We


def van_der_waals(C, mu, limit, a, beta):
    """Strain energy function of the Van der Waals material formulation.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.van_der_waals, mu=1.0, beta=0.1, a=0.5, limit=5.0)
    >>> ax = umat.plot(incompressible=True)

    ..  image:: images/umat_van_der_waals.png
        :width: 400px
    """
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (trace(C) ** 2 - J3**2 * trace(C @ C)) / 2
    Im = (1 - beta) * I1 + beta * I2
    Im.x[np.isclose(Im.x, 3)] += 1e-8
    eta = sqrt((Im - 3) / (limit**2 - 3))
    return mu * (
        -(limit**2 - 3) * (log(1 - eta) + eta) - 2 / 3 * a * ((Im - 3) / 2) ** (3 / 2)
    )


@isochoric_volumetric_split
def finite_strain_viscoelastic(C, Cin, mu, eta, dtime):
    """Finite strain viscoelastic material formulation.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(
    >>>     fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, dtime=1.0, nstatevars=6
    >>> )
    >>> ax = umat.plot(
    >>>    incompressible=True,
    >>>    ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
    >>>    ps=None,
    >>>    bx=None,
    >>> )

    ..  image:: images/umat_finite_strain_viscoelastic.png
        :width: 400px
    """

    # update of state variables by evolution equation
    Ci = from_triu_1d(Cin, like=C) + mu / eta * dtime * C
    Ci = det(Ci) ** (-1 / 3) * Ci

    # first invariant of elastic part of right Cauchy-Green deformation tensor
    I1 = trace(C @ inv(Ci))

    # strain energy function and state variable
    return mu / 2 * (I1 - 3), triu_1d(Ci)
