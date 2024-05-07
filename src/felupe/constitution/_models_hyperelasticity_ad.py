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
from tensortrax.math import linalg, log, special, sqrt
from tensortrax.math import sum as sum1
from tensortrax.math import trace

det = linalg.det
inv = linalg.inv
eigvalsh = linalg.eigvalsh

from_triu_1d = special.from_triu_1d
triu_1d = special.triu_1d


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
    The strain energy function is given in Eq. :eq:`psi-nh`.

    ..  math::
        :label: psi-nh

        \psi = \frac{\mu}{2} \left(\text{tr}\left(\hat{\boldsymbol{C}}\right) - 3\right)

    Examples
    --------

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.neo_hooke, mu=1.0)
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

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.mooney_rivlin, C10=0.3, C01=0.8)
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
        label: shear-modulus-yeoh

        \mu = 2 C_{10}

    Examples
    --------

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.yeoh, C10=0.5, C20=-0.1, C30=0.02)
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

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...     fem.third_order_deformation, C10=0.5, C01=0.1, C11=0.01, C20=-0.1, C30=0.02
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
    return sum1([2 * m / a**2 * (sum1(wC ** (a / 2)) - 3) for m, a in zip(mu, alpha)])


def arruda_boyce(C, C1, limit):
    r"""Strain energy function of the isotropic hyperelastic
    `Arruda-Boyce <https://en.wikipedia.org/wiki/Arruda-Boyce_model>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    C1 : float
        Initial shear modulus.
    limit : float
        Limiting stretch :math:`\lambda_m` at which the polymer chain network becomes
        locked.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-ab`
    
    ..  math::
        :label: psi-ab

        \psi = C_1 \sum_{i=1}^5 \alpha_i \beta^{i-1} \left( \hat{I}_1^i - 3^i \right)

    with the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor as given in Eq. :eq:`invariants-ab`
    
    ..  math::
        :label: invariants-ab
    
        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)
    
    and :math:`\alpha_i` and :math:`\beta` as denoted in Eq. :eq:`ab-param`.
    
    ..  math::
        :label: ab-param
        
        \boldsymbol{\alpha} &= \begin{bmatrix} 
            \frac{1}{2} \\ 
            \frac{1}{20} \\
            \frac{11}{1050} \\
            \frac{19}{7000} \\
            \frac{519}{673750}
        \end{bmatrix}
        
        \beta &= \frac{1}{\lambda_m^2}
    
    The initial shear modulus is a function of both material parameters, see Eq.
    :eq:`shear-modulus-ab`.
    
    ..  math::
        :label: shear-modulus-ab
        
        \mu = C_1 \left( 
            1 + \frac{3}{5 \lambda_m^2} + \frac{99}{175 \lambda_m^4} 
              + \frac{513}{875 \lambda_m^6} + \frac{42039}{67375 \lambda_m^8} 
        \right)

    Examples
    --------
    
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.arruda_boyce, C1=1.0, limit=3.2)
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

    alphas = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    beta = 1 / limit**2

    out = []
    for j, alpha in enumerate(alphas):
        i = j + 1
        out.append(alpha * beta ** (i - 1) * (I1**i - 3**i))

    return C1 * sum1(out)


def extended_tube(C, Gc, delta, Ge, beta):
    r"""Strain energy function of the isotropic hyperelastic
    `Extended Tube <https://www.doi.org/10.5254/1.3538822>`_ [1]_ material formulation.

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

        \lambda^2_\alpha &= \text{eigvals}\left \boldsymbol{C} \right)

        \hat{\lambda}_\alpha &= J^{-1/3} \lambda_\alpha

    The initial shear modulus results from the sum of the cross-link and the constraint
    contributions to the total initial shear modulus as denoted in Eq.
    :eq:`shear-modulus-et`.

    ..  math::
        :label: shear-modulus-et

        \mu = G_e + G_c

    Examples
    --------

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...     fem.extended_tube, Gc=0.1867, Ge=0.2169, beta=0.2, delta=0.09693
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
    We = 2 * Ge / beta**2 * sum1(wC ** (-beta / 2) - 1)
    return Wc + We


def van_der_waals(C, mu, limit, a, beta):
    r"""Strain energy function of the
    `Van der Waals <https://doi.org/10.1016/0032-3861(81)90200-7>`_ [1]_ material
    formulation.,

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : float
        Initial shear modulus.
    limit : float
        Limiting stretch :math:`\lambda_m` at which the polymer chain network becomes
        locked.
    a : float
        Attractive interactions between the quasi-particles.
    beta : float
        Mixed-Invariant factor: 0 for pure I1- and 1 for pure I2-contribution.

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.van_der_waals, mu=1.0, beta=0.1, a=0.5, limit=5.0)
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
    ..  [1] H.-G. Kilian, "Equation of state of real networks", Polymer, vol. 22, no. 2.
        Elsevier BV, pp. 209–217, Feb. 1981. doi:
        `10.1016/0032-3861(81)90200-7 <https://www.doi.org/10.1016/0032-3861(81)90200-7>`_.

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


def finite_strain_viscoelastic(C, Cin, mu, eta, dtime):
    r"""Multiplicative
    `finite strain viscoelastic <https://doi.org/10.1016/j.cma.2013.07.004>`_ [1]_
    material formulation.

    Notes
    -----
    The material formulation is built upon the multiplicative decomposition of the
    deformation gradient tensor into an elastic and an inelastic part, see Eq.
    :eq:`multiplicative-split`.

    ..  math::
        :label: multiplicative-split

        \boldsymbol{F} &= \boldsymbol{F}_e \boldsymbol{F}_i

        \boldsymbol{C}_e &= \boldsymbol{F}_e^T \boldsymbol{F}_e

        \boldsymbol{C}_i &= \boldsymbol{F}_i^T \boldsymbol{F}_i

        \text{tr}\left( \boldsymbol{C}_e \right) &= \text{tr}\left(
            \boldsymbol{C} \boldsymbol{C}_i^{-1}
        \right)

    The components of the inelastic right Cauchy-Green deformation tensor are used as
    state variables with the evolution equation and its explicit update formula as given
    in Eq. :eq:`evolution` [1]_. Here, the inelastic right Cauchy-Green deformation
    tensor is enforced to be an unimodular tensor.

    ..  math::
        :label: evolution

        \dot{\boldsymbol{C}}_i &= \frac{\mu}{\eta} \text{dev}\left(
            \hat{\boldsymbol{C}} \boldsymbol{C}_i^{-1}
        \right) \boldsymbol{C}_i

        \boldsymbol{X} &= \boldsymbol{C}_{i,n}
            + \frac{\Delta t \mu}{\eta} \hat{\boldsymbol{C}}

        \boldsymbol{C}_i &= \det(\boldsymbol{X})^{-1/3}\ \boldsymbol{X}

    The distortional part of the strain energy density per unit undeformed volume is
    assumed to be of a Neo-Hookean form, see Eq. :eq:`nh-w`.

    ..  math::
        :label: nh-w

        \hat{\psi} = \frac{\mu}{2} \left( \text{tr}\left(
            \hat{\boldsymbol{C}} \boldsymbol{C}_i^{-1}
        \right) - 3 \right)

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...     fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, dtime=1.0, nstatevars=6
        ... )
        >>> ax = umat.plot(
        ...    incompressible=True,
        ...    ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
        ...    ps=None,
        ...    bx=None,
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
    ..  [1] A. V. Shutov, R. Landgraf, and J. Ihlemann, "An explicit solution for
        implicit time stepping in multiplicative finite strain viscoelasticity",
        Computer Methods in Applied Mechanics and Engineering, vol. 265. Elsevier BV,
        pp. 213–225, Oct. 2013. doi:
        `10.1016/j.cma.2013.07.004 <https://doi.org/10.1016/j.cma.2013.07.004>`_.

    """
    J3 = det(C) ** (-1 / 3)

    # update of state variables by evolution equation
    Ci = from_triu_1d(Cin, like=C) + (mu / eta * dtime) * (J3 * C)
    Ci = det(Ci) ** (-1 / 3) * Ci

    # first invariant of elastic part of right Cauchy-Green deformation tensor
    I1 = J3 * trace(C @ inv(Ci))

    # strain energy function and state variable
    return mu / 2 * (I1 - 3), triu_1d(Ci)


# default material parameters
saint_venant_kirchhoff.kwargs = dict(mu=0.0, lmbda=0.0)
neo_hooke.kwargs = dict(mu=0)
mooney_rivlin.kwargs = dict(C10=0, C01=0)
yeoh.kwargs = dict(C10=0, C20=0, C30=0)
third_order_deformation.kwargs = dict(C10=0, C01=0, C11=0, C20=0, C30=0)
ogden.kwargs = dict(mu=[1, 1], alpha=[2, -2])
arruda_boyce.kwargs = dict(C1=0, limit=1000)
extended_tube.kwargs = dict(Gc=0, Ge=0, beta=1, delta=0)
van_der_waals.kwargs = dict(mu=0, beta=0, a=0, limit=1000)
