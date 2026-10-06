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

from tensortrax.math import array, maximum, sqrt
from tensortrax.math.linalg import det, eigvalsh, expm, inv
from tensortrax.math.special import dev, from_triu_1d, sym, triu_1d, try_stack

from ..._total_lagrange import total_lagrange


@total_lagrange
def morph(F, statevars, p):
    r"""Second Piola-Kirchhoff stress tensor of the
    `MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_ model formulation [1]_.

    Parameters
    ----------
    F : tensortrax.Tensor or jax.Array
        Deformation gradient tensor.
    statevars : array
        Vector of stacked state variables (CTS, C, SA).
    p : list of float
        A list which contains the 8 material parameters.

    Notes
    -----
    The MORPH material model is implemented as a second Piola-Kirchhoff stress-based
    formulation with automatic differentiation. The Tresca invariant of the distortional
    part of the right Cauchy-Green deformation tensor is used as internal state
    variable, see Eq. :eq:`morph-state`.

    ..  warning::
        While the `MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_-material
        formulation captures the Mullins effect and quasi-static hysteresis effects of
        rubber mixtures very nicely, it has been observed to be unstable for medium- to
        highly-distorted states of deformation.

    ..  math::
        :label: morph-state

        \boldsymbol{C} &= \boldsymbol{F}^T \boldsymbol{F}

        I_3 &= \det (\boldsymbol{C})

        \hat{\boldsymbol{C}} &= I_3^{-1/3} \boldsymbol{C}

        \hat{\lambda}^2_\alpha &= \text{eigvals}(\hat{\boldsymbol{C}})

        \hat{C}_T &= \max \left( \hat{\lambda}^2_\alpha - \hat{\lambda}^2_\beta \right)

        \hat{C}_T^S &= \max \left( \hat{C}_T, \hat{C}_{T,n}^S \right)

    A sigmoid-function is used inside the deformation-dependent variables
    :math:`\alpha`, :math:`\beta` and :math:`\gamma`, see Eq. :eq:`morph-sigmoid`.

    ..  math::
        :label: morph-sigmoid

        f(x) &= \frac{1}{\sqrt{1 + x^2}}

        \alpha &= p_1 + p_2 \ f(p_3\ C_T^S)

        \beta &= p_4\ f(p_3\ C_T^S)

        \gamma &= p_5\ C_T^S\ \left( 1 - f\left(\frac{C_T^S}{p_6}\right) \right)

    The rate of deformation is described by the incremental modified Jaumann rate of
    the left Cauchy-Green deformation tensor and its Tresca-invariant, see Eq.
    :eq:`morph-rate-of-deformation`. It is the push-forward of the non-symmetric
    Lagrangian rate tensor :math:`\hat{\boldsymbol{L}}` of [1]_. Both tensors are
    similar and share their eigenvalues, but only the push-forward is symmetric. Note
    that :math:`\Delta\hat{\boldsymbol{b}}^\ast` is not the difference of the left
    Cauchy-Green deformation tensors of the current and the previous state.

    ..  math::
        :label: morph-rate-of-deformation

        \Delta\boldsymbol{C} &= \boldsymbol{C} - \boldsymbol{C}_n

        \hat{\boldsymbol{L}} &= \text{sym}\left(
                \text{dev}(\boldsymbol{C}^{-1} \Delta\boldsymbol{C})
            \right) \hat{\boldsymbol{C}}

        \Delta\hat{\boldsymbol{b}}^\ast &= \boldsymbol{F}\ \hat{\boldsymbol{L}}\
            \boldsymbol{F}^{-1} = I_3^{-1/3}\ \boldsymbol{F}\ \text{sym}\left(
                \text{dev}(\boldsymbol{C}^{-1} \Delta\boldsymbol{C})
            \right) \boldsymbol{F}^T

        \lambda_\alpha &= \text{eigvals}(\Delta\hat{\boldsymbol{b}}^\ast)

        \Delta\hat{b}^\ast_T &= \max \left( \lambda_\alpha - \lambda_\beta \right)

    The additional stresses evolve between the limiting stresses, see Eq.
    :eq:`morph-stresses`. The additional deviatoric-enforcement terms [1]_ are neglected
    in this implementation.

    ..  math::
        :label: morph-stresses

        \boldsymbol{\tau}_L &= \gamma \exp \left(
                p_7 \frac{\Delta\hat{\boldsymbol{b}}^\ast}{\Delta\hat{b}^\ast_T}
                \frac{\hat{C}_T}{\hat{C}_T^S}
            \right) + p_8 \frac{\Delta\hat{\boldsymbol{b}}^\ast}{\Delta\hat{b}^\ast_T}

        \boldsymbol{S}_L &= \boldsymbol{F}^{-1}\ \boldsymbol{\tau}_L\ \boldsymbol{F}^{-T}

        \boldsymbol{S}_A &= \frac{
            \boldsymbol{S}_{A,n} + \beta\ \Delta\hat{b}^\ast_T\ \boldsymbol{S}_L
        }{1 + \beta\ \Delta\hat{b}^\ast_T}

        \boldsymbol{S} &= 2 \alpha\ \text{dev}( \hat{\boldsymbol{C}} )
            \boldsymbol{C}^{-1}+\text{dev}\left(\boldsymbol{S}_A\ \boldsymbol{C}\right)
            \boldsymbol{C}^{-1}

    ..  note::
        Only the upper-triangle entries of the symmetric stress-tensor state
        variables are stored in the solid body. Hence, it is necessary to extract such
        variables with :func:`tm.special.from_triu_1d` and export them as
        :func:`tm.special.triu_1d`.

    Examples
    --------
    First, choose the desired automatic differentiation backend

    ..  plot::
        :context: close-figs

        >>> import felupe as fem
        >>>
        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    and create the material.

    ..  plot::
        :context: close-figs

        >>> umat = mat.Material(
        ...     mat.models.lagrange.morph,
        ...     p=[0.039, 0.371, 0.174, 2.41, 0.0094, 6.84, 5.65, 0.244],
        ...     nstatevars=13,
        ... )
        >>> ax = umat.plot(
        ...    incompressible=True,
        ...    ux=fem.math.linsteps(
        ...        # [1, 2, 1, 2.75, 1, 3.5, 1, 4.2, 1, 4.8, 1, 4.8, 1],
        ...        [1, 2.75, 1, 2.75],
        ...        num=20,
        ...    ),
        ...    ps=None,
        ...    bx=None,
        ... )

    References
    ----------
    .. [1] D. Besdo and J. Ihlemann, "A phenomenological constitutive model for
       rubberlike materials and its numerical applications", International Journal
       of Plasticity, vol. 19, no. 7. Elsevier BV, pp. 1019–1036, Jul. 2003. doi:
       `10.1016/s0749-6419(02)00091-8 <https://doi.org/10.1016/s0749-6419(02)00091-8>`_.

    See Also
    --------
    felupe.constitution.tensortrax.models.lagrange.morph_representative_directions :
        Strain energy function of the MORPH model formulation, implemented by the
        concept of representative directions.
    felupe.constitution.jax.models.lagrange.morph_representative_directions : Strain
        energy function of the MORPH model formulation, implemented by the concept of
        representative directions.
    """

    # right Cauchy-Green deformation tensor
    C = F.T @ F

    # extract old state variables
    CTSn = array(statevars[0], like=C[0, 0])
    Cn = from_triu_1d(statevars[1:7], like=C)
    SAn = from_triu_1d(statevars[7:13], like=C)

    # distortional part of right Cauchy-Green deformation tensor
    I3 = det(C)
    CG = C * I3 ** (-1 / 3)

    # inverse of and incremental right Cauchy-Green deformation tensor
    invC = inv(C)
    dC = C - Cn

    # eigenvalues of right Cauchy-Green deformation tensor (sorted in ascending order)
    λCG = eigvalsh(CG)

    # Tresca invariant of distortional part of right Cauchy-Green deformation tensor
    CTG = λCG[-1] - λCG[0]

    # maximum Tresca invariant in load history
    CTS = maximum(CTG, CTSn)

    def sigmoid(x):
        "Algebraic sigmoid function."
        return 1 / sqrt(1 + x**2)

    # material parameters
    α = p[0] + p[1] * sigmoid(p[2] * CTS)
    β = p[3] * sigmoid(p[2] * CTS)
    γ = p[4] * CTS * (1 - sigmoid(CTS / p[5]))

    invF = inv(F)
    dbG = det(F) ** (-2 / 3) * F @ sym(dev(invC @ dC)) @ F.T

    λdbG = eigvalsh(dbG)
    dbTG = λdbG[-1] - λdbG[0]

    # limiting stresses "L" and additional stresses "A"
    # 𝜏 is a Kirchhoff stress tensor, S are 2nd Piola-Kirchhoff stress tensors
    𝜏L = γ * expm(p[6] * dbG / dbTG * CTG / CTS) + p[7] * dbG / dbTG
    SL = invF @ 𝜏L @ invF.T
    SA = (SAn + β * dbTG * SL) / (1 + β * dbTG)

    # second Piola-Kirchhoff stress tensor
    S = 2 * α * dev(CG) @ invC + dev(SA @ C) @ invC
    statevars_new = try_stack([[CTS], triu_1d(C), triu_1d(SA)], fallback=statevars)

    return S, statevars_new
