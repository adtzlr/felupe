# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import warnings

import numpy as np

from ...math import cdya_ik, dot, eigh, identity, ravel, reshape, sym, transpose
from .._base import ConstitutiveMaterial
from .._material import takes_time_increment


def _diagonal_function(λ, m):
    r"""Return the diagonal functions :math:`e_a(\lambda_a)` of the Seth-Hill strain
    tensors and their first four derivatives w.r.t. the eigenvalues :math:`\lambda_a`
    of the right Cauchy-Green deformation tensor, i.e. :math:`d_a / 2` and :math:`f_a /
    4` are the first and the second derivatives, see :class:`~felupe.MaterialStrain`."""

    if m == 0:
        return np.log(λ) / 2, 1 / (2 * λ), -1 / (2 * λ**2), 1 / λ**3, -3 / λ**4

    p = m / 2
    return (
        (λ**p - 1) / m,
        λ ** (p - 1) / 2,
        (p - 1) * λ ** (p - 2) / 2,
        (p - 1) * (p - 2) * λ ** (p - 3) / 2,
        (p - 1) * (p - 2) * (p - 3) * λ ** (p - 4) / 2,
    )


def _zeta_index(a, b, c):
    r"""Return the index of the coefficient :math:`\zeta_{abc}` in the list of the
    coefficients :math:`[\xi_{00}, \xi_{01}, \dots, \xi_{22}, \eta]`, i.e.
    :math:`\xi_{ab}` for the indices :math:`(a, b, b)` in any order and :math:`\eta`
    for distinct indices."""

    if len({a, b, c}) == 3:
        return 9

    x, y = (a, b) if b == c else (b, a) if a == c else (c, a)

    return 3 * x + y


_ZETA_INDICES = [_zeta_index(a, b, c) for a, b, c in np.ndindex(3, 3, 3)]


def _seth_hill_coefficients(λ, m, zeta=True, tol=1e-3):
    r"""Return the coefficients :math:`\vartheta_{ab}` of the projection tensor
    :math:`\mathbb{P}` with :math:`\vartheta_{aa} = d_a / 2` and (optionally) the
    coefficients :math:`\zeta_{abc}` of the tensor :math:`\boldsymbol{T} : \mathbb{L}`,
    i.e. :math:`\zeta_{aaa} = f_a / 8`, :math:`\zeta_{abb} = \zeta_{bab} = \zeta_{bba} =
    \xi_{ab}` and :math:`\zeta_{abc} = \eta` for distinct indices, for the eigenvalues
    :math:`\lambda_a` in ascending order, see :class:`~felupe.MaterialStrain`. For
    (nearly) equal eigenvalues, i.e. a relative difference below ``tol``, the
    coefficients are evaluated by Taylor series."""

    e, de, d2e, d3e, d4e = _diagonal_function(λ, m)

    # differences of the eigenvalues Δ_ab = λ_a - λ_b
    Δ = λ[:, None] - λ[None, :]
    equal = np.abs(Δ) < tol * λ[None, :]

    # Taylor series about λ_b
    de, d2e, d3e, d4e = de[None, :], d2e[None, :], d3e[None, :], d4e[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        ϑ = (e[:, None] - e[None, :]) / Δ

    ϑ = np.where(equal, de + d2e * Δ / 2 + d3e * Δ**2 / 6 + d4e * Δ**3 / 24, ϑ)

    if not zeta:
        return ϑ

    # ξ_ab = (ϑ_ab - d_b / 2) / (λ_a - λ_b) with d_b / 2 = e'_b and the limit f_a / 8
    with np.errstate(divide="ignore", invalid="ignore"):
        ξ = (ϑ - de) / Δ

    ξ = np.where(equal, d2e / 2 + d3e * Δ / 6 + d4e * Δ**2 / 24, ξ)

    # η for the eigenvalues in ascending order (Taylor series about λ_0)
    Δ10, Δ20 = Δ[1, 0], Δ[2, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        η = (ϑ[2, 1] - ϑ[1, 0]) / Δ20

    η_taylor = (
        d2e[0, 0] / 2
        + d3e[0, 0] * (Δ10 + Δ20) / 6
        + d4e[0, 0] * (Δ10**2 + Δ10 * Δ20 + Δ20**2) / 24
    )
    η = np.where(equal[2, 0], η_taylor, η)

    # symmetric coefficients ζ_abc
    coefficients = np.concatenate([ξ.reshape(9, *λ.shape[1:]), η[None]])

    return ϑ, coefficients[_ZETA_INDICES].reshape(3, 3, 3, *λ.shape[1:])


def _seth_hill_stress(decomposition, stress, m):
    r"""Return the second Piola-Kirchhoff stress tensor :math:`\boldsymbol{S} =
    \boldsymbol{T} : \mathbb{P}` of the Seth-Hill framework for a given eigenvalue
    decomposition of the right Cauchy-Green deformation tensor and the (symmetric)
    stress tensor :math:`\boldsymbol{T}` of the material, see
    :class:`~felupe.MaterialStrain`."""

    λ, n = decomposition
    ϑ = _seth_hill_coefficients(λ, m, zeta=False)

    T = np.einsum("Ia...,IJ...,Jb...->ab...", n, stress, n)

    return np.einsum("Ia...,ab...,Jb...->IJ...", n, 2 * ϑ * T, n)


def _seth_hill_elasticity(F, decomposition, stress, elasticity, m):
    r"""Return the fourth-order elasticity tensor of the first Piola-Kirchhoff stress
    tensor, see Eq. :eq:`pk1-elasticity`, with the second Piola-Kirchhoff stress tensor
    :math:`\boldsymbol{S} = \boldsymbol{T} : \mathbb{P}` and the elasticity tensor
    :math:`\mathbb{C} = \mathbb{P} : \mathbb{E} : \mathbb{P} + \boldsymbol{T} :
    \mathbb{L}` of the Seth-Hill framework for a given deformation gradient, its
    eigenvalue decomposition of the right Cauchy-Green deformation tensor, the
    (symmetric) stress tensor :math:`\boldsymbol{T}` and the (minor-symmetric)
    elasticity tensor :math:`\mathbb{E} = \partial \boldsymbol{T} / \partial
    \boldsymbol{E}` of the material, see :class:`~felupe.MaterialStrain`."""

    λ, n = decomposition
    ϑ, ζ = _seth_hill_coefficients(λ, m)

    # stress and elasticity tensors of the material in the eigenvalue bases
    T = np.einsum("Ia...,IJ...,Jb...->ab...", n, stress, n)
    E = np.einsum(
        "Ia...,Jb...,Kc...,Ld...,IJKL...->abcd...",
        n,
        n,
        n,
        n,
        elasticity,
        optimize=True,
    )

    # P : E : P in the eigenvalue bases
    C = np.einsum("ab...,cd...,abcd...->abcd...", 4 * ϑ, ϑ, E)

    # T : L in the eigenvalue bases, the parts with Kronecker deltas are added to the
    # diagonals (writeable views) of C
    C_bc = np.einsum("abbd...->abd...", C)  # c = b
    C_ac = np.einsum("abad...->abd...", C)  # c = a
    C_bd = np.einsum("abcb...->abc...", C)  # d = b
    C_ad = np.einsum("abca...->abc...", C)  # d = a

    C_bc += np.einsum("abd...,ad...->abd...", 2 * ζ, T)
    C_ac += np.einsum("abd...,bd...->abd...", 2 * ζ, T)
    C_bd += np.einsum("abc...,ac...->abc...", 2 * ζ, T)
    C_ad += np.einsum("abc...,bc...->abc...", 2 * ζ, T)

    # A_iJkL = F_iI F_kK C_IJKL + δ_ik S_JL with C_IJKL = n_Ia n_Jb n_Kc n_Ld C_abcd
    Fn = dot(F, n)
    S = np.einsum("Ia...,ab...,Jb...->IJ...", n, 2 * ϑ * T, n)

    A = np.einsum(
        "ia...,Jb...,kc...,Ld...,abcd...->iJkL...", Fn, n, Fn, n, C, optimize=True
    )
    A += cdya_ik(np.eye(len(F)), S)

    return A


class MaterialStrain(ConstitutiveMaterial):
    r"""A strain-based user-defined material definition with a given function for the
    stress tensor and the (fourth-order) elasticity tensor.

    Parameters
    ----------
    material : callable
        The material model formulation. Function signature must be
        ``lambda dε, εn, σn, ζn, **kwargs: dσdε, σ, ζ``. Input arguments are the strain
        increment, old strain, old stress, the list of old state variables and optional
        keyword arguments. The function must return the algorithmic consistent
        elasticity tensor, new stress and the list of new state variables. The provided
        strain and required stress quantities are selected by the framework argument.
        A material formulation with an argument ``dtime`` without a default value, which
        is not given as keyword-argument, is time-dependent, see the notes.
    dim : int, optional
        The dimension of the material formulation. Default is 3.
    statevars : tuple of int, optional
        A tuple containing the shape of each state variable. Default is (0, ).
    framework : str, optional
        The framework to be used for the stress and strain formulations. "small-strain",
        "total-lagrange", "co-rotational" and "seth-hill" are supported. Default is
        "small-strain".
    symmetry : bool, optional
        Take the symmetric part of the returned stress and the minor symmetric-parts of
        the algorithmic consistent elasticity tensor. Default is True. May enhance
        performance if the material returns symmetric tensors.
    strain_exponent : float or None, optional
        The strain exponent :math:`m` of the Seth-Hill strain tensors, which is required
        for (and only allowed with) the Seth-Hill framework, e.g. 0 for the logarithmic
        (Hencky) strain tensor. Default is None.
    **kwargs : dict, optional
        Optional keyword-arguments (the material parameters) for the material model
        formulation.

        ..  deprecated:: 12.0.0
            The material parameter ``Δt``, the former name of the time increment of
            :func:`~felupe.linear_elastic_viscoelastic`, is still taken as ``dtime``
            (for a material model formulation with an argument ``dtime`` and without
            an argument ``Δt``), but it emits a :class:`DeprecationWarning`. It will be
            removed in 13.0.0. Use ``dtime`` instead, e.g.
            ``MaterialStrain(linear_elastic_viscoelastic, ..., dtime=0.5)`` instead of
            ``MaterialStrain(linear_elastic_viscoelastic, ..., Δt=0.5)``.

    Notes
    -----
    **Time-dependent materials**: If the material model formulation has an argument
    ``dtime`` without a default value, which is not given as keyword-argument, the
    material is time-dependent (the attribute ``time_dependent`` is True), e.g.
    :func:`~felupe.linear_elastic_viscoelastic`. Then, the time increment is an input of
    the material, like the deformation gradient, i.e. it is given by the keyword-argument
    ``dtime`` of :meth:`gradient` and :meth:`hessian` and it is passed to the material
    model formulation. A time increment which is given as keyword-argument is a constant
    material parameter.

    The (default) small-strain framework evaluates the strain tensor as the symmetric
    part of the displacement gradient, see Eq. :eq:`small-strain`.

    ..  math::
        :label: small-strain

        \boldsymbol{\varepsilon} = \operatorname{sym} \left(
            \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{x}} \right)

    The Total-Lagrange framework uses the Green-Lagrange strain, see Eq.
    :eq:`gl-strain`,

    ..  math::
        :label: gl-strain

        \boldsymbol{E} = \frac{1}{2} \left( \boldsymbol{C} - \boldsymbol{1} \right)

    with the right Cauchy-Green deformation tensor, as denoted in Eq.
    :eq:`right-cauchy-green-deformation`.

    ..  math::
        :label: right-cauchy-green-deformation

        \boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}

    Within the Total-Lagrange framework, the second Piola-Kirchhoff stress tensor
    :math:`\boldsymbol{S}`, as a function of the Green-Lagrange strain tensor
    :math:`\boldsymbol{E}`, is converted to the first Piola-Kirchhoff stress tensor
    :math:`\boldsymbol{P}`, see Eq. :eq:`pk1-stress`.

    ..  math::
        :label: pk1-stress

        \boldsymbol{P} = \boldsymbol{F}\ \boldsymbol{S}

    Furthermore, the fourth-order material elasticity tensor in the Total-Lagrange
    framework is also converted for algorithmic consistency, see Eq.
    :eq:`pk1-elasticity`.

    ..  math::
        :label: pk1-elasticity

        \mathbb{A}_{iJkL} = F_{iI}\ F_{kK}\ \mathbb{C}_{IJKL}
            + \delta_{ik}\ S_{JL}

    The Seth-Hill framework uses the generalized strain tensors of the Seth-Hill family
    with the strain exponent :math:`m` (the argument ``strain_exponent``), in the
    notation of Miehe and Lambrecht [1]_. They are evaluated by the eigenvalues
    :math:`\lambda_a` (the squares of the principal stretches) and the eigenvalue bases
    :math:`\boldsymbol{M}_a = \boldsymbol{n}_a \otimes \boldsymbol{n}_a` of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`seth-hill-strain`.

    ..  math::
        :label: seth-hill-strain

        \boldsymbol{C} = \sum_{a=1}^3 \lambda_a\ \boldsymbol{M}_a
        \qquad
        \boldsymbol{E} = \sum_{a=1}^3 e_a\ \boldsymbol{M}_a
        \qquad
        e_a = \begin{cases}
            \frac{1}{m} \left( \lambda_a^{m/2} - 1 \right) & m \neq 0 \\
            \frac{1}{2} \ln \lambda_a & m = 0
        \end{cases}

    The particular choices :math:`m = 0, 1, 2` define the Hencky strain, the Biot
    strain and the Green strain. The stress tensor :math:`\boldsymbol{T} = \partial_{
    \boldsymbol{E}} \hat{\psi}` and the elasticity tensor :math:`\mathbb{E} =
    \partial^2_{\boldsymbol{E}\boldsymbol{E}} \hat{\psi}` of the material are dual to
    the Seth-Hill strain tensor. They are converted to the second Piola-Kirchhoff
    stress tensor and to the elasticity tensor of the Total-Lagrange framework by the
    projection tensors :math:`\mathbb{P} = 2\ \partial_{\boldsymbol{C}}
    \boldsymbol{E}` and :math:`\mathbb{L} = 4\ \partial^2_{\boldsymbol{C}\boldsymbol{C}}
    \boldsymbol{E}`, see Eq. :eq:`seth-hill-stress`, which are further converted by Eq.
    :eq:`pk1-stress` and Eq. :eq:`pk1-elasticity`.

    ..  math::
        :label: seth-hill-stress

        \boldsymbol{S} = \boldsymbol{T} : \mathbb{P}
        \qquad
        \mathbb{C} = \mathbb{P} : \mathbb{E} : \mathbb{P} + \boldsymbol{T} : \mathbb{L}

    The spectral representations of the projection tensors are given in Eq.
    :eq:`seth-hill-projection` with the tensor operators :math:`\mathbb{G}^{ijkl}_{ab}
    = M_a^{ik} M_b^{jl} + M_a^{il} M_b^{jk}` and :math:`\mathbb{F}^{ijkl}_{a(bc)} =
    M_a^{ik} (\boldsymbol{M}_b \boldsymbol{T} \boldsymbol{M}_c)^{jl} + M_a^{il}
    (\boldsymbol{M}_b \boldsymbol{T} \boldsymbol{M}_c)^{jk}` (and
    :math:`\mathbb{F}_{(bc)a}` with interchanged arguments).

    ..  math::
        :label: seth-hill-projection

        \mathbb{P} &= \sum_a^3 d_a\ \boldsymbol{M}_a \otimes \boldsymbol{M}_a
            + \sum_a^3 \sum_{b \neq a}^3 \vartheta_{ab}\ \mathbb{G}_{ab}

        \boldsymbol{T} : \mathbb{L} &= \sum_a^3 f_a\ (\boldsymbol{M}_a : \boldsymbol{T})
            \ \boldsymbol{M}_a \otimes \boldsymbol{M}_a
            + \sum_a^3 \sum_{b \neq a}^3 \sum_{c \neq a, c \neq b}^3 2\ \eta \left(
                \mathbb{F}_{a(bc)} + \mathbb{F}_{(bc)a} \right)

            &+ \sum_a^3 \sum_{b \neq a}^3 2\ \xi_{ab} \left( \mathbb{F}_{a(bb)}
                + \mathbb{F}_{(bb)a} + \mathbb{F}_{b(ab)} + \mathbb{F}_{(ab)b}
                + \mathbb{F}_{b(ba)} + \mathbb{F}_{(ba)b} \right)

    The coefficients are given in Eq. :eq:`seth-hill-coefficients`.

    ..  math::
        :label: seth-hill-coefficients

        d_a &= 2\ \partial_{\lambda_a} e_a
        \qquad
        f_a = 4\ \partial^2_{\lambda_a \lambda_a} e_a

        \vartheta_{ab} &= \frac{e_a - e_b}{\lambda_a - \lambda_b}
        \qquad
        \xi_{ab} = \frac{\vartheta_{ab} - \frac{1}{2} d_b}{\lambda_a - \lambda_b}
        \qquad
        \eta = \sum_a^3 \sum_{b \neq a}^3 \sum_{c \neq a, c \neq b}^3
            \frac{e_a}{2 (\lambda_a - \lambda_b) (\lambda_a - \lambda_c)}

    For two equal eigenvalues :math:`\lambda_{\bar{a}} = \lambda_{\bar{b}} \neq
    \lambda_{\bar{c}}`, the coefficients approach the limits
    :math:`\vartheta_{\bar{a}\bar{b}} = \frac{1}{2} d_{\bar{a}}`,
    :math:`\xi_{\bar{a}\bar{b}} = \frac{1}{8} f_{\bar{a}}` and :math:`\eta =
    \xi_{\bar{c}\bar{a}}` and for three equal eigenvalues :math:`\eta = \frac{1}{8}
    f_{\bar{a}}`. Here, instead of the case distinction of the limits, the coefficients
    are evaluated by Taylor series for (nearly) equal eigenvalues, i.e. if their
    relative difference is below :math:`10^{-3}`, which avoids the cancellation of the
    difference quotients. The projections are evaluated by their components in the
    eigenvalue bases, where the tensor operators have only a few non-zero components.
    With the components :math:`\hat{T}_{ab} = n^I_a\ T_{IJ}\ n^J_b` and
    :math:`\hat{\mathbb{E}}_{abcd} = n^I_a\ n^J_b\ n^K_c\ n^L_d\ \mathbb{E}_{IJKL}` of
    the stress and of the elasticity tensor of the material, :math:`\vartheta_{aa} =
    \frac{1}{2} d_a` and the abbreviations (not used in [1]_) :math:`\zeta_{aaa} =
    \frac{1}{8} f_a`, :math:`\zeta_{abb} = \zeta_{bab} = \zeta_{bba} = \xi_{ab}` and
    :math:`\zeta_{abc} = \eta` for distinct indices, Eq. :eq:`seth-hill-stress` is
    given by Eq. :eq:`seth-hill-eigenbasis` (without a summation over the indices of
    the eigenvalue bases).

    ..  math::
        :label: seth-hill-eigenbasis

        \hat{S}_{ab} &= 2\ \vartheta_{ab}\ \hat{T}_{ab}

        \hat{\mathbb{C}}_{abcd} &= 4\ \vartheta_{ab}\ \vartheta_{cd}\
            \hat{\mathbb{E}}_{abcd}
            + 2\ \zeta_{abd} \left( \delta_{bc}\ \hat{T}_{ad}
                + \delta_{ac}\ \hat{T}_{bd} \right)
            + 2\ \zeta_{abc} \left( \delta_{bd}\ \hat{T}_{ac}
                + \delta_{ad}\ \hat{T}_{bc} \right)

    The co-rotational framework is the Seth-Hill framework with the Biot strain tensor
    :math:`\boldsymbol{E} = \boldsymbol{U} - \boldsymbol{1}` (:math:`m = 1`) with the
    right stretch tensor :math:`\boldsymbol{U}` of the polar decomposition
    :math:`\boldsymbol{F} = \boldsymbol{R}\ \boldsymbol{U}`. Its stress tensor
    :math:`\boldsymbol{T}` is the (symmetric) Biot stress tensor, i.e.
    :math:`\boldsymbol{P} : \delta \boldsymbol{F} = \boldsymbol{T} : \delta
    \boldsymbol{U}`, and :math:`\vartheta_{ab} = 1 / (\sqrt{\lambda_a} +
    \sqrt{\lambda_b})`, i.e. the second Piola-Kirchhoff stress tensor is the solution of
    the Lyapunov equation, see Eq. :eq:`corotational-lyapunov`. For a Biot stress tensor
    which is coaxial with the stretch tensor, e.g. of an isotropic elastic material, the
    first Piola-Kirchhoff stress tensor is :math:`\boldsymbol{P} = \boldsymbol{R}\
    \boldsymbol{T}`. The Total-Lagrange framework is the Seth-Hill framework with
    :math:`m = 2` (:math:`\vartheta_{ab} = \frac{1}{2}`, :math:`\xi_{ab} = \eta = 0`), which
    is evaluated without the eigenvalue decomposition.

    ..  math::
        :label: corotational-lyapunov

        \boldsymbol{U}\ \boldsymbol{S} + \boldsymbol{S}\ \boldsymbol{U}
            = 2\ \boldsymbol{T}

    The elasticity tensors of the frameworks are the exact linearizations of the first
    Piola-Kirchhoff stress tensors, i.e. Newton's method converges quadratically. The
    Cauchy stress tensors of the Total-Lagrange and of the Seth-Hill frameworks are
    symmetric. A symmetric elasticity tensor of the material leads to a symmetric
    elasticity tensor of the framework.

    Examples
    --------
    Take this code-block for a linear-elastic material formulation

    ..  plot::
        :context: close-figs

        import felupe as fem
        from felupe.math import identity, cdya_ik, dya, trace

        def linear_elastic(dε, εn, σn, ζn, λ, μ, **kwargs):
            '''3D linear-elastic material formulation.

            Arguments
            ---------
            dε : ndarray
                Incremental strain tensor.
            εn : ndarray
                Old strain tensor.
            σn : ndarray
                Old stress tensor.
            ζn : ndarray
                Old state variables.
            λ : float
                First Lamé-constant.
            μ : float
                Second Lamé-constant (shear modulus).
            '''

            # change of stress due to change of strain
            I = identity(dε)
            dσ = 2 * μ * dε + λ * trace(dε) * I

            # update stress and evaluate elasticity tensor
            σ = σn + dσ
            dσdε = 2 * μ * cdya_ik(I, I) + λ * dya(I, I)

            # update state variables (not used here)
            ζ = ζn

            return dσdε, σ, ζ

        umat = fem.MaterialStrain(material=linear_elastic, μ=1, λ=2)
        ax = umat.plot()

    or this minimal header as template.

    ..  plot::
        :context: close-figs

        def fun(dε, εn, σn, ζn, **kwargs):
            return dσdε, σ, ζ

        umat = fem.MaterialStrain(material=fun, **kwargs)

    The Total-Lagrange framework changes the linear-elastic material formulation to the
    Saint-Venant Kirchhoff material model formulation.

    ..  plot::
        :context: close-figs

        umat = fem.MaterialStrain(
            material=fem.linear_elastic, μ=1, λ=2, framework="total-lagrange"
        )
        ax = umat.plot(
            ux=fem.math.linsteps([1, 1.5], num=20),
            ps=fem.math.linsteps([1, 1.5], num=20),
            bx=fem.math.linsteps([1, 1.25], num=10),
        )

    The Seth-Hill framework with the logarithmic strain tensor (:math:`m = 0`) changes
    the linear-elastic material formulation to the Hencky material model formulation.
    In the same way, the linear-elastic-plastic material formulation is used in the
    logarithmic strain space, i.e. with an additive decomposition of the logarithmic
    strain tensor into elastic and plastic parts.

    ..  plot::
        :context: close-figs

        umat = fem.MaterialStrain(
            material=fem.linear_elastic,
            μ=1,
            λ=2,
            framework="seth-hill",
            strain_exponent=0,
        )
        ax = umat.plot(
            ux=fem.math.linsteps([1, 1.5], num=20),
            ps=fem.math.linsteps([1, 1.5], num=20),
            bx=fem.math.linsteps([1, 1.25], num=10),
        )

    See Also
    --------
    linear_elastic : 3D linear-elastic material formulation
    linear_elastic_plastic_isotropic_hardening : Linear-elastic-plastic material
        formulation with linear isotropic hardening (return mapping algorithm).
    LinearElasticPlasticIsotropicHardening : Linear-elastic-plastic material formulation
        with linear isotropic hardening (return mapping algorithm).

    References
    ----------
    ..  [1] C. Miehe and M. Lambrecht, "Algorithms for computation of stresses and
        elasticity moduli in terms of Seth-Hill's family of generalized strain
        tensors", Communications in Numerical Methods in Engineering, vol. 17, no. 5,
        pp. 337-353, 2001. doi: `10.1002/cnm.404 <https://doi.org/10.1002/cnm.404>`_.

    """

    def __init__(
        self,
        material,
        dim=3,
        statevars=(0,),
        framework="small-strain",
        symmetry=True,
        strain_exponent=None,
        **kwargs,
    ):
        self.material = material
        self.statevars_shape = statevars
        self.statevars_size = [np.prod(shape) for shape in statevars]
        self.statevars_offsets = np.cumsum(self.statevars_size)
        self.nstatevars = sum(self.statevars_size)

        self.framework = framework
        self.symmetry = symmetry

        # the strain exponent of the Seth-Hill strain tensor of the framework
        if framework == "seth-hill" and strain_exponent is None:
            raise ValueError(
                'The framework "seth-hill" requires the argument `strain_exponent`, '
                "e.g. `strain_exponent=0` for the logarithmic strain tensor."
            )

        if framework != "seth-hill" and strain_exponent is not None:
            raise ValueError(
                "The argument `strain_exponent` is only supported by the framework "
                '"seth-hill".'
            )

        exponents = {"total-lagrange": 2, "co-rotational": 1}
        self.strain_exponent = exponents.get(framework, strain_exponent)

        # the flag `tangent` is passed to the material for each evaluation, i.e. the
        # keyword-arguments contain only the parameters of the material
        kwargs.pop("tangent", None)

        # the former name `Δt` of the time increment of linear_elastic_viscoelastic()
        if "Δt" in kwargs and "dtime" not in kwargs:
            try:
                parameters = inspect.signature(material).parameters
            except (TypeError, ValueError):  # no signature available
                parameters = {}

            if "dtime" in parameters and "Δt" not in parameters:
                warnings.warn(
                    "The material parameter `Δt` of `MaterialStrain` is deprecated "
                    "and will be removed in v13.0.0. Use `dtime` instead, e.g. "
                    "`MaterialStrain(linear_elastic_viscoelastic, ..., dtime=0.5)` "
                    "instead of `MaterialStrain(linear_elastic_viscoelastic, ..., "
                    "Δt=0.5)`.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs["dtime"] = kwargs.pop("Δt")

        self.kwargs = kwargs

        # the time increment is an input, if it is an argument of the material model
        # formulation which is not given as material parameter
        self.time_dependent = takes_time_increment(material, self.kwargs)

        self.dim = dim
        self.x = [np.eye(dim), np.zeros(2 * dim**2 + self.nstatevars)]

        self.stress = self.gradient
        self.elasticity = self.hessian

    def extract(self, x):
        "Extract the input and evaluate strains, stresses and state variables."

        *values, decomposition = self._extract(x)

        rotation = None

        if self.framework == "co-rotational":
            # rotation tensor of the polar decomposition R = F U^(-1)
            λ, n = decomposition
            rotation = np.einsum("iJ...,Ja...,a...,Ka...->iK...", x[0], n, λ**-0.5, n)

        return (*values, rotation)

    def _extract(self, x):
        """Extract the input and evaluate strains, stresses and state variables. Instead
        of the rotation tensor, the eigenvalue decomposition of the right Cauchy-Green
        deformation tensor is returned for the co-rotational and the Seth-Hill
        frameworks (else None)."""

        # unpack deformation gradient F = dx/dX
        dim = self.dim
        dxdX, statevars = x

        decomposition = None

        if self.framework == "small-strain":
            strain = sym(dxdX - identity(dxdX))

        elif self.framework == "total-lagrange":
            strain = (dot(transpose(dxdX), dxdX) - identity(dxdX)) / 2

        elif self.framework in ["co-rotational", "seth-hill"]:
            # eigenvalues and eigenvectors of the right Cauchy-Green deformation tensor
            decomposition = eigh(dot(transpose(dxdX), dxdX))
            λ, n = decomposition
            e = _diagonal_function(λ, self.strain_exponent)[0]
            strain = np.einsum("Ia...,a...,Ja...->IJ...", n, e, n)

        else:
            raise NotImplementedError(f'Framework "{self.framework}" not implemented.')

        # separate strain and stress from state variables
        statevars_all = np.split(
            statevars, [*self.statevars_offsets, self.nstatevars + dim**2]
        )
        strain_old_1d, stress_old_1d = statevars_all[-2:]

        # list of state variables with original shapes
        shapes = self.statevars_shape
        statevars_old = [
            reshape(sv, shape).copy() for sv, shape in zip(statevars_all[:-2], shapes)
        ]

        # reshape strain and stress from (dim**2,) to (dim, dim)
        strain_old = strain_old_1d.reshape(dim, dim, *strain_old_1d.shape[1:])
        stress_old = stress_old_1d.reshape(dim, dim, *stress_old_1d.shape[1:])

        # change of strain
        dstrain = strain - strain_old

        return strain_old, dstrain, stress_old, statevars_old, decomposition

    def _gradient(self, x, **kwargs):
        strain_old, dstrain, stress_old, statevars_old, decomposition = self._extract(x)

        # the chunks of cells may be evaluated by threads: don't change the kwargs
        # (the given keyword-arguments contain the time increment of a time-dependent
        # material)
        kwargs = {**self.kwargs, **kwargs, "tangent": False}

        # unpack deformation gradient F = dx/dX
        dxdX, statevars = x

        dsde, stress_new, statevars_new_list = self.material(
            dstrain, strain_old, stress_old, statevars_old, **kwargs
        )

        strain_new_1d = (strain_old + dstrain).reshape(-1, *strain_old.shape[2:])
        stress_new_1d = stress_new.reshape(-1, *strain_old.shape[2:])

        statevars_new = np.concatenate(
            [
                *[ravel(sv) for sv in statevars_new_list],
                strain_new_1d,
                stress_new_1d,
            ],
            axis=0,
        )

        if self.symmetry:
            stress_new = sym(stress_new)

        if self.framework in ["co-rotational", "seth-hill"]:
            # convert the stress to the second Piola-Kirchhoff stress
            m = self.strain_exponent
            stress_new = _seth_hill_stress(decomposition, stress_new, m)

        if self.framework in ["total-lagrange", "co-rotational", "seth-hill"]:
            # convert second to first Piola-Kirchhoff stress
            stress_new = dot(dxdX, stress_new)

        return [stress_new, statevars_new]

    def _hessian(self, x, **kwargs):
        strain_old, dstrain, stress_old, statevars_old, decomposition = self._extract(x)

        # the chunks of cells may be evaluated by threads: don't change the kwargs
        # (the given keyword-arguments contain the time increment of a time-dependent
        # material)
        kwargs = {**self.kwargs, **kwargs, "tangent": True}

        # unpack deformation gradient F = dx/dX
        dxdX, statevars = x

        dsde, stress_new, statevars_new_list = self.material(
            dstrain, strain_old, stress_old, statevars_old, **kwargs
        )
        dsde = np.ascontiguousarray(dsde)

        if self.symmetry:
            # enforce minor symmetries on the algorithmic consistent elasticity tensor
            dsde = np.add(dsde, np.einsum("ijlk...->ijkl...", dsde), out=dsde)
            dsde = np.add(dsde, np.einsum("jikl...->ijkl...", dsde), out=dsde)
            dsde = np.multiply(dsde, 0.25, out=dsde)

        if self.symmetry and self.framework != "small-strain":
            stress_new = sym(stress_new)

        if self.framework == "total-lagrange":
            # convert elasticity tensor
            dsde = np.einsum(
                "iI...,kK...,IJKL...->iJkL...", dxdX, dxdX, dsde, optimize=True
            )
            dsde += cdya_ik(np.eye(self.dim), stress_new)

        elif self.framework in ["co-rotational", "seth-hill"]:
            # convert the elasticity tensor via the Total-Lagrange framework
            m = self.strain_exponent
            dsde = _seth_hill_elasticity(dxdX, decomposition, stress_new, dsde, m)

        return [dsde]
