# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import warnings

import numpy as np

from ...math import cdya_ik, dot, identity, ravel, reshape, svd, sym, transpose
from .._base import ConstitutiveMaterial
from .._material import takes_time_increment


def _corotational_elasticity(decomposition, stress, elasticity):
    r"""Return the fourth-order elasticity tensor of the co-rotational framework, i.e.
    the derivative of the first Piola-Kirchhoff stress tensor
    :math:`\boldsymbol{P} = \boldsymbol{R}\ \boldsymbol{T}` w.r.t. the deformation
    gradient, for a given singular value decomposition of the deformation gradient, the
    (symmetric) stress tensor :math:`\boldsymbol{T}` and the (minor-symmetric)
    elasticity tensor :math:`\mathbb{C} = \partial \boldsymbol{T} / \partial
    \boldsymbol{U}` of the material, see :class:`~felupe.MaterialStrain`."""

    Q, λ, Vh = decomposition
    V = transpose(Vh)
    eye = np.eye(len(λ))

    # stress and elasticity tensors in the principal axes of the stretch tensor
    T = np.einsum("Ia...,IJ...,Jb...->ab...", V, stress, V)
    C = np.einsum(
        "Ia...,Jb...,Kc...,Ld...,IJKL...->abcd...",
        V,
        V,
        V,
        V,
        elasticity,
        optimize=True,
    )

    h = 1 / (λ[:, None] + λ[None, :])  # 1 / (λ_a + λ_b)
    g = λ[:, None] * h  # λ_a / (λ_a + λ_b)

    # material and geometric parts in the principal axes
    A = 2 * g[None, None] * C
    A += np.einsum("ac,ad...,db...->abcd...", eye, h, T)
    A -= np.einsum("ad,ac...,cb...->abcd...", eye, h, T)

    return np.einsum(
        "ia...,Jb...,kc...,Ld...,abcd...->iJkL...", Q, V, Q, V, A, optimize=True
    )


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
        "total-lagrange" and "co-rotational" are supported. Default is "small-strain".
    symmetry : bool, optional
        Take the symmetric part of the returned stress and the minor symmetric-parts of
        the algorithmic consistent elasticity tensor. Default is True. May enhance
        performance if the material returns symmetric tensors.
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

    The co-rotational framework is based on the polar decomposition of the deformation
    gradient into the rotation tensor :math:`\boldsymbol{R}` and the right stretch tensor
    :math:`\boldsymbol{U}`, see Eq. :eq:`corotational-polar-decomposition`. It is
    evaluated by the singular value decomposition of the deformation gradient with the
    principal stretches :math:`\lambda_a`.

    ..  math::
        :label: corotational-polar-decomposition

        \boldsymbol{F} = \boldsymbol{R}\ \boldsymbol{U}
            = \boldsymbol{Q}\ \boldsymbol{\Lambda}\ \boldsymbol{V}^T
        \qquad \boldsymbol{R} = \boldsymbol{Q}\ \boldsymbol{V}^T
        \qquad \boldsymbol{U} = \boldsymbol{V}\ \boldsymbol{\Lambda}\ \boldsymbol{V}^T

    The co-rotational framework uses the Biot strain tensor, see Eq.
    :eq:`corotational-biot-strain`.

    ..  math::
        :label: corotational-biot-strain

        \boldsymbol{\varepsilon} = \boldsymbol{U} - \boldsymbol{1}

    The stress tensor :math:`\boldsymbol{T}` of the material is converted to the first
    Piola-Kirchhoff stress tensor, see Eq. :eq:`corotational-pk1-stress`.

    ..  math::
        :label: corotational-pk1-stress

        \boldsymbol{P} = \boldsymbol{R}\ \boldsymbol{T}

    The fourth-order elasticity tensor of the co-rotational framework is the exact
    linearization of the first Piola-Kirchhoff stress tensor, i.e. Newton's method
    converges quadratically. It is evaluated by the components of the stress tensor
    :math:`\hat{T}_{ab} = V_{Ia}\ T_{IJ}\ V_{Jb}` and of the elasticity tensor
    :math:`\hat{\mathbb{C}}_{abcd} = V_{Ia}\ V_{Jb}\ V_{Kc}\ V_{Ld}\ \mathbb{C}_{IJKL}` of
    the material in the principal axes of the stretch tensor, see Eq.
    :eq:`corotational-elasticity-principal` (without a summation over the indices of
    the principal axes), and it is converted by Eq. :eq:`corotational-elasticity`.

    ..  math::
        :label: corotational-elasticity-principal

        \hat{\mathbb{A}}_{abcd} = \frac{2 \lambda_c}{\lambda_c + \lambda_d}\
            \hat{\mathbb{C}}_{abcd}
            + \delta_{ac}\ \frac{\hat{T}_{db}}{\lambda_a + \lambda_d}
            - \delta_{ad}\ \frac{\hat{T}_{cb}}{\lambda_a + \lambda_c}

    ..  math::
        :label: corotational-elasticity

        \mathbb{A}_{iJkL} = Q_{ia}\ V_{Jb}\ Q_{kc}\ V_{Ld}\ \hat{\mathbb{A}}_{abcd}

    ..  note::

        The first Piola-Kirchhoff stress tensor of Eq. :eq:`corotational-pk1-stress`
        is work-conjugate to the Biot strain tensor only if the stress tensor of the
        material is coaxial with the stretch tensor, e.g. for isotropic elastic
        materials.

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

    See Also
    --------
    linear_elastic : 3D linear-elastic material formulation
    linear_elastic_plastic_isotropic_hardening : Linear-elastic-plastic material
        formulation with linear isotropic hardening (return mapping algorithm).
    LinearElasticPlasticIsotropicHardening : Linear-elastic-plastic material formulation
        with linear isotropic hardening (return mapping algorithm).

    """

    def __init__(
        self,
        material,
        dim=3,
        statevars=(0,),
        framework="small-strain",
        symmetry=True,
        **kwargs,
    ):
        self.material = material
        self.statevars_shape = statevars
        self.statevars_size = [np.prod(shape) for shape in statevars]
        self.statevars_offsets = np.cumsum(self.statevars_size)
        self.nstatevars = sum(self.statevars_size)

        self.framework = framework
        self.symmetry = symmetry

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

        if decomposition is not None:
            rotation = dot(decomposition.U, decomposition.Vh)

        return (*values, rotation)

    def _extract(self, x):
        """Extract the input and evaluate strains, stresses and state variables. Instead
        of the rotation tensor, the singular value decomposition of the deformation
        gradient is returned for the co-rotational framework (else None)."""

        # unpack deformation gradient F = dx/dX
        dim = self.dim
        dxdX, statevars = x

        decomposition = None

        if self.framework == "small-strain":
            strain = sym(dxdX - identity(dxdX))

        elif self.framework == "total-lagrange":
            strain = (dot(transpose(dxdX), dxdX) - identity(dxdX)) / 2

        elif self.framework == "co-rotational":
            decomposition = svd(dxdX)
            rotation = dot(decomposition.U, decomposition.Vh)
            stretch = dot(transpose(rotation), dxdX)
            strain = stretch - identity(dxdX)

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

        if self.framework == "total-lagrange":
            # convert second to first Piola-Kirchhoff stress
            stress_new = dot(dxdX, stress_new)

        if self.framework == "co-rotational":
            rotation = dot(decomposition.U, decomposition.Vh)
            stress_new = dot(rotation, stress_new)

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
            dsde = np.einsum("iI...,kK...,IJKL...->iJkL...", dxdX, dxdX, dsde)

            geometric = cdya_ik(np.eye(self.dim), stress_new)
            dsde = np.sum(np.broadcast_arrays(dsde, geometric), axis=0)

        elif self.framework == "co-rotational":
            dsde = _corotational_elasticity(decomposition, stress_new, dsde)

        return [dsde]
