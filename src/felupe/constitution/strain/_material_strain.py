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

import numpy as np

from ...math import cdya_ik, dot, identity, ravel, reshape, sym, transpose
from .._base import ConstitutiveMaterial


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
    dim : int, optional
        The dimension of the material formulation. Default is 3.
    statevars : tuple of int, optional
        A tuple containing the shape of each state variable. Default is (0, ).
    framework : str, optional
        The framework to be used for the stress and strain formulations. "small-strain"
        and "total-lagrange" are supported. Default is "small-strain".
    symmetry : bool, optional
        Take the symmetric part of the returned stress and the minor symmetric-parts of
        the algorithmic consistent elasticity tensor. Default is True. May enhance
        performance if the material returns symmetric tensors.

    Notes
    -----
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

        self.kwargs = {**kwargs, "tangent": None}

        self.dim = dim
        self.x = [np.eye(dim), np.zeros(2 * dim**2 + self.nstatevars)]

        self.stress = self.gradient
        self.elasticity = self.hessian

    def extract(self, x):
        "Extract the input and evaluate strains, stresses and state variables."

        # unpack deformation gradient F = dx/dX
        dim = self.dim
        dxdX, statevars = x

        if self.framework == "small-strain":
            strain = sym(dxdX - identity(dxdX))

        elif self.framework == "total-lagrange":
            strain = (dot(transpose(dxdX), dxdX) - identity(dxdX)) / 2

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

        return strain_old, dstrain, stress_old, statevars_old

    def gradient(self, x):
        strain_old, dstrain, stress_old, statevars_old = self.extract(x)
        self.kwargs["tangent"] = False

        # unpack deformation gradient F = dx/dX
        dxdX, statevars = x

        dsde, stress_new, statevars_new_list = self.material(
            dstrain, strain_old, stress_old, statevars_old, **self.kwargs
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

        return [stress_new, statevars_new]

    def hessian(self, x):
        strain_old, dstrain, stress_old, statevars_old = self.extract(x)
        self.kwargs["tangent"] = True

        # unpack deformation gradient F = dx/dX
        dxdX, statevars = x

        dsde, stress_new, statevars_new_list = self.material(
            dstrain, strain_old, stress_old, statevars_old, **self.kwargs
        )
        dsde = np.ascontiguousarray(dsde)

        if self.symmetry:

            # enforce minor symmetries on the algorithmic consistent elasticity tensor
            dsde = np.add(dsde, np.einsum("ijlk...->ijkl...", dsde), out=dsde)
            dsde = np.add(dsde, np.einsum("jikl...->ijkl...", dsde), out=dsde)
            dsde = np.multiply(dsde, 0.25, out=dsde)

        if self.framework == "total-lagrange":
            # convert elasticity tensor and add geometric part
            dsde = np.einsum("iI...,kK...,IJKL...->iJkL...", dxdX, dxdX, dsde)

            if self.symmetry:
                stress_new = sym(stress_new)

            geometric = cdya_ik(np.eye(3), stress_new)

            dsde = np.sum(np.broadcast_arrays(dsde, geometric), axis=0)

        return [dsde]
