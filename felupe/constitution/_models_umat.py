# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from ..math import (
    trace,
    identity,
    kronecker,
    cdya,
    dya,
    sym,
    ddot,
    sqrt,
    ravel,
    reshape,
)


class UserMaterial:
    """A user-defined material definition with given functions for the (first
    Piola-Kirchhoff) stress tensor and the according fourth-order elasticity
    tensor. Both functions take a list of the deformation gradient and optional
    state variables as the first input argument. The stress-function also
    returns the updated state variables.

    Take this code-block as template:

    ..  code-block::

        def stress(x, **kwargs):
            "First Piola-Kirchhoff stress tensor."

            # extract variables
            F, statevars = x[0], x[-1]

            # user code for (first Piola-Kirchhoff) stress tensor
            P = None

            # update state variables
            statevars_new = None

            return [P, statevars_new]

        def elasticity(x, **kwargs):
            "Fourth-order elasticity tensor."

            # extract variables
            F, statevars = x[0], x[-1]

            # user code for fourth-order elasticity tensor
            # according to the (first Piola-Kirchhoff) stress tensor
            dPdF = None

            return [dPdF]

        umat = UserMaterial(stress, elasticity, **kwargs)

    """

    def __init__(self, stress, elasticity, nstatevars=0, **kwargs):

        self.umat = {"stress": stress, "elasticity": elasticity}
        self.kwargs = kwargs
        self.x = [np.eye(3), np.zeros(nstatevars)]

    def gradient(self, x):
        return self.umat["stress"](x, **self.kwargs)

    def hessian(self, x):
        return self.umat["elasticity"](x, **self.kwargs)


class UserMaterialStrain:
    """A strain-based user-defined material definition with a given functions
    for the stress tensor and the (fourth-order) elasticity tensor.

    Take this code-block (linear-elastic material formulation) as template:

    ..  code-block::

        from felupe.math import kronecker, cdya, dya, trace

        def linear_elastic(δε, σn, ζn, λ, μ):
            '''3D linear-elastic material formulation.

            Arguments
            ---------
            δε : ndarray
                Incremental strain tensor.
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
            δ = kronecker(δε)
            δσ = 2 * μ * δε + λ * trace(δε) * δ

            # update stress and evaluate elasticity tensor
            σ = σn + δσ
            dσdε = 2 * μ * cdya(δ, δ) + λ * dya(δ, δ)

            # update state variables (not used here)
            ζ = ζn

            return dσdε, σ, ζ

        umat = UserMaterialStrain(material=linear_elastic, μ=1, λ=2)

    """

    def __init__(self, material, dim=3, statevars=(0,), **kwargs):

        self.material = material
        self.statevars_shape = statevars
        self.statevars_size = [np.product(shape) for shape in statevars]
        self.statevars_offsets = np.cumsum(self.statevars_size)
        self.nstatevars = sum(self.statevars_size)

        self.kwargs = kwargs

        self.dim = dim
        self.x = [np.eye(dim), np.zeros(2 * dim**2 + self.nstatevars)]

        self.stress = self.gradient
        self.elasticity = self.hessian

    def extract(self, x):
        "Extract the input and evaluate strains, stresses and state variables."

        # unpack deformation gradient F = dx/dX
        dim = self.dim
        F, statevars = x

        # small-strain tensor as eps = sym(dx/dX - 1)
        dudx = F - identity(F)
        strain = sym(dudx)

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

        dsde, stress_new, statevars_new_list = self.material(
            dstrain, stress_old, statevars_old.copy(), **self.kwargs
        )

        strain_new_1d = (strain_old + dstrain).reshape(-1, *strain_old.shape[2:])
        stress_new_1d = stress_new.reshape(-1, *strain_old.shape[2:])

        statevars_new = np.concatenate(
            [*[ravel(sv) for sv in statevars_new_list], strain_new_1d, stress_new_1d],
            axis=0,
        )

        return [stress_new, statevars_new]

    def hessian(self, x):

        strain_old, dstrain, stress_old, statevars_old = self.extract(x)

        dsde = self.material(dstrain, stress_old, statevars_old.copy(), **self.kwargs)[
            :1
        ]

        return dsde


def linear_elastic(δε, σn, ζn, λ=1, μ=1):
    """3D linear-elastic material formulation.

    Arguments
    ---------
    δε : ndarray
        Strain increment.
    σn : ndarray
        Old stress tensor.
    ζn : list
        List of old state variables.
    λ : float
        First Lamé-constant.
    μ : float
        Second Lamé-constant (shear modulus).
    """

    # change of stress due to change of strain
    δ = kronecker(δε)
    δσ = 2 * μ * δε + λ * trace(δε) * δ

    # update stress and evaluate elasticity tensor
    σ = σn + δσ
    dσdε = 2 * μ * cdya(δ, δ) + λ * dya(δ, δ)

    # update state variables (not used here)
    ζ = ζn

    return dσdε, σ, ζ


def linear_elastic_isotropic_harding(δε, σn, ζn, λ=1, μ=1, σy=1, K=0.2):
    """Linear-elastic-plastic material formulation with linear isotropic
    hardening (return mapping algorithm).

    Arguments
    ---------
    δε : ndarray
        Strain increment.
    σn : ndarray
        Old stress tensor.
    ζn : list
        List of old state variables.
    λ : float
        First Lamé-constant.
    μ : float
        Second Lamé-constant (shear modulus).
    σy : float
        Initial yield stress.
    K : float
        Isotropic hardening modulus.
    """

    # elasticity tensor
    δ = kronecker(δε)
    dσdε = λ * dya(δ, δ) + 2 * μ * cdya(δ, δ)

    # elastic hypothetic (trial) stress and deviatoric stress
    σ = σn + ddot(dσdε, δε)
    s = σ - 1 / 3 * trace(σ) * δ

    # unpack old state variables
    α, εp = ζn

    # hypothetic (trial) yield function
    norm_s = sqrt(ddot(s, s))
    f = norm_s - sqrt(2 / 3) * (σy + K * α)

    ζ = ζn

    # check yield function and create a mask where plasticity occurs
    mask = (f > 0)[0]

    # update stress, tangent and state due to plasticity
    if np.any(mask):

        δγ = f / (2 * μ + 2 / 3 * K)
        n = s / norm_s
        εp = εp + δγ * n
        α = α + sqrt(2 / 3) * δγ

        # stress
        σ[..., mask] = (σ - 2 * μ * δγ * n)[..., mask]

        # algorithmic consistent tangent modulus
        dσdε[..., mask] = (
            dσdε
            - 2 * μ / (1 + K / (3 * μ)) * dya(n, n)
            - 2
            * μ
            * δγ
            / norm_s
            * (2 * μ * (cdya(δ, δ) - 1 / 3 * dya(δ, δ)) - 2 * μ * dya(n, n))
        )[..., mask]

        # update list of state variables
        ζ[0][..., mask] = α[..., mask]
        ζ[1][..., mask] = εp[..., mask]

    return dσdε, σ, ζ
