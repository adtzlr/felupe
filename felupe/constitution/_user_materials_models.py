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
    kronecker,
    cdya,
    dya,
    ddot,
    sqrt,
)


def linear_elastic(δε, εn, σn, ζn, λ, μ, **kwargs):
    """3D linear-elastic material formulation.

    Arguments
    ---------
    δε : ndarray
        Strain increment.
    εn : ndarray
        Old strain tensor.
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

    # update stress
    σ = σn + δσ

    # evaluate elasticity tensor
    if kwargs["tangent"]:
        dσdε = 2 * μ * cdya(δ, δ) + λ * dya(δ, δ)
    else:
        dσdε = None

    # update state variables (not used here)
    ζ = ζn

    return dσdε, σ, ζ


def linear_elastic_plastic_isotropic_hardening(δε, εn, σn, ζn, λ, μ, σy, K, **kwargs):
    """Linear-elastic-plastic material formulation with linear isotropic
    hardening (return mapping algorithm).

    Arguments
    ---------
    δε : ndarray
        Strain increment.
    εn : ndarray
        Old strain tensor.
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

    δ = kronecker(δε)

    # elasticity tensor
    if kwargs["tangent"]:
        dσdε = λ * dya(δ, δ) + 2 * μ * cdya(δ, δ)
    else:
        dσdε = None

    # elastic hypothetic (trial) stress and deviatoric stress
    δσ = 2 * μ * δε + λ * trace(δε) * δ
    σ = σn + δσ
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
        if kwargs["tangent"]:
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
