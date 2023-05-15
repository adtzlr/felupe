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

from ..math import cdya_ik, ddot, dya, identity, sqrt, trace


def linear_elastic(dε, εn, σn, ζn, λ, μ, **kwargs):
    """3D linear-elastic material formulation.

    1.  Given state in point x (σn) (valid).

    2.  Given strain increment dε, so that ε = εn + dε.

    3.  Evaluation of the stress σ and the algorithmic consistent tangent modulus dσdε.

        dσdε = λ 1 ⊗ 1 + 2μ 1 ⊙ 1

        σ = σn + dσdε : dε

    Arguments
    ---------
    dε : ndarray
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
    eye = identity(dim=3, shape=(1, 1))
    dσ = 2 * μ * dε + λ * trace(dε) * eye

    # update stress
    σ = σn + dσ

    # evaluate elasticity tensor
    if kwargs["tangent"]:
        dσdε = 2 * μ * cdya_ik(eye, eye) + λ * dya(eye, eye)
    else:
        dσdε = None

    # update state variables (not used here)
    ζ = ζn

    return dσdε, σ, ζ


def linear_elastic_plastic_isotropic_hardening(dε, εn, σn, ζn, λ, μ, σy, K, **kwargs):
    r"""Linear-elastic-plastic material formulation with linear isotropic
    hardening (return mapping algorithm).

    1.  Given state in point x (σn, ζn=[εpn, αn]) (valid).

    2.  Given strain increment dε, so that ε = εn + dε.

    3.  Evaluation of the hypothetic trial state:

        dσdε = λ 1 ⊗ 1 + 2μ 1 ⊙ 1

        σ = σn + dσdε : dε

        s = dev(σ)

        εp = εpn

        α = αn

        f = ||s|| - sqrt(2/3) (σy + K α)

    4.  If f ≤ 0, then elastic step:

            Set y = yn + dy, y=(σ, ζ=[εp, α]),

            algorithmic consistent tangent modulus dσdε.

        Else:

            dγ = f / (2μ + 2/3 K)

            n = s / ||s||

            σ = σ - 2μ dγ n

            εp = εpn + dγ n

            α = αn + sqrt(2 / 3) dγ

            Algorithmic consistent tangent modulus:

            dσdε = dσdε - 2μ / (1 + K / 3μ) n ⊗ n
                 - 2μ dγ / ||s|| ((2μ 1 ⊙ 1 - 1/3 1 ⊗ 1) - 2μ n ⊗ n)

    Arguments
    ---------
    dε : ndarray
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

    eye = identity(dε)

    # elasticity tensor
    if kwargs["tangent"]:
        dσdε = λ * dya(eye, eye) + 2 * μ * cdya_ik(eye, eye)
    else:
        dσdε = None

    # elastic hypothetic (trial) stress and deviatoric stress
    dσ = 2 * μ * dε + λ * trace(dε) * eye
    σ = σn + dσ
    s = σ - 1 / 3 * trace(σ) * eye

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
        dγ = f / (2 * μ + 2 / 3 * K)
        n = s / norm_s
        εp = εp + dγ * n
        α = α + sqrt(2 / 3) * dγ

        # stress
        σ[..., mask] = (σ - 2 * μ * dγ * n)[..., mask]

        # algorithmic consistent tangent modulus
        if kwargs["tangent"]:
            dσdε[..., mask] = (
                dσdε
                - 2 * μ / (1 + K / (3 * μ)) * dya(n, n)
                - 2
                * μ
                * dγ
                / norm_s
                * (
                    2 * μ * (cdya_ik(eye, eye) - 1 / 3 * dya(eye, eye))
                    - 2 * μ * dya(n, n)
                )
            )[..., mask]

        # update list of state variables
        ζ[0][..., mask] = α[..., mask]
        ζ[1][..., mask] = εp[..., mask]

    return dσdε, σ, ζ
