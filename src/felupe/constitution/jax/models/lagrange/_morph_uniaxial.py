# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import abs as jabs
from jax.numpy import concatenate, exp, maximum, sqrt

from ....tensortrax.models.lagrange import morph_uniaxial as morph_ux


@wraps(morph_ux)
def morph_uniaxial(λ, statevars, p, ε=1e-6):
    CTSn = statevars[:21]
    λn = 1 + statevars[21:42]
    SA1n = statevars[42:63]
    SA2n = statevars[63:84]

    CT = jabs(λ**2 - 1 / λ)
    CTS = maximum(CT, CTSn)

    L1 = 2 * (λ**3 / λn - λn**2) / 3
    L2 = (λn**2 / λ**3 - 1 / λn) / 3
    LT = jabs(L1 - L2)

    sigmoid = lambda x: 1 / sqrt(1 + x**2)
    α = p[0] + p[1] * sigmoid(p[2] * CTS)
    β = p[3] * sigmoid(p[2] * CTS)
    γ = p[4] * CTS * (1 - sigmoid(CTS / p[5]))

    L1_LT = L1 / (ε + LT)
    L2_LT = L2 / (ε + LT)
    CT_CTS = CT / (ε + CTS)

    SL1 = (γ * exp(p[6] * L1_LT * CT_CTS) + p[7] * L1_LT) / λ**2
    SL2 = (γ * exp(p[6] * L2_LT * CT_CTS) + p[7] * L2_LT) * λ

    SA1 = (SA1n + β * LT * SL1) / (1 + β * LT)
    SA2 = (SA2n + β * LT * SL2) / (1 + β * LT)

    dψdλ = (2 * α + SA1) * λ - (2 * α + SA2) / λ**2
    statevars_new = concatenate([CTS, (λ - 1), SA1, SA2])

    return dψdλ, statevars_new
