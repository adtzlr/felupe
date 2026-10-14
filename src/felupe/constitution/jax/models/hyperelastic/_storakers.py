# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import array, sqrt
from jax.numpy import sum as asum

from ....tensortrax.models.hyperelastic import storakers as storakers_docstring
from ..._helpers import eigvalsh


@wraps(storakers_docstring)
def storakers(C, mu, alpha, beta):
    λ1, λ2, λ3 = sqrt(eigvalsh(C))
    J = λ1 * λ2 * λ3

    μ = array(mu)
    α = array(alpha)
    β = array(beta)

    return asum(2 * μ / α**2 * (λ1**α + λ2**α + λ3**α - 3 + (J ** (-α * β) - 1) / β))
