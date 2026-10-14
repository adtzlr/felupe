# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import log, sqrt, trace
from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import extended_tube as extended_tube_docstring
from ..._helpers import eigvalsh


@wraps(extended_tube_docstring)
def extended_tube(C, Gc, delta, Ge, beta):
    J3 = det(C) ** (-1 / 3)
    D = J3 * trace(C)
    λ1, λ2, λ3 = sqrt(J3 * eigvalsh(C))
    β = beta
    δ = delta
    γ = (1 - δ**2) * (D - 3) / (1 - δ**2 * (D - 3))
    Wc = Gc / 2 * (γ + log(1 - δ**2 * (D - 3)))
    We = 2 * Ge / β**2 * (λ1**-β + λ2**-β + λ3**-β - 3)
    return Wc + We
