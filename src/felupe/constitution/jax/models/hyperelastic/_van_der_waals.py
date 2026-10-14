# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import log, sqrt, trace
from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import van_der_waals as van_der_waals_docstring


@wraps(van_der_waals_docstring)
def van_der_waals(C, mu, limit, a, beta):
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (trace(C) ** 2 - J3**2 * trace(C @ C)) / 2
    Im = (1 - beta) * I1 + beta * I2
    Im += 1e-4
    eta = sqrt((Im - 3) / (limit**2 - 3))
    return mu * (
        -(limit**2 - 3) * (log(1 - eta) + eta) - 2 / 3 * a * ((Im - 3) / 2) ** (3 / 2)
    )
