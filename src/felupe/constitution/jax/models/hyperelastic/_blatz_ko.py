# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import sqrt, trace
from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import blatz_ko as blatz_ko_docstring


@wraps(blatz_ko_docstring)
def blatz_ko(C, mu):
    I1 = trace(C)
    I2 = (I1**2 - trace(C @ C)) / 2
    I3 = det(C)

    return mu * (I2 / I3 + 2 * sqrt(I3) - 5)
