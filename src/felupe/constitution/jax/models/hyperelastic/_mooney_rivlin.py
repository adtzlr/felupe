# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import trace
from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import mooney_rivlin as mooney_rivlin_docstring


@wraps(mooney_rivlin_docstring)
def mooney_rivlin(C, C10, C01):
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (I1**2 - J3**2 * trace(C @ C)) / 2
    return C10 * (I1 - 3) + C01 * (I2 - 3)
