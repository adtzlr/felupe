# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import trace
from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import yeoh as yeoh_docstring


@wraps(yeoh_docstring)
def yeoh(C, C10, C20, C30):
    I1 = det(C) ** (-1 / 3) * trace(C)
    return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3
