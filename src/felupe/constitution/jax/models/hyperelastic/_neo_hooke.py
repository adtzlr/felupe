# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import trace
from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import neo_hooke as neo_hooke_docstring


@wraps(neo_hooke_docstring)
def neo_hooke(C, mu):
    return mu / 2 * (det(C) ** (-1 / 3) * trace(C) - 3)
