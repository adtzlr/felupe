# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import ogden as ogden_docstring
from ..._helpers import eigvalsh


@wraps(ogden_docstring)
def ogden(C, mu, alpha):

    wC = det(C) ** (-1 / 3) * eigvalsh(C)
    return sum([2 * m / a**2 * (sum(wC ** (a / 2)) - 3) for m, a in zip(mu, alpha)])
