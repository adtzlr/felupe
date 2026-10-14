# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from jax.numpy import trace
from jax.numpy.linalg import det

from ....tensortrax.models.hyperelastic import third_order_deformation as tod_docstring


@wraps(tod_docstring)
def third_order_deformation(C, C10, C01, C11, C20, C30):
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (I1**2 - J3**2 * trace(C @ C)) / 2
    return (
        C10 * (I1 - 3)
        + C01 * (I2 - 3)
        + C11 * (I1 - 3) * (I2 - 3)
        + C20 * (I1 - 3) ** 2
        + C30 * (I1 - 3) ** 3
    )
