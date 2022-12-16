# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

from functools import wraps

from tensortrax.math._linalg import det, eigvalsh
from tensortrax.math import trace, sum as sum1


def isochoric_volumetric_split(fun):
    """Apply the material formulation only on the isochoric part of the
    multiplicative split of the deformation gradient."""

    @wraps(fun)
    def apply_iso(C, *args, **kwargs):
        return fun(det(C) ** (-1 / 3) * C, *args, **kwargs)

    return apply_iso


@isochoric_volumetric_split
def neo_hooke(C, mu):
    "Strain energy function of the Neo-Hookean material formulation."
    return mu / 2 * (trace(C) - 3)


@isochoric_volumetric_split
def yeoh(C, C10, C20, C30):
    I1 = trace(C)
    return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3


@isochoric_volumetric_split
def third_order_deformation(C, C10, C01, C11, C20, C30):
    "Strain energy function of the Third-Order-Deformation material formulation."
    I1 = trace(C)
    I2 = (I1**2 - trace(C @ C)) / 2
    return (
        C10 * (I1 - 3)
        + C01 * (I2 - 3)
        + C11 * (I1 - 3) * (I2 - 3)
        + C20 * (I1 - 3) ** 2
        + C30 * (I1 - 3) ** 3
    )


@isochoric_volumetric_split
def ogden(C, mu, alpha):
    "Strain energy function of the Ogden material formulation."
    wC = eigvalsh(C)
    return sum1([2 * m / a**2 * (sum1(wC ** (a / 2)) - 3) for m, a in zip(mu, alpha)])
