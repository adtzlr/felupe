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

import numpy as np
import tensortrax as tr
from tensortrax.math import log, sqrt
from tensortrax.math import sum as sum1
from tensortrax.math import trace
from tensortrax.math._linalg import det, eigvalsh, inv
from tensortrax.math._special import from_triu_1d, triu_1d


def isochoric_volumetric_split(fun):
    """Apply the material formulation only on the isochoric part of the
    multiplicative split of the deformation gradient."""

    @wraps(fun)
    def apply_iso(C, *args, **kwargs):
        return fun(det(C) ** (-1 / 3) * C, *args, **kwargs)

    return apply_iso


def saint_venant_kirchhoff(C, mu, lmbda):
    """Strain energy function of the Saint Venant-Kirchhoff material formulation.
    Here, ``I1`` and ``I2`` are strain invariants of the Green-Lagrange strain tensor."""
    I1 = trace(C) / 2 - 3 / 2
    I2 = trace(C @ C) / 4 - trace(C) / 2 + 3 / 4
    return mu * I2 + lmbda * I1**2 / 2


@isochoric_volumetric_split
def neo_hooke(C, mu):
    "Strain energy function of the Neo-Hookean material formulation."
    return mu / 2 * (trace(C) - 3)


@isochoric_volumetric_split
def mooney_rivlin(C, C10, C01):
    "Strain energy function of the Mooney-Rivlin material formulation."
    I1 = trace(C)
    I2 = (I1**2 - trace(C @ C)) / 2
    return C10 * (I1 - 3) + C01 * (I2 - 3)


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


@isochoric_volumetric_split
def arruda_boyce(C, C1, limit):
    "Strain energy function of the Arruda-Boyce material formulation."
    I1 = trace(C)

    alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    beta = 1 / limit**2

    out = []
    for i, a in enumerate(alpha):
        j = i + 1
        out.append(a * beta ** (2 * j - 2) * (I1**j - 3**j))

    return C1 * sum1(out)


@isochoric_volumetric_split
def extended_tube(C, Gc, delta, Ge, beta):
    "Strain energy function of the Extended-Tube material formulation."
    D = trace(C)
    wC = eigvalsh(C)
    g = (1 - delta**2) * (D - 3) / (1 - delta**2 * (D - 3))
    Wc = Gc / 2 * (g + log(1 - delta**2 * (D - 3)))
    We = 2 * Ge / beta**2 * sum1(wC ** (-beta / 2) - 1)
    return Wc + We


@isochoric_volumetric_split
def van_der_waals(C, mu, limit, a, beta):
    "Strain energy function of the Van der Waals material formulation."
    I1 = trace(C)
    I2 = (trace(C) ** 2 - trace(C @ C)) / 2
    I = (1 - beta) * I1 + beta * I2
    I.x[np.isclose(I.x, 3)] += 1e-8
    eta = sqrt((I - 3) / (limit**2 - 3))
    return mu * (
        -(limit**2 - 3) * (log(1 - eta) + eta) - 2 / 3 * a * ((I - 3) / 2) ** (3 / 2)
    )


@isochoric_volumetric_split
def finite_strain_viscoelastic(C, Cin, mu, eta, dtime):
    "Finite strain viscoelastic material formulation."

    # update of state variables by evolution equation
    Ci = from_triu_1d(Cin, like=C) + mu / eta * dtime * C
    Ci = det(Ci) ** (-1 / 3) * Ci

    # first invariant of elastic part of right Cauchy-Green deformation tensor
    I1 = trace(C @ inv(Ci))

    # strain energy function and state variable
    return mu / 2 * (I1 - 3), triu_1d(Ci)
