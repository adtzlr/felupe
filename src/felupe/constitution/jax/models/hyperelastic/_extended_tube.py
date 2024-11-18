# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""
from functools import wraps

from jax.numpy import array, diag, log, sqrt, trace
from jax.numpy.linalg import det, eigvalsh

from ....tensortrax.models.hyperelastic import extended_tube as extended_tube_docstring


@wraps(extended_tube_docstring)
def extended_tube(C, Gc, delta, Ge, beta):
    J3 = det(C) ** (-1 / 3)
    D = J3 * trace(C)
    λ1, λ2, λ3 = sqrt(J3 * eigvalsh(C + diag(array([0, 1e-4, -1e-4]))))
    β = beta
    δ = delta
    γ = (1 - δ**2) * (D - 3) / (1 - δ**2 * (D - 3))
    Wc = Gc / 2 * (γ + log(1 - δ**2 * (D - 3)))
    We = 2 * Ge / β**2 * (λ1**-β + λ2**-β + λ3**-β - 3)
    return Wc + We
