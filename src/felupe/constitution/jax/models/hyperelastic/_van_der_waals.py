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
