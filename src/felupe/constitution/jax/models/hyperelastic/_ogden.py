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

from jax.numpy import array, diag
from jax.numpy import sum as tsum
from jax.numpy.linalg import det, eigvalsh

from ....tensortrax.models.hyperelastic import ogden as ogden_docstring


@wraps(ogden_docstring)
def ogden(C, mu, alpha):

    wC = det(C) ** (-1 / 3) * eigvalsh(C + diag(array([0, 1e-4, -1e-4])))
    return sum([2 * m / a**2 * (sum(wC ** (a / 2)) - 3) for m, a in zip(mu, alpha)])
