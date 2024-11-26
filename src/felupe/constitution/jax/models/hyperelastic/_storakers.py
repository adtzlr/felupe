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

from jax.numpy import array, diag, sqrt
from jax.numpy import sum as asum
from jax.numpy.linalg import eigvalsh

from ....tensortrax.models.hyperelastic import storakers as storakers_docstring


@wraps(storakers_docstring)
def storakers(C, mu, alpha, beta):
    λ1, λ2, λ3 = sqrt(eigvalsh(C + diag(array([0, -1e-4, 1e-4]))))
    J = λ1 * λ2 * λ3

    μ = array(mu)
    α = array(alpha)
    β = array(beta)

    return asum(2 * μ / α**2 * (λ1**α + λ2**α + λ3**α - 3 + (J ** (-α * β) - 1) / β))
