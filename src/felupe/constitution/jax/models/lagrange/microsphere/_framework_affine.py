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
from jax.numpy import einsum, sqrt, trace
from jax.numpy.linalg import det, inv

from ......quadrature import BazantOh
from ...._total_lagrange import total_lagrange


@total_lagrange
def affine_force_statevars(F, statevars, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Affine force (stretch) part."

    r = quadrature.points
    M = einsum("ai,aj->aij", r, r)
    Mw = einsum("aij,a->aij", M, quadrature.weights)

    # affine stretches (unimodular part)
    J = det(F)
    C = F.T @ F
    λ = J ** (-1 / 3) * sqrt(einsum("ij...,aij->a...", C, M))

    dψdλ, statevars_new = f(λ, statevars, **kwargs)
    dψdE = einsum("a...,aij->ij...", dψdλ / λ, Mw)

    S = J ** (-2 / 3) * (dψdE - trace(dψdE @ C) / 3 * inv(C))

    return S, statevars_new
