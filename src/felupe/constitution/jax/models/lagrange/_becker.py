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

from jax.numpy import einsum, eye, log, sqrt, trace
from jax.numpy.linalg import inv

from ....tensortrax.models.lagrange import becker as becker_docstring
from ..._helpers import eigh
from ..._total_lagrange import total_lagrange


@wraps(becker_docstring)
@total_lagrange
def becker(F, mu, lmbda):
    # right Cauchy-Green deformation tensor C
    C = F.T @ F

    # eigenvalues λC, principal stretches λ and eigenbases M
    # (perturbed relative to the norm of C to separate repeated eigenvalues)
    λC, N = eigh(C)
    M = einsum("ia,ja->aij", N, N)
    λ = sqrt(λC)

    # right stretch tensor U and Lagrangian logarithmic strain tensor E
    U = einsum("a,aij->ij", λ, M)
    E = einsum("a,aij->ij", log(λ), M)

    # Biot stress tensor T and second Piola-Kirchhoff stress tensor S
    T = 2 * mu * E + lmbda * trace(E) * eye(3)
    S = inv(U) @ T

    return S
