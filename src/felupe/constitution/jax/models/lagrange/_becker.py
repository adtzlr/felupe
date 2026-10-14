# SPDX-License-Identifier: GPL-3.0-or-later

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
