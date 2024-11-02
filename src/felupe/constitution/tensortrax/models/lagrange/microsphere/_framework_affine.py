from tensortrax.math import einsum, sqrt, trace
from tensortrax.math.linalg import det, inv

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
