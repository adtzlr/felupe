from tensortrax.math import einsum, sqrt
from tensortrax.math.linalg import det, inv

from ......quadrature import BazantOh


def nonaffine_stretch(C, p, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    # affine stretches
    λ = sqrt(einsum("ai,ij...,aj->a...", r, C, r))

    # non-affine stretch (distortional part)
    Λ = det(C) ** (-1 / 6) * einsum("a...,a->...", λ**p, w) ** (1 / p)

    return f(Λ, **kwargs)


def nonaffine_tube(C, q, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine tube part."

    r = quadrature.points
    w = quadrature.weights

    # affine area-stretches
    λa = sqrt(einsum("ai,ij...,aj->a...", r, inv(C), r))

    # non-affine tube contraction (distortional part)
    Λt = det(C) ** (q / 6) * einsum("a...,a->...", λa**q, w)

    return f(Λt, **kwargs)
