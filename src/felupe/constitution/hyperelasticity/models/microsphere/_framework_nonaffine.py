import numpy as np
from tensortrax.math import diagonal, einsum, sqrt
from tensortrax.math import sum as tsum
from tensortrax.math.linalg import det, inv

from .....quadrature import BazantOh


def nonaffine_stretch(C, p, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    # affine stretches
    Ciso = det(C) ** (-1 / 3) * C
    λ = sqrt(einsum("ai,ij...,aj->a...", r, Ciso, r))

    # non-affine stretch
    Λ = einsum("a...,a->...", λ**p, w) ** (1 / p)

    return f(Λ, **kwargs)


def nonaffine_tube(C, q, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine tube part."

    r = quadrature.points
    w = quadrature.weights

    # affine area-stretches
    Ciso = det(C) ** (-1 / 3) * C
    λa = sqrt(einsum("ai,ij...,aj->a...", r, inv(Ciso), r))

    # non-affine tube contraction
    Λt = einsum("a...,a->...", λa**q, w)

    return f(Λt, **kwargs)
