from tensortrax.math import einsum, sqrt
from tensortrax.math.linalg import det, inv

from .....quadrature import BazantOh


def affine_stretch(C, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    # affine stretches (distortional part)
    λ = det(C) ** (-1 / 6) * sqrt(einsum("ai,ij...,aj->a...", r, C, r))

    return einsum("a...,a->...", f(λ, **kwargs), w)


def affine_tube(C, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Affine area-stretch part."

    r = quadrature.points
    w = quadrature.weights

    # affine area-stretches (distortional part)
    λa = det(C) ** (1 / 6) * sqrt(einsum("ai,ij...,aj->a...", r, inv(C), r))

    return einsum("a...,a->...", f(λa, **kwargs), w)
