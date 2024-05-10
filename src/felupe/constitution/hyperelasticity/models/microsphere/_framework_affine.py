import numpy as np
from tensortrax.math import einsum, sqrt
from tensortrax.math import sum as tsum
from tensortrax.math.linalg import det, inv

from ......quadrature import BazantOh


def affine_stretch(C, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    M = np.einsum("ai,aj->aij", r, r)
    affine_stretch = sqrt(einsum("ij,aij->a", C, M))

    return tsum(f(affine_stretch, **kwargs) * w)


def affine_tube(C, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Affine area-stretch part."

    r = quadrature.points
    w = quadrature.weights

    Cs = det(C) * inv(C)
    M = np.einsum("ai,aj->aij", r, r)
    affine_areastretch = sqrt(einsum("ij,aij->a", Cs, M))

    return tsum(f(affine_areastretch, **kwargs) * w)
