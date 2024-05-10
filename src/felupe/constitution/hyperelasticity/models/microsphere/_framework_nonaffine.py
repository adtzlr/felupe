import numpy as np
from tensortrax.math import einsum, sqrt
from tensortrax.math import sum as tsum
from tensortrax.math.linalg import det, inv

from ......quadrature import BazantOh


def nonaffine_stretch(C, p, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    M = np.einsum("ai,aj->aij", r, r)
    affine_stretch = sqrt(einsum("ij,aij->a", C, M))
    nonaffine_stretch = tsum(affine_stretch**p * w) ** (1 / p)

    return f(nonaffine_stretch, **kwargs)


def nonaffine_tube(C, q, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine tube part."

    r = quadrature.points
    w = quadrature.weights

    Cs = det(C) * inv(C)
    M = np.einsum("ai,aj->aij", r, r)
    affine_areastretch = sqrt(einsum("ij,aij->a", Cs, M))

    nonaffine_tubecontraction = tsum(affine_areastretch**q * w)
    # nonaffine_areastretch = nonaffine_tube_contraction ** (1 / q)

    return f(nonaffine_tubecontraction, **kwargs)
