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

from jax.numpy import array, concatenate, diag, eye, maximum, sqrt, trace, triu_indices
from jax.numpy.linalg import det, eigvalsh, inv
from jax.scipy.linalg import expm

from ....tensortrax.models.lagrange import morph as morph_docstring
from ..._total_lagrange import total_lagrange


@wraps(morph_docstring)
@total_lagrange
def morph(F, statevars, p):
    # right Cauchy-Green deformation tensor
    C = F.T @ F

    # extract old state variables
    CTSn = statevars[0]
    from_triu = lambda C: C[array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])]
    Cn = from_triu(statevars[1:7])
    SAn = from_triu(statevars[7:13])

    # distortional part of right Cauchy-Green deformation tensor
    I3 = det(C)
    CG = C * I3 ** (-1 / 3)

    # inverse of and incremental right Cauchy-Green deformation tensor
    invC = inv(C)
    dC = C - Cn

    # eigenvalues of right Cauchy-Green deformation tensor (sorted in ascending order)
    eigvalsh2 = lambda C: eigvalsh(C + diag(array([1e-4, -1e-4, 0])))
    λCG = eigvalsh2(CG)

    # Tresca invariant of distortional part of right Cauchy-Green deformation tensor
    CTG = λCG[-1] - λCG[0]

    # maximum Tresca invariant in load history
    CTS = maximum(CTG, CTSn)

    def sigmoid(x):
        "Algebraic sigmoid function."
        return 1 / sqrt(1 + x**2)

    # material parameters
    α = p[0] + p[1] * sigmoid(p[2] * CTS)
    β = p[3] * sigmoid(p[2] * CTS)
    γ = p[4] * CTS * (1 - sigmoid(CTS / p[5]))

    dev = lambda C: C - trace(C) / 3 * eye(3)
    sym = lambda C: (C + C.T) / 2

    LG = sym(dev(invC @ dC)) @ CG
    λLG = eigvalsh2(LG)
    LTG = λLG[-1] - λLG[0]

    # limiting stresses "L" and additional stresses "A"
    SL = (γ * expm(p[6] * LG / LTG * CTG / CTS) + p[7] * LG / LTG) @ invC
    SA = (SAn + β * LTG * SL) / (1 + β * LTG)

    # second Piola-Kirchhoff stress tensor
    S = 2 * α * dev(CG) @ invC + dev(SA @ C) @ invC

    i, j = triu_indices(3)
    to_triu = lambda C: C[i, j]
    statevars_new = concatenate([array([CTS]), to_triu(C), to_triu(SA)])

    return S, statevars_new
