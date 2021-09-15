# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from ..math import (
    dot,
    inv,
    cdya_ik,
    det,
    transpose,
)


class Material:
    def __init__(self, material, parallel=True):
        self.material = material
        self.F = 0
        self.parallel = parallel

    def update(self, F):
        if np.all(F == self.F):
            pass
        else:
            self.F = F
            self.J = det(F)

            self.C = dot(transpose(F), F)
            self.invC = inv(self.C, determinant=self.J ** 2, sym=True)

            self.S = self.material.stress(self.C, self.F, self.J)

    def P(self, F):
        self.update(F)
        return dot(self.F, self.S)

    def A(self, F):
        self.update(F)
        C4 = self.material.elasticity(self.C, self.F, self.J) + cdya_ik(
            self.invC, self.S
        )
        if self.parallel:
            return transform13(self.F, self.F, C4)
        else:
            return np.einsum(
                "iI...,kK...,IJKL...->iJkL...", self.F, self.F, C4, optimize=True
            )


try:
    from numba import jit, prange

    jitargs = {"nopython": True, "nogil": True, "fastmath": True, "parallel": True}

    @jit(**jitargs)
    def transform13(F, G, C4):  # pragma: no cover

        ndim, ngauss, nelems = C4.shape[-3:]

        out = np.zeros((ndim, ndim, ndim, ndim, ngauss, nelems))

        for i in prange(ndim):
            for I in prange(ndim):
                for J in prange(ndim):
                    for k in prange(ndim):
                        for K in prange(ndim):
                            for L in prange(ndim):
                                for p in prange(ngauss):
                                    for e in prange(nelems):
                                        out[i, J, k, L, p, e] += (
                                            F[i, I, p, e]
                                            * G[k, K, p, e]
                                            * C4[I, J, K, L, p, e]
                                        )

        return out

    @jit(**jitargs)
    def transform24(F, G, C4):  # pragma: no cover

        ndim, ngauss, nelems = C4.shape[-3:]

        out = np.zeros((ndim, ndim, ndim, ndim, ngauss, nelems))

        for i in prange(ndim):
            for j in prange(ndim):
                for J in prange(ndim):
                    for k in prange(ndim):
                        for l in prange(ndim):
                            for L in prange(ndim):
                                for p in prange(ngauss):
                                    for e in prange(nelems):
                                        out[i, J, k, L, p, e] += (
                                            F[j, J, p, e]
                                            * G[l, L, p, e]
                                            * C4[i, j, k, l, p, e]
                                        )

        return out


except:
    pass
