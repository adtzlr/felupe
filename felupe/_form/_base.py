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
from scipy.sparse import csr_matrix as sparsematrix


class IntegralForm:
    def __init__(self, fun, v, dV, u=None, grad_v=False, grad_u=False):
        self.fun = fun
        self.dV = dV

        self.v = v
        self.grad_v = grad_v

        self.u = u
        self.grad_u = grad_u

        if not self.u:
            self.indices = self.v.indices.ai
            self.shape = self.v.indices.shape

        else:
            eai = self.v.indices.eai
            ebk = self.u.indices.eai

            eaibk0 = np.stack(
                [np.repeat(ai.ravel(), bk.size) for ai, bk in zip(eai, ebk)]
            )
            eaibk1 = np.stack(
                [np.tile(bk.ravel(), ai.size) for ai, bk in zip(eai, ebk)]
            )

            self.indices = (eaibk0.ravel(), eaibk1.ravel())
            self.shape = (self.v.indices.shape[0], self.u.indices.shape[0])

    def assemble(self, values=None, parallel=False):

        if values is None:
            values = self.integrate(parallel=parallel)

        permute = np.append(len(values.shape) - 1, range(len(values.shape) - 1)).astype(
            int
        )

        out = sparsematrix(
            (values.transpose(permute).ravel(), self.indices), shape=self.shape
        )

        return out

    def integrate(self, parallel=False):
        grad_v, grad_u = self.grad_v, self.grad_u
        v, u = self.v, self.u
        dV = self.dV
        fun = self.fun

        # if parallel:
        #    from numba import jit

        if not grad_v:
            vb = np.tile(v.region.h.reshape(*v.region.h.shape, 1), v.region.mesh.ncells)
        else:
            vb = v.region.dhdX

        if u is not None:
            if not grad_u:
                ub = np.tile(
                    u.region.h.reshape(*u.region.h.shape, 1), u.region.mesh.ncells
                )
            else:
                ub = u.region.dhdX

        if u is None:

            if not grad_v:
                return np.einsum("ape,...pe,pe->a...e", vb, fun, dV, optimize=True)
            else:
                if parallel:
                    return integrate_gradv(vb, fun, dV)
                else:
                    return np.einsum(
                        "aJpe,...Jpe,pe->a...e", vb, fun, dV, optimize=True
                    )

        else:

            if not grad_v and not grad_u:
                return np.einsum(
                    "ape,...pe,bpe,pe->a...be", vb, fun, ub, dV, optimize=True
                )
            elif grad_v and not grad_u:
                return np.einsum(
                    "aJpe,iJ...pe,bpe,pe->aib...e", vb, fun, ub, dV, optimize=True
                )
            elif not grad_v and grad_u:
                return np.einsum(
                    "a...pe,...kLpe,bLpe,pe->a...bke", vb, fun, ub, dV, optimize=True
                )
            else:  # grad_v and grad_u
                if parallel:
                    return integrate_gradv_gradu(vb, fun, ub, dV)
                else:
                    return np.einsum(
                        "aJpe,iJkLpe,bLpe,pe->aibke", vb, fun, ub, dV, optimize=True
                    )


try:
    from numba import jit, prange

    jitargs = {"nopython": True, "nogil": True, "fastmath": True, "parallel": True}

    @jit(**jitargs)
    def integrate_gradv_u(v, fun, u, dV):  # pragma: no cover

        npoints_a = v.shape[0]
        npoints_b = u.shape[0]
        ndim1, ndim2, ngauss, ncells = fun.shape

        out = np.zeros((npoints_a, ndim1, npoints_b, ncells))
        for a in prange(npoints_a):  # basis function "a"
            for b in prange(npoints_b):  # basis function "b"
                for p in prange(ngauss):  # integration point "p"
                    for c in prange(ncells):  # cell "c"
                        for i in prange(ndim1):  # first index "i"
                            for J in prange(ndim2):  # second index "J"
                                out[a, i, b, c] += (
                                    v[a, J, p, c]
                                    * u[b, p, c]
                                    * fun[i, J, p, c]
                                    * dV[p, c]
                                )

        return out

    @jit(**jitargs)
    def integrate_v_gradu(v, fun, u, dV):  # pragma: no cover

        npoints_a = v.shape[0]
        npoints_b = u.shape[0]
        ndim1, ndim2, ngauss, ncells = fun.shape

        out = np.zeros((npoints_a, npoints_b, ndim1, ncells))
        for a in prange(npoints_a):  # basis function "a"
            for b in prange(npoints_b):  # basis function "b"
                for p in prange(ngauss):  # integration point "p"
                    for c in prange(ncells):  # cell "c"
                        for k in prange(ndim1):  # third index "k"
                            for L in prange(ndim2):  # fourth index "L"
                                out[a, b, k, c] += (
                                    v[a, p, c]
                                    * u[b, L, p, c]
                                    * fun[k, L, p, c]
                                    * dV[p, c]
                                )

        return out

    @jit(**jitargs)
    def integrate_gradv(v, fun, dV):  # pragma: no cover

        npoints = v.shape[0]
        ndim1, ndim2, ngauss, ncells = fun.shape

        out = np.zeros((npoints, ndim1, ncells))

        for a in prange(npoints):  # basis function "a"
            for p in prange(ngauss):  # integration point "p"
                for c in prange(ncells):  # cell "c"
                    for i in prange(ndim1):  # first index "i"
                        for J in prange(ndim2):  # second index "J"
                            out[a, i, c] += v[a, J, p, c] * fun[i, J, p, c] * dV[p, c]

        return out

    @jit(**jitargs)
    def integrate_gradv_gradu(v, fun, u, dV):  # pragma: no cover

        npoints_a = v.shape[0]
        npoints_b = u.shape[0]
        ndim1, ndim2, ndim3, ndim4, ngauss, ncells = fun.shape

        out = np.zeros((npoints_a, ndim1, npoints_b, ndim3, ncells))
        for a in prange(npoints_a):  # basis function "a"
            for b in prange(npoints_b):  # basis function "b"
                for p in prange(ngauss):  # integration point "p"
                    for c in prange(ncells):  # cell "c"
                        for i in prange(ndim1):  # first index "i"
                            for J in prange(ndim2):  # second index "J"
                                for k in prange(ndim3):  # third index "k"
                                    for L in prange(ndim4):  # fourth index "L"
                                        out[a, i, b, k, c] += (
                                            v[a, J, p, c]
                                            * u[b, L, p, c]
                                            * fun[i, J, k, L, p, c]
                                            * dV[p, c]
                                        )

        return out


except:
    pass
