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

from numba import jit, prange

jitargs = {"nopython": True, "nogil": True, "fastmath": True, "parallel": True}

@jit(**jitargs)
def gradv_u(v, fun, u, dV):  # pragma: no cover

    npoints_a = v.shape[0]
    npoints_b = u.shape[0]
    dim1, dim2, ngauss, ncells = fun.shape

    out = np.zeros((npoints_a, dim1, npoints_b, ncells))
    for a in prange(npoints_a):  # basis function "a"
        for b in prange(npoints_b):  # basis function "b"
            for p in prange(ngauss):  # integration point "p"
                for c in prange(ncells):  # cell "c"
                    for i in prange(dim1):  # first index "i"
                        for J in prange(dim2):  # second index "J"
                            out[a, i, b, c] += (
                                v[a, J, p, c]
                                * u[b, p, c]
                                * fun[i, J, p, c]
                                * dV[p, c]
                            )

    return out

@jit(**jitargs)
def v_gradu(v, fun, u, dV):  # pragma: no cover

    npoints_a = v.shape[0]
    npoints_b = u.shape[0]
    dim1, dim2, ngauss, ncells = fun.shape

    out = np.zeros((npoints_a, npoints_b, dim1, ncells))
    for a in prange(npoints_a):  # basis function "a"
        for b in prange(npoints_b):  # basis function "b"
            for p in prange(ngauss):  # integration point "p"
                for c in prange(ncells):  # cell "c"
                    for k in prange(dim1):  # third index "k"
                        for L in prange(dim2):  # fourth index "L"
                            out[a, b, k, c] += (
                                v[a, p, c]
                                * u[b, L, p, c]
                                * fun[k, L, p, c]
                                * dV[p, c]
                            )

    return out

@jit(**jitargs)
def gradv(v, fun, dV):  # pragma: no cover

    npoints = v.shape[0]
    dim1, dim2, ngauss, ncells = fun.shape

    out = np.zeros((npoints, dim1, ncells))

    for a in prange(npoints):  # basis function "a"
        for p in prange(ngauss):  # integration point "p"
            for c in prange(ncells):  # cell "c"
                for i in prange(dim1):  # first index "i"
                    for J in prange(dim2):  # second index "J"
                        out[a, i, c] += v[a, J, p, c] * fun[i, J, p, c] * dV[p, c]

    return out

@jit(**jitargs)
def gradv_gradu(v, fun, u, dV):  # pragma: no cover

    npoints_a = v.shape[0]
    npoints_b = u.shape[0]
    dim1, dim2, dim3, dim4, ngauss, ncells = fun.shape

    out = np.zeros((npoints_a, dim1, npoints_b, dim3, ncells))
    for a in prange(npoints_a):  # basis function "a"
        for b in prange(npoints_b):  # basis function "b"
            for p in prange(ngauss):  # integration point "p"
                for c in prange(ncells):  # cell "c"
                    for i in prange(dim1):  # first index "i"
                        for J in prange(dim2):  # second index "J"
                            for k in prange(dim3):  # third index "k"
                                for L in prange(dim4):  # fourth index "L"
                                    out[a, i, b, k, c] += (
                                        v[a, J, p, c]
                                        * u[b, L, p, c]
                                        * fun[i, J, k, L, p, c]
                                        * dV[p, c]
                                    )

    return out