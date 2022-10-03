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

try:
    from einsumt import einsumt
except:
    from numpy import einsum as einsumt
from scipy.sparse import csr_matrix as sparsematrix


class IntegralForm:
    r"""Integral Form constructed by a function result ``fun``,
    a virtual field ``v``, differential volumes ``dV`` and optionally a
    field ``u``. For both fields ``v`` and ``u`` gradients may be passed by
    setting ``grad_v`` and ``grad_u`` to True (default is False for both).

    **Linearform**

    without gradient of ``v``

    ..  code-block::

        L(v) = ∫ fun v dV                                      (1)

           (or ∫ fun_i v_i dV)


    with gradient of ``v``

    ..  code-block::

        L(v) = ∫ fun grad(v) dV                                (2)

           (or ∫ fun_ij grad(v)_ij dV)


    **Bilinearform**

    without gradient of ``v`` and without gradient of ``u``

    ..  code-block::

        b(v, u) = ∫ v fun u dV                                 (3)

              (or ∫ v_i fun_ij u_j dV)

    with gradient of ``v`` and with gradient of ``u``

    ..  code-block::

        b(v, u) = ∫ grad(v) fun grad(u) dV                     (4)

              (or ∫ grad(v)_ij fun_ijkl grad(u)_kl dV)

    with gradient of ``v`` and without gradient of ``u``

    ..  code-block::

        b(v, u) = ∫ grad(v) fun u dV                           (5)

              (or ∫ grad(v)_ij fun_ijk u_k dV)

    without gradient of ``v`` and with gradient of ``u``

    ..  code-block::

        b(v, u) = ∫ v fun grad(u) dV                           (6)

              (or ∫ v_i fun_ikl grad(u)_kl dV)

    Arguments
    ---------
    fun : array
        The pre-evaluated function.
    v : Field
        The virtual Field.
    dV : array
        The differential volumes.
    u : Field, optional (default is None)
        If a Field is passed, a Bilinear-Form is created.
    grad_v : bool, optional (default is False)
        Flag to activate the gradient on Field ``v``.
    grad_u : bool, optional (default is False)
        Flag to activate the gradient on Field ``u``.
    """

    def __init__(self, fun, v, dV, u=None, grad_v=False, grad_u=False):
        self.fun = fun
        self.dV = dV

        self.v = v
        self.grad_v = grad_v

        self.u = u
        self.grad_u = grad_u

        # init indices

        # # linear form
        if not self.u:
            self.indices = self.v.indices.ai
            self.shape = self.v.indices.shape

        # # bilinear form
        else:
            eai = self.v.indices.eai
            ebk = self.u.indices.eai

            eaibk0 = np.repeat(eai, ebk.shape[1] * self.u.dim)
            eaibk1 = np.tile(ebk, (1, eai.shape[1] * self.v.dim, 1)).ravel()

            self.indices = (eaibk0, eaibk1)
            self.shape = (self.v.indices.shape[0], self.u.indices.shape[0])

    def assemble(self, values=None, parallel=False, jit=False):
        "Assembly of sparse region vectors or matrices."

        if values is None:
            values = self.integrate(parallel=parallel, jit=jit)

        permute = np.append(len(values.shape) - 1, range(len(values.shape) - 1)).astype(
            int
        )

        out = sparsematrix(
            (values.transpose(permute).ravel(), self.indices), shape=self.shape
        )

        return out

    def integrate(self, parallel=False, jit=False):
        "Return evaluated (but not assembled) integrals."

        grad_v, grad_u = self.grad_v, self.grad_u
        v, u = self.v, self.u
        dV = self.dV
        fun = self.fun

        # plane strain
        # trim 3d vector-valued functions to the dimension of the field
        function_dimension = len(fun.shape) - 2
        function_is_vector = function_dimension >= 1
        function_is_3d = len(fun) == 3
        field_is_2d = v.dim == 2

        if function_is_vector and function_is_3d and field_is_2d:
            fun = fun[tuple([slice(2)] * function_dimension)]

        if parallel:
            einsum = einsumt
        else:
            einsum = np.einsum

        if not grad_v:
            vb = v.region.h
        else:
            vb = v.region.dhdX

        if u is not None:
            if not grad_u:
                ub = u.region.h
            else:
                ub = u.region.dhdX

        if u is None:

            if not grad_v:
                return einsum("ape,...pe,pe->a...e", vb, fun, dV, optimize=True)
            else:
                if jit:
                    if fun.shape[-2:] == (1, 1):
                        return integrate_gradv_broadcast(vb, fun, dV)
                    else:
                        return integrate_gradv(vb, fun, dV)
                else:
                    return einsum("aJpe,...Jpe,pe->a...e", vb, fun, dV, optimize=True)

        else:

            if not grad_v and not grad_u:
                out = einsum("ape,...pe,bpe,pe->a...be", vb, fun, ub, dV, optimize=True)
                if len(out.shape) == 5:
                    return einsum("aijbe->aibje", out)
                else:
                    return out
            elif grad_v and not grad_u:
                return einsum(
                    "aJpe,iJ...pe,bpe,pe->aib...e", vb, fun, ub, dV, optimize=True
                )
            elif not grad_v and grad_u:
                return einsum(
                    "a...pe,...kLpe,bLpe,pe->a...bke",
                    vb,
                    fun,
                    ub,
                    dV,
                    optimize=True,
                )
            else:  # grad_v and grad_u
                if jit:
                    if fun.shape[-2:] == (1, 1):
                        return integrate_gradv_gradu_broadcast(vb, fun, ub, dV)
                    else:
                        return integrate_gradv_gradu(vb, fun, ub, dV)
                else:
                    return einsum(
                        "aJpe,iJkLpe,bLpe,pe->aibke", vb, fun, ub, dV, optimize=True
                    )


try:
    from numba import jit as numba_jit, prange

    jitargs = {"nopython": True, "nogil": True, "fastmath": True, "parallel": True}

    @numba_jit(**jitargs)
    def integrate_gradv_broadcast(v, fun, dV):  # pragma: no cover

        npoints = v.shape[0]
        ngauss, ncells = v.shape[-2:]
        dim1, dim2 = fun.shape[:-2]

        out = np.zeros((npoints, dim1, ncells))

        for a in prange(npoints):  # basis function "a"
            for p in range(ngauss):  # integration point "p"
                for c in prange(ncells):  # cell "c"
                    for i in prange(dim1):  # first index "i"
                        for J in range(dim2):  # second index "J"
                            out[a, i, c] += v[a, J, p, c] * fun[i, J, 0, 0] * dV[p, c]

        return out

    @numba_jit(**jitargs)
    def integrate_gradv(v, fun, dV):  # pragma: no cover

        npoints = v.shape[0]
        ngauss, ncells = v.shape[-2:]
        dim1, dim2 = fun.shape[:-2]

        out = np.zeros((npoints, dim1, ncells))

        for a in prange(npoints):  # basis function "a"
            for p in range(ngauss):  # integration point "p"
                for c in prange(ncells):  # cell "c"
                    for i in prange(dim1):  # first index "i"
                        for J in range(dim2):  # second index "J"
                            out[a, i, c] += v[a, J, p, c] * fun[i, J, p, c] * dV[p, c]

        return out

    @numba_jit(**jitargs)
    def integrate_gradv_gradu_broadcast(v, fun, u, dV):  # pragma: no cover

        npoints_a = v.shape[0]
        npoints_b = u.shape[0]
        ngauss, ncells = v.shape[-2:]
        dim1, dim2, dim3, dim4 = fun.shape[:-2]

        out = np.zeros((npoints_a, dim1, npoints_b, dim3, ncells))

        for p in range(ngauss):  # integration point "p"
            for c in prange(ncells):  # cell "c"
                for a in prange(npoints_a):  # basis function "a"
                    for b in prange(npoints_b):  # basis function "b"
                        for i in prange(dim1):  # first index "i"
                            for J in range(dim2):  # second index "J"
                                for k in prange(dim3):  # third index "k"
                                    for L in range(dim4):  # fourth index "L"
                                        out[a, i, b, k, c] += (
                                            v[a, J, p, c]
                                            * u[b, L, p, c]
                                            * fun[i, J, k, L, 0, 0]
                                            * dV[p, c]
                                        )

        return out

    @numba_jit(**jitargs)
    def integrate_gradv_gradu(v, fun, u, dV):  # pragma: no cover

        npoints_a = v.shape[0]
        npoints_b = u.shape[0]
        ngauss, ncells = v.shape[-2:]
        dim1, dim2, dim3, dim4 = fun.shape[:-2]

        out = np.zeros((npoints_a, dim1, npoints_b, dim3, ncells))

        for p in range(ngauss):  # integration point "p"
            for c in prange(ncells):  # cell "c"
                for a in prange(npoints_a):  # basis function "a"
                    for b in prange(npoints_b):  # basis function "b"
                        for i in prange(dim1):  # first index "i"
                            for J in range(dim2):  # second index "J"
                                for k in prange(dim3):  # third index "k"
                                    for L in range(dim4):  # fourth index "L"
                                        out[a, i, b, k, c] += (
                                            v[a, J, p, c]
                                            * u[b, L, p, c]
                                            * fun[i, J, k, L, p, c]
                                            * dV[p, c]
                                        )

        return out

except:
    pass
