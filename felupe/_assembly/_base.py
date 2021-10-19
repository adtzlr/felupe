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
    "Integral (weak) form."

    def __init__(self, fun, v, dV, u=None, grad_v=False, grad_u=False):
        """Integral Form constructed by a function result `fun`,
        a virtual field `v`, differential volumes `dV` and optionally a
        field `u`. For both fields `v` and `u` gradients may be passed by
        setting `grad_v` and `grad_u` to True (default is False for both).


        1) Linearform

            a) without gradient of `v`

                L(v) = ∫ fun v dV                                      (1)

                   (or ∫ fun_i v_i dV)

            b) with gradient of `v`

                L(v) = ∫ fun grad(v) dV                                (2)

                   (or ∫ fun_ij grad(v)_ij dV)


        2) Bilinearform

            a) without gradient of `v` and without gradient of `u`

                b(v, u) = ∫ v fun u dV                                 (3)

                      (or ∫ v_i fun_ij u_j dV)

            b) with gradient of `v` and with gradient of `u`

                b(v, u) = ∫ grad(v) fun grad(u) dV                     (4)

                      (or ∫ grad(v)_ij fun_ijkl grad(u)_kl dV)

            c) with gradient of `v` and without gradient of `u`

                b(v, u) = ∫ grad(v) fun u dV                           (5)

                      (or ∫ grad(v)_ij fun_ijk u_k dV)

            d) without gradient of `v` and with gradient of `u`

                b(v, u) = ∫ v fun grad(u) dV                           (6)

                      (or ∫ v_i fun_ikl grad(u)_kl dV)

        Arguments
        ---------
        fun : array
            The pre-evaluated function.

        v : Field
            The virtual field.

        dV : array
            The differential volumes.

        u : Field, optional (default is None)
            If a field is passed, a Bilinear-Form is created.

        grad_v : bool, optional (default is False)
            Flag to activate the gradient on field `v`.

        grad_u : bool, optional (default is False)
            Flag to activate the gradient on field `u`.

        Methods
        -------
        assemble
            Assembly of sparse region vectors or matrices.

        integrate
            Evaluated (but not assembled) integrals.
        """

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
    def integrate_v_gradu(v, fun, u, dV):  # pragma: no cover

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
    def integrate_gradv(v, fun, dV):  # pragma: no cover

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
    def integrate_gradv_gradu(v, fun, u, dV):  # pragma: no cover

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


except:
    pass
