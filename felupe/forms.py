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
from scipy.sparse import bmat, vstack

from .field import Field, FieldAxisymmetric


class IntegralFormMixed:
    def __init__(self, fun, v, dV, u=None, grad_v=None, grad_u=None):

        self.fun = fun
        self.v = list(v)
        self.nv = len(self.v)
        self.dV = dV

        if u is not None:
            self.u = list(u)
            self.nu = len(self.u)

        IntForm = {Field: IntegralForm, FieldAxisymmetric: IntegralFormAxisymmetric}[
            type(v[0])
        ]

        if isinstance(v[0], FieldAxisymmetric):
            for i in range(1, len(self.v)):
                self.v[i].radius = self.v[0].radius

        if grad_v is None:
            self.grad_v = np.zeros_like(v, dtype=bool)
            self.grad_v[0] = True
        else:
            self.grad_v = grad_v

        if grad_u is None and u is not None:
            self.grad_u = np.zeros_like(u, dtype=bool)
            self.grad_u[0] = True
        else:
            self.grad_u = grad_u

        self.forms = []

        if self.nv == 1:
            raise ValueError("IntegralFormMixed needs at least 2 fields.")

        if len(fun) == self.nv and u is None:
            # LinearForm
            self.mode = 1
            self.i = np.arange(self.nv)
            self.j = np.zeros_like(self.i)

            for fun, v, grad_v in zip(self.fun, self.v, self.grad_v):
                f = IntForm(fun=fun, v=v, dV=self.dV, grad_v=grad_v)
                self.forms.append(f)

        elif len(fun) == np.sum(1 + np.arange(self.nv)) and u is not None:
            # BilinearForm
            self.mode = 2
            self.i, self.j = np.triu_indices(3)

            for a, (i, j) in enumerate(zip(self.i, self.j)):
                f = IntForm(
                    self.fun[a],
                    v=self.v[i],
                    dV=self.dV,
                    u=self.u[j],
                    grad_v=self.grad_v[i],
                    grad_u=self.grad_u[j],
                )
                self.forms.append(f)
        else:
            raise ValueError("Unknown input format.")

    def assemble(self, values=None, parallel=False, block=True):

        out = []

        if values is None:
            values = [None] * len(self.forms)

        for val, form in zip(values, self.forms):
            out.append(form.assemble(val, parallel))

        if block and self.mode == 2:
            K = np.zeros((self.nv, self.nv), dtype=object)
            for a, (i, j) in enumerate(zip(self.i, self.j)):
                K[i, j] = out[a]
                if i != j:
                    K[j, i] = out[a].T

            return bmat(K).tocsr()

        if block and self.mode == 1:
            return vstack(out).tocsr()

        else:
            return out

    def integrate(self, parallel=False):

        out = []
        for form in self.forms:
            out.append(form.integrate(parallel))

        return out


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
                return np.einsum("ape,ipe,pe->aie", vb, fun, dV, optimize=True)
            else:
                if parallel:
                    return integrate_gradv(vb, fun, dV)
                else:
                    return np.einsum("aJpe,iJpe,pe->aie", vb, fun, dV, optimize=True)

        else:

            if not grad_v and not grad_u:
                return np.einsum(
                    "ape,...pe,bpe,pe->a...be", vb, fun, ub, dV, optimize=True
                )
            elif grad_v and not grad_u:
                if parallel:
                    return integrate_gradv_u(vb, fun, ub, dV)
                    # return np.einsum("aJpe,iJpe,bpe,pe->aibe", vb, fun, ub, dV)
                else:
                    return np.einsum(
                        "aJpe,iJpe,bpe,pe->aibe", vb, fun, ub, dV, optimize=True
                    )
            elif not grad_v and grad_u:
                if parallel:
                    return integrate_v_gradu(vb, fun, ub, dV)
                    # return np.einsum("ape,kLpe,bLpe,pe->abke", vb, fun, ub, dV)
                else:
                    return np.einsum(
                        "ape,kLpe,bLpe,pe->abke", vb, fun, ub, dV, optimize=True
                    )
            else:  # grad_v and grad_u
                if parallel:
                    return integrate_gradv_gradu(vb, fun, ub, dV)
                else:
                    return np.einsum(
                        "aJpe,iJkLpe,bLpe,pe->aibke", vb, fun, ub, dV, optimize=True
                    )


class IntegralFormAxisymmetric(IntegralForm):
    def __init__(self, fun, v, dV, u=None, grad_v=True, grad_u=True):

        R = v.radius
        self.dV = 2 * np.pi * R * dV

        if u is None:

            if isinstance(v, FieldAxisymmetric):

                self.mode = 1

                form_a = IntegralForm(fun[:-1, :-1], v, self.dV, grad_v=True)
                form_b = IntegralForm(fun[(-1,), (-1,)] / R, v.scalar, self.dV)

                self.forms = [form_a, form_b]

            else:

                self.mode = 10

                form_a = IntegralForm(fun, v, self.dV, grad_v=False)
                self.forms = [
                    form_a,
                ]

        else:

            if isinstance(v, FieldAxisymmetric) and isinstance(u, FieldAxisymmetric):

                self.mode = 2

                form_aa = IntegralForm(
                    fun[:-1, :-1, :-1, :-1], v, self.dV, u, True, True
                )
                form_bb = IntegralForm(
                    fun[-1, -1, -1, -1] / R ** 2,
                    v.scalar,
                    self.dV,
                    u.scalar,
                    False,
                    False,
                )
                form_ba = IntegralForm(
                    fun[-1, -1, :-1, :-1] / R, v.scalar, self.dV, u, False, True
                )
                form_ab = IntegralForm(
                    fun[:-1, :-1, -1, -1] / R, v, self.dV, u.scalar, True, False
                )

                self.forms = [form_aa, form_bb, form_ba, form_ab]

            elif isinstance(v, FieldAxisymmetric) and isinstance(u, Field):

                self.mode = 30

                form_a = IntegralForm(fun[:-1, :-1], v, self.dV, u, True, False)
                form_b = IntegralForm(
                    fun[-1, -1] / R, v.scalar, self.dV, u, False, False
                )

                self.forms = [form_a, form_b]

            elif isinstance(v, Field) and isinstance(u, Field):

                self.mode = 40

                form_a = IntegralForm(fun, v, self.dV, u, False, False)

                self.forms = [
                    form_a,
                ]

    def integrate(self, parallel=False):
        values = [form.integrate(parallel=parallel) for form in self.forms]

        if self.mode == 1:
            values[0] += np.pad(values[1], ((0, 0), (1, 0), (0, 0)))
            val = values[0]

        if self.mode == 30:
            a, b, e = values[1].shape
            values[1] = values[1].reshape(a, 1, b, e)

            values[0] += np.pad(values[1], ((0, 0), (1, 0), (0, 0), (0, 0)))
            val = values[0]

        elif self.mode == 2:
            a, b, e = values[1].shape
            values[1] = values[1].reshape(a, 1, b, 1, e)
            values[1] = np.pad(values[1], ((0, 0), (1, 0), (0, 0), (1, 0), (0, 0)))

            a, b, i, e = values[2].shape
            values[2] = values[2].reshape(a, 1, b, i, e)
            values[2] = np.pad(values[2], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))

            a, i, b, e = values[3].shape
            values[3] = values[3].reshape(a, i, b, 1, e)
            values[3] = np.pad(values[3], ((0, 0), (0, 0), (0, 0), (1, 0), (0, 0)))

            for i in range(1, len(values)):
                values[0] += values[i]

            val = values[0]

        elif self.mode == 10 or self.mode == 40:
            val = values[0]

        return val

    def assemble(self, values=None, parallel=False):
        if values is None:
            values = self.integrate(parallel=parallel)
        return self.forms[0].assemble(values)


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
