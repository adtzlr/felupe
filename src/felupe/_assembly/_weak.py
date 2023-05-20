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

import numpy as np

try:
    from einsumt import einsumt
except ModuleNotFoundError:
    from numpy import einsum as einsumt

from scipy.sparse import csr_matrix as sparsematrix


class WeakForm:
    r"""Single-Field Integral-Form constructed by a function result ``fun``,
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
        self.fun = np.ascontiguousarray(fun)
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
            cai = self.v.indices.cai
            cbk = self.u.indices.cai

            caibk0 = np.repeat(cai, cbk.shape[1] * self.u.dim)
            caibk1 = np.tile(cbk, (1, cai.shape[1] * self.v.dim, 1)).ravel()

            self.indices = (caibk0, caibk1)
            self.shape = (self.v.indices.shape[0], self.u.indices.shape[0])

    def assemble(self, values=None, parallel=False):
        "Assembly of sparse region vectors or matrices."

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
                return einsum("aqc,...qc,qc->a...c", vb, fun, dV, optimize=True)
            else:
                return einsum("aJqc,...Jqc,qc->a...c", vb, fun, dV, optimize=True)

        else:
            if not grad_v and not grad_u:
                out = einsum("aqc,...qc,bqc,qc->a...bc", vb, fun, ub, dV, optimize=True)
                if len(out.shape) == 5:
                    return einsum("aijbc->aibjc", out)
                else:
                    return out
            elif grad_v and not grad_u:
                return einsum(
                    "aJqc,iJ...qc,bqc,qc->aib...c", vb, fun, ub, dV, optimize=True
                )
            elif not grad_v and grad_u:
                return einsum(
                    "a...qc,...kLqc,bLqc,qc->a...bkc",
                    vb,
                    fun,
                    ub,
                    dV,
                    optimize=True,
                )
            else:  # grad_v and grad_u
                return einsum(
                    "aJqc,iJkLqc,bLqc,qc->aibkc", vb, fun, ub, dV, optimize=True
                )
