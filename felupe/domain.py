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
from numba import jit, prange

from scipy.sparse import csr_matrix

from .helpers import det, inv

class Domain:
    def __init__(self, element, mesh, quadrature):
        self.mesh = mesh
        self.element = element
        self.quadrature = quadrature

        self.ndim = self.element.ndim
        self.nbasis = self.element.nbasis
        self.nnodes = self.mesh.nnodes
        self.nelements = self.mesh.nelements
        self.ndof = self.mesh.ndof

        self.dof = np.arange(self.mesh.nodes.size).reshape(*self.mesh.nodes.shape)

        self.element.h = np.array(
            [self.element.basis(p) for p in self.quadrature.points]
        ).T
        self.element.dhdr = np.array(
            [self.element.basisprime(p) for p in self.quadrature.points]
        ).transpose(1, 2, 0)

        dhdr = np.tile(
            self.element.dhdr.reshape(*self.element.dhdr.shape, 1), self.nelements
        )
        dXdr = np.einsum("eaI,aJge->IJge", mesh.nodes[mesh.connectivity], dhdr)
        drdX = inv(dXdr)

        self.J = det(dXdr)
        self.w = self.quadrature.weights

        self.h = np.tile(
            self.element.h.reshape(*self.element.h.shape, 1), self.nelements
        )
        self.dhdX = np.einsum("aIge,IJge->aJge", dhdr, drdX)

        # indices for sparse matrices
        nd = self.ndim
        eai = np.stack(
            [
                nd * np.tile(conn, (nd, 1)).T + np.arange(nd)
                for conn in self.mesh.connectivity
            ]
        )
        eaibj0 = np.stack([np.repeat(ai.ravel(), ai.size) for ai in eai])
        eaibj1 = np.stack([np.tile(ai.ravel(), ai.size) for ai in eai])

        self.ai = (eai.ravel(), np.zeros_like(eai.ravel()))
        self.aibj = (eaibj0.ravel(), eaibj1.ravel())

        self.dof = np.arange(self.ndof).reshape(self.mesh.nodes.shape)

    def zeros(self, dim=None):
        if dim is None:
            dim = self.ndim
        elif isinstance(dim, tuple):
            return np.zeros((self.mesh.nnodes, *dim))
        else:
            return np.zeros((self.mesh.nnodes, dim))
    
    def fill(self, value, dim=None):
        if dim is None:
            dim = self.ndim
        elif isinstance(dim, tuple):
            return np.ones((self.mesh.nnodes, *dim)) * value
        else:
            return np.ones((self.mesh.nnodes, dim)) * value

    def empty(self, dim=None):
        if dim is None:
            dim = self.ndim
        elif isinstance(dim, tuple):
            return np.empty((self.mesh.nnodes, *dim))
        else:
            return np.empty((self.mesh.nnodes, dim))

    def grad(self, u, dhdX=None):
        if dhdX is None:
            dhdX = self.dhdX
        return np.einsum("ea...,aJge->...Jge", u[self.mesh.connectivity], dhdX)

    def volume(self, detF=1):
        return np.einsum("ge,g->e", detF * self.J, self.w)

    def integrate(self, A, parallel=True):
        if len(A.shape) == 4:
            itg2 = [_integrate2p, _integrate2]
            return itg2[int(parallel)](self.dhdX, A, self.J, self.w)
        elif len(A.shape) == 6:
            itg4 = [_integrate4p, _integrate4]
            return itg4[int(parallel)](self.dhdX, A, self.J, self.w)

    def asmatrix(self, A):
        if len(A.shape) == 3:
            return csr_matrix((A.transpose([2, 0, 1]).ravel(), self.ai))
        elif len(A.shape) == 5:
            return csr_matrix((A.transpose([4, 0, 1, 2, 3]).ravel(), self.aibj))

    def tonodes(self, A, sym=True, mode="tensor"):

        rows = self.mesh.connectivity.T.ravel()
        cols = np.zeros_like(rows)

        if mode == "tensor":
            if sym:
                if self.ndim == 3:
                    ij = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
                    out = self.empty(6)
                elif self.ndim == 2:
                    ij = [(0, 0), (1, 1), (0, 1)]
                    out = self.empty(3)
            else:
                if self.ndim == 3:
                    ij = [
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (1, 0),
                        (1, 1),
                        (1, 2),
                        (2, 0),
                        (2, 1),
                        (2, 2),
                    ]
                    out = self.empty(9)
                elif self.ndim == 2:
                    ij = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    out = self.empty(4)

            for a, (i, j) in enumerate(ij):
                out[:, a] = (
                    csr_matrix(
                        (A.reshape(self.ndim, self.ndim, -1)[i, j], (rows, cols))
                    ).toarray()[:, 0]
                    / self.mesh.elements_per_node
                )

        elif mode == "scalar":
            out = csr_matrix((A.ravel(), (rows, cols))).toarray()[:, 0]
            out = out / self.mesh.elements_per_node

        return out

    # def interpolate(self, A):
    #    return np.einsum("age,ijge->aijge", self.h, A)


def _integrate2(dhdX, P, Jr, w):
    return np.einsum("aJge,iJge,ge,g->aie", dhdX, P, Jr, w)


def _integrate4(dhdX, A, Jr, w):
    return np.einsum("aJge,bLge,iJkLge,ge,g->aibke", dhdX, dhdX, A, Jr, w)


# _integrate2p = _integrate2
# _integrate4p = _integrate4


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _integrate2p(dhdX, P, Jr, w):

    ndim, ngauss, nelems = P.shape[-3:]

    out = np.zeros((ngauss, ndim, nelems))

    for a in prange(ngauss):  # basis function
        for g in prange(ngauss):  # integration points
            for e in prange(nelems):  # element
                for i in prange(ndim):  # row index i
                    for J in prange(ndim):  # column index J
                        out[a, i, e] += (
                            dhdX[a, J, g, e] * P[i, J, g, e] * Jr[g, e] * w[g]
                        )

    return out


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _integrate4p(dhdX, A, Jr, w):

    ndim, ngauss, nelems = A.shape[-3:]

    out = np.zeros((ngauss, ndim, ngauss, ndim, nelems))
    for a in prange(ngauss):
        for b in prange(ngauss):
            for g in prange(ngauss):
                for e in prange(nelems):
                    for i in prange(ndim):
                        for J in prange(ndim):
                            for k in prange(ndim):
                                for L in prange(ndim):
                                    out[a, i, b, k, e] += (
                                        dhdX[a, J, g, e]
                                        * dhdX[b, L, g, e]
                                        * A[i, J, k, L, g, e]
                                        * Jr[g, e]
                                        * w[g]
                                    )

    return out
