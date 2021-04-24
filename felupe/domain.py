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

from copy import deepcopy

import numpy as np
from numba import jit, prange

from scipy.sparse import csr_matrix

from .helpers import det, inv

class Domain:
    def __init__(self, element, mesh, quadrature):
        self.mesh = mesh
        self.element = deepcopy(element)
        self.quadrature = quadrature
        
        # alias
        self.tointegrationpoints = self.interpolate

        self.ndim = self.element.ndim
        self.nbasis = self.element.nbasis
        self.nnodes = self.mesh.nnodes
        self.nelements = self.mesh.nelements
        self.ndof = self.mesh.ndof

        # array with degrees of freedom
        self.dof = np.arange(self.ndof).reshape(*self.mesh.nodes.shape)

        # h_ap
        # ----
        # basis function "a" evaluated at quadrature point "p"
        self.element.h = np.array(
            [self.element.basis(p) for p in self.quadrature.points]
        ).T
        
        # dhdr_aJp
        # --------
        # partial derivative of basis function "a" 
        # w.r.t. natural coordinate "J" evaluated at quadrature point "p"
        self.element.dhdr = np.array(
            [self.element.basisprime(p) for p in self.quadrature.points]
        ).transpose(1, 2, 0)
        
        # dXdr_IJpe
        # ---------
        # geometric gradient as partial derivative of undeformed coordinate "I" 
        # w.r.t. natural coordinate "J" evaluated at quadrature point "p"
        # for every element "e"
        dXdr = np.einsum("eaI,aJp->IJpe", mesh.nodes[mesh.connectivity], 
                         self.element.dhdr)
        drdX = inv(dXdr)

        # det(dXdr)_pe * w_p
        # determinant of geometric gradient evaluated at quadrature point "p"
        # for every element "e" multiplied by corresponding quadrature weight 
        self.Jw = det(dXdr)* self.quadrature.weights.reshape(-1,1)
        
        # dhdX_aJpe
        # ---------
        # partial derivative of basis function "a" 
        # w.r.t. undeformed coordinate "J" evaluated at quadrature point "p"
        # for every element "e"
        self.dhdX = np.einsum("aIp,IJpe->aJpe", self.element.dhdr, drdX)

        # indices for sparse matrices
        # ---------------------------
        eai = np.stack(
            [
                self.ndim * np.tile(conn, (self.ndim, 1)).T + np.arange(self.ndim)
                for conn in self.mesh.connectivity
            ]
        )
        eaibj0 = np.stack([np.repeat(ai.ravel(), ai.size) for ai in eai])
        eaibj1 = np.stack([np.tile(ai.ravel(), ai.size) for ai in eai])

        # export indices as (rows, cols)
        self.ai = (eai.ravel(), np.zeros_like(eai.ravel()))
        self.aibj = (eaibj0.ravel(), eaibj1.ravel())

    def zeros(self, dim=None):
        """Fill dof values with zeros with default dimension 
        identical to the mesh dimension."""
        if dim is None:
            dim = self.ndim
        if isinstance(dim, tuple):
            return np.zeros((self.mesh.nnodes, *dim))
        else:
            return np.zeros((self.mesh.nnodes, dim))
    
    def fill(self, value, dim=None):
        """Fill dof values with custom value with default dimension 
        identical to the mesh dimension."""
        if dim is None:
            dim = self.ndim
        if isinstance(dim, tuple):
            return np.ones((self.mesh.nnodes, *dim)) * value
        else:
            return np.ones((self.mesh.nnodes, dim)) * value
    
    def ones(self, dim=None):
        """Fill dof values with custom value with default dimension 
        identical to the mesh dimension."""
        return self.fill(value=1, dim=dim)

    def empty(self, dim=None):
        """Init an empty array of dof values with default dimension 
        identical to the mesh dimension."""
        if dim is None:
            dim = self.ndim
        if isinstance(dim, tuple):
            return np.empty((self.mesh.nnodes, *dim))
        else:
            return np.empty((self.mesh.nnodes, dim))
        
        
    def interpolate(self, u, h=None):
        "interpolated values u_Ipe"
        # interpolated given nodal values "aI" 
        # evaluated at quadrature point "p"
        # for element "e"
        if h is None:
            h = self.element.h
        return np.einsum("ea...,ap->...pe", u[self.mesh.connectivity], h)


    def grad(self, u, dhdX=None):
        "gradient dudX_IJpe"
        # gradient as partial derivative of given nodal values "aI" 
        # w.r.t. undeformed coordiante "J" evaluated at quadrature point "p"
        # for element "e"
        if dhdX is None:
            dhdX = self.dhdX
        return np.einsum("ea...,aJpe->...Jpe", u[self.mesh.connectivity], dhdX)

    def volume(self, detF=1):
        "Calculate element volume for element 'e'."
        return np.einsum("ge->e", detF * self.Jw)

    def integrate(self, A, parallel=True):
        if len(A.shape) == 4:
            itg2 = [_integrate2parallel, _integrate2]
            return itg2[int(parallel)](self.dhdX, A, self.Jw)
        elif len(A.shape) == 6:
            itg4 = [_integrate4parallel, _integrate4]
            return itg4[int(parallel)](self.dhdX, A, self.Jw)

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



def _integrate2(dhdX, P, Jw):
    return np.einsum("aJge,iJge,ge->aie", dhdX, P, Jw)


def _integrate4(dhdX, A, Jw):
    return np.einsum("aJge,bLge,iJkLge,ge->aibke", dhdX, dhdX, A, Jw)


# remove in future releases
# -------------------------
#_integrate2parallel = _integrate2
#_integrate4parallel = _integrate4


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _integrate2parallel(dhdX, P, Jw):

    ndim, ngauss, nelems = P.shape[-3:]

    out = np.zeros((ngauss, ndim, nelems))

    for a in prange(ngauss):  # basis function "a"
        for p in prange(ngauss):  # integration point "p"
            for e in prange(nelems):  # element "e"
                for i in prange(ndim):  # first index "i"
                    for J in prange(ndim):  # second index "J"
                        out[a, i, e] += (
                            dhdX[a, J, p, e] * P[i, J, p, e] * Jw[p, e]
                        )

    return out


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _integrate4parallel(dhdX, A, Jw):

    ndim, ngauss, nelems = A.shape[-3:]

    out = np.zeros((ngauss, ndim, ngauss, ndim, nelems))
    for a in prange(ngauss):  # basis function "a"
        for b in prange(ngauss):  # basis function "b"
            for p in prange(ngauss):  # integration point "p"
                for e in prange(nelems):  # element "e"
                    for i in prange(ndim):  # first index "i"
                        for J in prange(ndim):  # second index "J"
                            for k in prange(ndim):  # third index "k"
                                for L in prange(ndim):  # fourth index "L"
                                    out[a, i, b, k, e] += (
                                        dhdX[a, J, p, e]
                                        * dhdX[b, L, p, e]
                                        * A[i, J, k, L, p, e]
                                        * Jw[p, e]
                                    )

    return out
