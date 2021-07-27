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


def values(fields):
    "Return values of a field or a tuple of fields."
    if isinstance(fields, tuple):
        return np.concatenate([field.values.ravel() for field in fields])
    else:
        return fields.values.ravel()


def norms(arrays):
    "Calculate norms of a list of arrays."
    return [np.linalg.norm(arr) for arr in arrays]


def interpolate(A):
    "Interpolate method of field A."
    return A.interpolate()


def grad(A):
    "Gradient method of field A."
    return A.grad()


def sym(A):
    "Symmetric part of matrix A."
    return (A + transpose(A)) / 2


def identity(A=None, ndim=None, shape=None):
    "identity according to matrix A with optional specified dim."
    if A is not None:
        ndimA, g, e = A.shape[-3:]
        if ndim is None:
            ndim = ndimA
    else:
        g, e = shape
    return np.tile(np.eye(ndim), (g, e, 1, 1)).transpose([2, 3, 0, 1])


def dya(A, B, mode=2):
    "Dyadic (outer or kronecker) product of two tensors."
    if mode == 2:
        return np.einsum("ij...,kl...->ijkl...", A, B)
    elif mode == 1:
        return np.einsum("i...,j...->ij...", A, B)
    else:
        raise ValueError("unknown mode. (1 or 2)", mode)


def inv(A, determinant=None, full_output=False, sym=False):
    """ "Inverse of A with optionally pre-calculated determinant,
    optional additional output of the calculated determinant or
    a simplified calculation of the inverse for sym. matrices."""

    detAinvA = np.zeros_like(A)

    # if sym is None:
    #    sym = np.all(A == transpose(A))

    if determinant is None:
        detA = det(A)
    else:
        detA = determinant

    if A.shape[0] == 3:

        detAinvA[0, 0] = -A[1, 2] * A[2, 1] + A[1, 1] * A[2, 2]
        detAinvA[1, 1] = -A[0, 2] * A[2, 0] + A[0, 0] * A[2, 2]
        detAinvA[2, 2] = -A[0, 1] * A[1, 0] + A[0, 0] * A[1, 1]

        detAinvA[0, 1] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
        detAinvA[0, 2] = -A[0, 2] * A[1, 1] + A[0, 1] * A[1, 2]
        detAinvA[1, 2] = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]

        if sym:
            detAinvA[1, 0] = detAinvA[0, 1]
            detAinvA[2, 0] = detAinvA[0, 2]
            detAinvA[2, 1] = detAinvA[1, 2]
        else:
            detAinvA[1, 0] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
            detAinvA[2, 0] = -A[1, 1] * A[2, 0] + A[1, 0] * A[2, 1]
            detAinvA[2, 1] = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]

    elif A.shape[0] == 2:
        detAinvA[0, 0] = A[1, 1]
        detAinvA[0, 1] = -A[0, 1]
        detAinvA[1, 0] = -A[1, 0]
        detAinvA[1, 1] = A[0, 0]

    if full_output:
        return detAinvA / detA, detA
    else:
        return detAinvA / detA


def det(A):
    "Determinant of matrix A."
    if A.shape[0] == 3:
        detA = (
            A[0, 0] * A[1, 1] * A[2, 2]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            - A[2, 0] * A[1, 1] * A[0, 2]
            - A[2, 1] * A[1, 2] * A[0, 0]
            - A[2, 2] * A[1, 0] * A[0, 1]
        )
    elif A.shape[0] == 2:
        detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    elif A.shape[0] == 1:
        detA = A[0, 0]
    return detA


def dev(A):
    "Deviator of matrix A."
    dim = A.shape[0]
    return A - trace(A) / dim * identity(A)


def cof(A):
    "Cofactor matrix of A (as a wrapper for the inverse of A)."
    return transpose(inv(A, determinant=1.0))


def eig(A, eig=np.linalg.eig):
    "Eigenvalues and -vectors of matrix A."
    wA, vA = eig(A.transpose([2, 3, 0, 1]))
    return wA.transpose([2, 0, 1]), vA.transpose([2, 3, 0, 1])


def eigh(A):
    "Eigenvalues and -vectors of a symmetric matrix A."
    return eig(A, eig=np.linalg.eigh)


def eigvals(A, shear=False, eig=np.linalg.eig):
    "Eigenvalues (and optional principal shear values) of a matrix A."
    wA = eig(A.transpose([2, 3, 0, 1]))[0].transpose([2, 0, 1])
    if shear:
        ndim = wA.shape[0]
        if ndim == 3:
            ij = [(1, 0), (2, 0), (2, 1)]
        elif ndim == 2:
            ij = [(1, 0)]
        dwA = np.array([wA[i] - wA[j] for i, j in ij])
        return np.vstack((wA, dwA))
    else:
        return wA


def eigvalsh(A, shear=False):
    "Eigenvalues (and optional principal shear values) of a symmetric matrix A."
    return eigvals(A, shear=shear, eig=np.linalg.eigh)


def transpose(A, mode=1):
    "Transpose (mode=1) or major-transpose (mode=2) of matrix A."
    if mode == 1:
        return A.transpose([1, 0, 2, 3])
    elif mode == 2:
        return np.einsum("ijkl...->klij...", A, optimize=True)
    else:
        raise ValueError("Unknown value of mode.")


def majortranspose(A):
    return transpose(A, mode=2)


def trace(A):
    "The sum of the diagonal elements of A."
    return np.trace(A)


def cdya_ik(A, B):
    "ik - crossed dyadic-product of A and B."
    return np.einsum("ij...,kl...->ikjl...", A, B, optimize=True)


def cdya_il(A, B):
    "il - crossed dyadic-product of A and B."
    return np.einsum("ij...,kl...->ilkj...", A, B, optimize=True)


def cdya(A, B, parallel=True):
    "symmetric - crossed dyadic-product of A and B."
    if parallel:
        return cdya_parallel(A, B)
    else:
        return (cdya_ik(A, B) + cdya_il(A, B)) * 0.5


def dot(A, B):
    "Dot-product of A and B."
    if len(A.shape) == len(B.shape):
        return np.einsum("ik...,kj...->ij...", A, B)
    elif len(A.shape) + 2 == len(B.shape):
        return np.einsum("im...,mjkl...->ijkl...", A, B)
    elif len(A.shape) == len(B.shape) + 2:
        return np.einsum("ijkm...,ml...->ijkl...", A, B)
    else:
        raise TypeError("Unknown shape of A and B.")


def ddot(A, B):
    "Double-Dot-product of A and B."
    if len(A.shape) == len(B.shape):
        return np.einsum("ij...,ij...->...", A, B)
    elif len(A.shape) + 2 == len(B.shape):
        return np.einsum("ij...,ijkl...->kl...", A, B)
    elif len(A.shape) == len(B.shape) + 2:
        return np.einsum("ijkl...,kl...->ij...", A, B)
    else:
        raise TypeError("Unknown shape of A and B.")


def ddot44(A, B):
    "Double-Dot-product of A and B where A and B being two fourth-order tensors."
    return np.einsum("ijkl...,klmn...->ijmn...", A, B, optimize=True)


def ddot444(A, B, C, parallel=True):
    "Double-Dot-product of A, B and C where A, B and C being three fourth-order tensors."
    if parallel:
        return ddot444_parallel(A, B, C)
    else:
        return np.einsum("ijkl...,klmn...,mnpq...->ijpq...", A, B, C, optimize=True)


def tovoigt(A):
    "Convert tensor to voigt notation."
    B = np.zeros((6, *A.shape[-2:]))
    ij = [(0, 0), (1, 1), (2, 2), (0, 0), (1, 2), (0, 2)]
    for i6, (i, j) in enumerate(ij):
        B[i6, :, :] = A[i, j, :, :]
    return B


def laplace(field):
    n = field.dim
    m = field.region.mesh.ndim
    p = field.region.quadrature.npoints
    e = field.region.mesh.ncells

    return identity(grad(field)).reshape(n, m, n, m, p, e)


try:
    from numba import jit, prange

    jitargs = {"nopython": True, "nogil": True, "fastmath": True, "parallel": True}

    @jit(**jitargs)
    def cdya_parallel(A, B):  # pragma: no cover

        ndim, ngauss, nelems = A.shape[-3:]

        out = np.zeros((ndim, ndim, ndim, ndim, ngauss, nelems))

        for i in prange(ndim):
            for j in prange(ndim):
                for k in prange(ndim):
                    for l in prange(ndim):
                        for p in prange(ngauss):
                            for e in prange(nelems):
                                out[i, j, k, l, p, e] += (
                                    A[i, k, p, e] * B[j, l, p, e]
                                    + A[i, l, p, e] * B[k, j, p, e]
                                ) / 2

        return out

    @jit(**jitargs)
    def ddot444_parallel(A, B, C):  # pragma: no cover

        ndim, ngauss, nelems = A.shape[-3:]

        out = np.zeros((ndim, ndim, ndim, ndim, ngauss, nelems))

        for i in prange(ndim):
            for j in prange(ndim):
                for k in prange(ndim):
                    for l in prange(ndim):
                        for m in prange(ndim):
                            for n in prange(ndim):
                                for r in prange(ndim):
                                    for s in prange(ndim):
                                        for p in prange(ngauss):
                                            for e in prange(nelems):
                                                out[i, j, r, s, p, e] += (
                                                    A[i, j, k, l, p, e]
                                                    * B[k, l, m, n, p, e]
                                                    * C[m, n, r, s, p, e]
                                                )

        return out


except:
    pass
