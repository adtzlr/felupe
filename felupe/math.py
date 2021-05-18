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
    if isinstance(fields, tuple):
        return np.concatenate([field.values.ravel() for field in fields])
    else:
        return fields.values.ravel()


def norms(arrays):
    return [np.linalg.norm(arr) for arr in arrays]


def interpolate(A):
    return A.interpolate()


def grad(A):
    return A.grad()


def sym(A):
    return (A + transpose(A)) / 2


def identity(A):
    _, ndim, g, e = A.shape
    return np.tile(np.eye(ndim), (g, e, 1, 1)).transpose([2, 3, 0, 1])


def dya(A, B):
    return np.einsum("ij...,kl...->ijkl...", A, B)


def inv(A):
    return np.linalg.inv(A.T).T


def det(A):
    return np.linalg.det(A.T).T


def cof(A):
    return det(A) * transpose(inv(A))


def eigvals(A, shear=False):
    wA = np.linalg.eigh(A.transpose([2, 3, 0, 1]))[0].transpose([2, 0, 1])
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


def transpose(A):
    return A.transpose([1, 0, 2, 3])


def majortranspose(A):
    return np.einsum("ijkl...->klij...", A)


def trace(A):
    return np.trace(A)


def cdya_ik(A, B):
    return np.einsum("ij...,kl...->ikjl...", A, B)


def cdya_il(A, B):
    return np.einsum("ij...,kl...->ilkj...", A, B)


def cdya(A, B):
    return (cdya_ik(A, B) + cdya_il(A, B)) * 0.5


def dot(A, B):
    if len(A.shape) == len(B.shape):
        return np.einsum("ik...,kj...->ij...", A, B)
    elif len(A.shape)+2 == len(B.shape):
        return np.einsum("ik...,kj...->ij...", A, B)
    elif len(A.shape) == len(B.shape)+2:
        return np.einsum("ijkm...,ml...->ijkl...", A, B)
    else:
        raise TypeError("Unknown shape of A and B.")


def ddot(A, B):
    if len(A.shape) == len(B.shape):
        return np.einsum("ij...,ij...->...", A, B)
    elif len(A.shape)+2 == len(B.shape):
        return np.einsum("ij...,ijkl...->kl...", A, B)
    elif len(A.shape) == len(B.shape)+2:
        return np.einsum("ijkl...,kl...->ij...", A, B)
    else:
        raise TypeError("Unknown shape of A and B.")


def tovoigt(A):
    B = np.zeros((6, *A.shape[-2:]))
    ij = [(0, 0), (1, 1), (2, 2), (0, 0), (1, 2), (0, 2)]
    for i6, (i, j) in enumerate(ij):
        B[i6, :, :] = A[i, j, :, :]
    return B


def tovoigt2(A):
    B = np.zeros((A.shape[0], 6))
    ij = [(0, 0), (1, 1), (2, 2), (0, 0), (1, 2), (0, 2)]
    for i6, (i, j) in enumerate(ij):
        B[:, i6] = A[:, i, j]
    return B
