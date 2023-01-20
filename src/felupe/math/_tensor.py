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


def identity(A=None, dim=None, shape=None):
    "Identity according to matrix A with optional specified dim."
    if A is not None:
        dimA, g, e = A.shape[-3:]
        if dim is None:
            dim = dimA
    else:
        g, e = shape
    return np.tile(np.eye(dim), (g, e, 1, 1)).transpose([2, 3, 0, 1])


def sym(A):
    "Symmetric part of matrix A."
    return (A + transpose(A)) / 2


def dya(A, B, mode=2, parallel=False):
    "Dyadic (outer or kronecker) product of two tensors."

    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum

    if mode == 2:
        return einsum("ij...,kl...->ijkl...", A, B)
    elif mode == 1:
        return einsum("i...,j...->ij...", A, B)
    else:
        raise ValueError("unknown mode. (1 or 2)", mode)


def inv(A, determinant=None, full_output=False, sym=False):
    """ "Inverse of A with optionally pre-calculated determinant,
    optional additional output of the calculated determinant or
    a simplified calculation of the inverse for sym. matrices."""

    detAinvA = np.zeros_like(A)

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
        dim = wA.shape[0]
        if dim == 3:
            ij = [(1, 0), (2, 0), (2, 1)]
        elif dim == 2:
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
        return np.einsum("ijkl...->klij...", A)
    else:
        raise ValueError("Unknown value of mode.")


def majortranspose(A):
    return transpose(A, mode=2)


def trace(A):
    "The sum of the diagonal elements of A."
    return np.trace(A)


def cdya_ik(A, B, parallel=False):
    "ik - crossed dyadic-product of A and B."
    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum
    return einsum("ij...,kl...->ikjl...", A, B)


def cdya_il(A, B, parallel=False):
    "il - crossed dyadic-product of A and B."
    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum
    return einsum("ij...,kl...->ilkj...", A, B)


def cdya(A, B, parallel=False):
    "symmetric - crossed dyadic-product of A and B."
    return (cdya_ik(A, B, parallel=parallel) + cdya_il(A, B, parallel=parallel)) * 0.5


def cross(a, b):
    "Cross product of two vectors a and b."
    return np.einsum(
        "...i->i...", np.cross(np.einsum("i...->...i", a), np.einsum("i...->...i", b))
    )


def dot(A, B, n=2, parallel=False):
    "Dot-product of A and B with inputs of n trailing axes.."

    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum

    if len(A.shape) == 2 + n and len(B.shape) == 2 + n:
        return einsum("ik...,kj...->ij...", A, B)

    elif len(A.shape) == 1 + n and len(B.shape) == 1 + n:
        return einsum("i...,i...->...", A, B)

    elif len(A.shape) == 4 + n and len(B.shape) == 4 + n:
        return einsum("ijkp...,plmn...->ijklmn...", A, B)

    elif len(A.shape) == 2 + n and len(B.shape) == 1 + n:
        return einsum("ij...,j...->i...", A, B)

    elif len(A.shape) == 1 + n and len(B.shape) == 2 + n:
        return einsum("i...,ij...->j...", A, B)

    elif len(A.shape) == 4 + n and len(B.shape) == 1 + n:
        return einsum("ijkl...,l...->ijk...", A, B)

    elif len(A.shape) == 1 + n and len(B.shape) == 4 + n:
        return einsum("i...,ijkl...->jkl...", A, B)

    elif len(A.shape) == 2 + n and len(B.shape) == 4 + n:
        return einsum("im...,mjkl...->ijkl...", A, B)

    elif len(A.shape) == 4 + n and len(B.shape) == 2 + n:
        return einsum("ijkm...,ml...->ijkl...", A, B)

    else:
        raise TypeError("Unknown shape of A and B.")


def ddot(A, B, n=2, parallel=False):
    "Double-Dot-product of A and B with inputs of `n` trailing axes."

    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum

    if len(A.shape) == 2 + n and len(B.shape) == 2 + n:
        return einsum("ij...,ij...->...", A, B)
    elif len(A.shape) == 2 + n and len(B.shape) == 4 + n:
        return einsum("ij...,ijkl...->kl...", A, B)
    elif len(A.shape) == 4 + n and len(B.shape) == 2 + n:
        return einsum("ijkl...,kl...->ij...", A, B)
    elif len(A.shape) == 4 + n and len(B.shape) == 4 + n:
        return einsum("ijkl...,klmn...->ijmn...", A, B)
    else:
        raise TypeError("Unknown shape of A and B.")


def tovoigt(A, strain=False):
    "Convert (3, 3) tensor to (6, ) voigt notation."
    dim = A.shape[0]
    if dim == 2:
        B = np.zeros((3, *A.shape[2:]))
        ij = [(0, 0), (1, 1), (0, 1)]
    else:
        B = np.zeros((6, *A.shape[2:]))
        ij = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    for i6, (i, j) in enumerate(ij):
        B[i6] = A[i, j]
    if strain:
        B[dim:] *= 2
    return B


def reshape(A, shape, trailing_axes=2):
    return A.reshape(np.append(shape, A.shape[-trailing_axes:]))


def ravel(A, trailing_axes=2):
    ij, shape = np.split(A.shape, [-trailing_axes])
    return reshape(A, shape=np.product(ij))
