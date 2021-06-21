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


def inv(A, detA=None, full_output=False):
    invA = np.zeros_like(A)

    if detA is None:
        detA = det(A)

    if A.shape[0] == 3:

        invA[0, 0] = -A[1, 2] * A[1, 1] + A[1, 1] * A[2, 2]
        invA[1, 1] = -A[0, 2] * A[0, 2] + A[0, 0] * A[2, 2]
        invA[2, 2] = -A[0, 1] * A[0, 1] + A[0, 0] * A[1, 1]

        invA[0, 1] = A[0, 2] * A[1, 2] - A[0, 1] * A[2, 2]
        invA[1, 2] = A[0, 2] * A[0, 1] - A[0, 0] * A[1, 2]
        invA[0, 2] = -A[0, 2] * A[1, 1] + A[0, 1] * A[1, 2]

        invA[1, 0] = invA[0, 1]
        invA[2, 1] = invA[1, 2]
        invA[2, 0] = invA[0, 2]

    elif A.shape[0] == 2:

        invA[0, 0] = A[1, 1]
        invA[0, 1] = -A[0, 1]
        invA[1, 0] = -A[0, 1]
        invA[1, 1] = A[0, 0]

    if full_output:
        return invA / detA, detA
    else:
        return invA / detA


def det(A):

    if A.shape[0] == 3:

        detA = (
            A[0, 0] * A[1, 1] * A[2, 2]
            + A[0, 1] * A[1, 2] * A[0, 2]
            + A[0, 2] * A[0, 1] * A[1, 2]
            - A[0, 2] * A[1, 1] * A[0, 2]
            - A[1, 2] * A[1, 2] * A[0, 0]
            - A[2, 2] * A[0, 1] * A[0, 1]
        )

    elif A.shape[0] == 2:

        detA = A[0, 0] * A[1, 1] - A[0, 1] ** 2

    return detA


def cof(A):
    return inv(A, detA=1.0)


def eigh(A):
    wA, vA = np.linalg.eigh(A.transpose([2, 3, 0, 1]))
    return wA.transpose([2, 0, 1]), vA.transpose([2, 3, 0, 1])


def eigvalsh(A, shear=False):
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
