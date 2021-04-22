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

# from scipy.sparse.linalg import spsolve
from pypardiso import spsolve


def partition(u, r, K, I, D):
    # uI = u.ravel()[I]
    uD = u.ravel()[D]
    rI = r[I]
    KII = K[I, :][:, I]
    KID = K[I, :][:, D]
    return u, rI, uD, KII, KID, I, D


def solve(u, rI, uD, KII, KID, I, D, uDext):
    drD = KID.dot(uDext - uD)
    duI = spsolve(KII, -rI - drD.reshape(*rI.shape))
    du = np.zeros(u.size)
    du[I] = duI
    du[D] = uDext - uD
    return du.reshape(*u.shape)
