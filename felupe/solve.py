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

from scipy.sparse.linalg import spsolve

from .math import values


def partition(v, K, dof1, dof0, r=None):

    u = values(v)
    if r is None:
        r1 = None
    else:
        r1 = r[dof1]

    u0 = u.ravel()[dof0]
    K11 = K[dof1, :][:, dof1]
    K10 = K[dof1, :][:, dof0]
    return u, u0, K11, K10, dof1, dof0, r1


def solve(u, u0, K11, K10, dof1, dof0, r1=None, u0ext=None, solver=spsolve):

    if r1 is None:
        r1 = np.zeros(len(dof1))

    if u0ext is None:
        u0ext = 0
        dr0 = np.zeros(len(dof1))
    else:
        dr0 = K10.dot(u0ext - u0)

    du1 = solver(K11, -r1 - dr0.reshape(*r1.shape))
    du = np.empty(u.size)
    du[dof1] = du1
    du[dof0] = u0ext - u0
    return du.reshape(*u.shape)
