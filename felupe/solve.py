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
    from pypardiso import spsolve
except:
    from scipy.sparse.linalg import spsolve


def partition(u, r, K, dof1, dof0):
    u0 = u.ravel()[dof0]
    r1 = r[dof1]
    K11 = K[dof1, :][:, dof1]
    K10 = K[dof1, :][:, dof0]
    return u, r1, u0, K11, K10, dof1, dof0


def solve(u, r1, u0, K11, K10, dof1, dof0, u0ext):
    dr0 = K10.dot(u0ext - u0)
    du1 = spsolve(K11, -r1 - dr0.reshape(*r1.shape))
    du = np.empty(u.size)
    du[dof1] = du1
    du[dof0] = u0ext - u0
    return du.reshape(*u.shape)
