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
from scipy.sparse.linalg import spsolve

from ..math import values


def partition(v, K, dof1, dof0, r=None):
    """Perform partitioning of field values (unknowns), (stiffness) matrix
    and (residuals) vector with given lists of active (dof1) and
    prescribed degrees of freedom (dof0)."""

    # extract values
    u = values(v)

    # check if residuals vector is passed
    if r is None:
        r1 = None
    else:
        # partition residuals vector
        r1 = r[dof1]

    # prescribed dofs of unknowns
    u0 = u.ravel()[dof0]

    # partition (stiffness) matrix
    K11 = K[dof1, :][:, dof1]
    K10 = K[dof1, :][:, dof0]

    return u, u0, K11, K10, dof1, dof0, r1


def solve(u, u0, K11, K10, dof1, dof0, r1=None, ext0=None, solver=spsolve):
    """Linear solution of equation system with optional given values of
    unknowns at prescribed deegrees of freedom.

        K_11 du_1 = -r1 - K10 (u0_ext - u0)

    """

    # init active residuals
    if r1 is None:
        r1 = np.zeros(len(dof1))

    # init external displacements
    if ext0 is None:
        ext0 = 0
        # init inactive dofs of residuals
        dr0 = np.zeros(len(dof1))
    else:
        # evaluate inactive dofs of residuals
        dr0 = K10.dot(ext0 - u0)

    # solve linear system (active dofs)
    du1 = solver(K11, -r1 - dr0.reshape(*r1.shape))

    # full solution
    du = np.empty(u.size)
    du[dof1] = du1
    du[dof0] = ext0 - u0

    # reshape solution to shape of input
    return du.reshape(*u.shape)
