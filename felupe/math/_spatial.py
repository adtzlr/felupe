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


def rotation_matrix(alpha_deg, dim=3, axis=0):
    """Rotation matrix with given rotation axis and dimension (2d or 3d).

    Parameters
    ----------
    alpha_deg : int
        Rotation angle in degree.
    dim : int, optional (default is 3)
        Dimension of the rotation matrix.
    axis : int, optional (default is 0)
        Rotation axis.

    Returns
    -------
    rotation_matrix : ndarray
        Rotation matrix of dim 2 or 3 with given rotation axis.
    """

    a = np.deg2rad(alpha_deg)
    rotation_matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    if dim == 3:
        # rotation_matrix = np.pad(rotation_matrix, (1, 0))
        # rotation_matrix[0, 0] = 1
        rotation_matrix = np.insert(rotation_matrix, [axis], np.zeros((1, 2)), axis=0)
        rotation_matrix = np.insert(rotation_matrix, [axis], np.zeros((3, 1)), axis=1)
        rotation_matrix[axis, axis] = 1

    return rotation_matrix
